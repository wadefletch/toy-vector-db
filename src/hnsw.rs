use crate::index::Index;
use crate::point::{Point, Score};
use crate::vector::{distance, Vector};
use rand::Rng;

pub struct HNSWIndex {
    pub points: Vec<Point>, // a list of points, the indicies of which will be used to represent them in the graph
    pub neighbors: Vec<Vec<usize>>, // a list of lists in which the first index is the point and the returned list are the indicies of adjacent points
}

// https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW

#[allow(non_snake_case)]
impl HNSWIndex {
    pub fn new() -> Self {
        Self {
            points: vec![],
            neighbors: vec![],
        }
    }

    fn get_vector(&self, i: usize) -> &Vector {
        &self.points[i].vector
    }

    fn get_nearest_element(&self, q: &Vector, ep: &Vec<usize>) -> usize {
        let mut min_distance = f32::INFINITY;
        let mut nearest_element = 0;
        for e in ep.clone() {
            let d = distance(q, self.get_vector(e));
            if d < min_distance {
                min_distance = d;
                nearest_element = e;
            }
        }
        nearest_element
    }

    fn get_furthest_element(&self, q: &Vector, ep: &Vec<usize>) -> usize {
        let mut max_distance = 0.0;
        let mut furthest_element = 0;
        for e in ep.iter().cloned() {
            let d = distance(q, self.get_vector(e));
            if d > max_distance {
                max_distance = d;
                furthest_element = e;
            }
        }
        furthest_element
    }

    fn get_enter_point(&self) -> usize {
        todo!()
    }

    fn top_layer(&self) -> usize {
        todo!()
    }

    fn get_level(&self, mL: usize) -> usize {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let l: usize = (r.ln() * mL as f32).floor() as usize;
        l
    }

    fn extract_nearest_element(&self, q: &Vector, ep: &Vec<usize>) -> usize {
        let e = self.get_nearest_element(q, ep);
        ep.remove(e);
        e
    }

    /// ALGORITHM 1
    pub fn insert(
        &self,
        q: &Vector,             // new element
        M: usize,               // number of established connections
        M_max: usize,           // maximum number of connections for each element per layer
        ef_construction: usize, // size of the dyanmic candidate list
        mL: usize,              // normalization factor for level generation
    ) {
        let mut W = vec![];
        let ep = self.get_enter_point();
        let L = self.top_layer();
        let l = self.get_level(mL);

        for lc in L..l + 1 {
            W = self.search_layer(q, ep, 1, lc);
            ep = self.get_nearest_element(q, &W);
        }

        for lc in (L.min(l)..0).rev() {
            W = self.search_layer(q, ep, ef_construction, lc);
            let neighbors = self.select_neighbors_simple(q, &W, M);
            self.add_bidirectional_connections(q, neighbors, lc);
            for e in neighbors {
                let e_conn = self.neighbors[e];
                if e_conn.len() > M_max {
                    let e_new_conn =
                        self.select_neighbors_simple(self.get_vector(e), &e_conn, M_max);
                    self.neighbors[e] = e_new_conn;
                }
            }
        }
    }

    /// ALGORITHM 2
    pub fn search_layer(
        &self,
        q: &Vector, // query
        ep: usize,  // entry point
        ef: usize,  // basically k
        lc: usize,  // layer number
    ) -> Vec<usize> {
        let mut v = vec![ep]; // visited points
        let mut C = vec![ep]; // candidate points
        let mut W = vec![ep]; // nearest neighbors

        while C.len() > 0 {
            let c = self.get_nearest_element(q, &C);
            let f = self.get_furthest_element(q, &W);

            if distance(self.get_vector(c), q) > distance(self.get_vector(f), q) {
                break; // all elements have been evaluated
            }

            for e in self.neighbors[c].iter().cloned() {
                if !v.contains(&e) {
                    v.push(e);
                    let f = self.get_furthest_element(q, &W);
                    if distance(self.get_vector(e), q) > distance(self.get_vector(f), q)
                        || W.len() < ef
                    {
                        C.push(e);
                        W.push(e);
                        if W.len() > ef {
                            let x = self.get_furthest_element(q, &W);
                            W.remove(x);
                        }
                    }
                }
            }
        }

        W
    }

    /// ALGORITHM 3
    fn select_neighbors_simple(
        &self,
        q: &Vector,     // query vector
        C: &Vec<usize>, // candidate elements
        M: usize,       // number of neighbors to return
    ) -> Vec<usize> {
        let mut distances: Vec<(f32, usize)> = C
            .iter()
            .map(|&v| (distance(q, self.get_vector(v)), v))
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(M);
        distances.iter().map(|&(_, v)| v).collect()
    }

    /// ALGORITHM 4
    fn select_neighbors_heuristic(
        &self,
        q: &Vector,                    // query vector
        C: &Vec<usize>,                // candidate elements
        M: usize,                      // number of neighbors to return
        lc: usize,                     // layer number
        extend_candidates: bool,       // flag indicating whether or not to extend candidate list
        keep_pruned_connections: bool, // flag indicating whether or not to add discarded elements
    ) -> Vec<usize> {
        let mut R = vec![];
        let mut W = C.clone();

        if extend_candidates {
            for e in C {
                for e_adj in self.neighbors[e] {
                    if !W.contains(&e_adj) {
                        W.push(e_adj);
                    }
                }
            }
        }

        let mut W_d = vec![];

        while W.len() > 0 && R.len() < M {
            let e = self.get_nearest_element(q, &W);
            if distance(q, self.get_vector(e))
                < distance(
                    q,
                    self.get_vector(self.extract_nearest_element(self.get_vector(e), &R)),
                )
            {
                R.push(e);
            } else {
                W_d.push(e);
            }
        }

        if keep_pruned_connections {
            while W_d.len() > 0 && R.len() < M {
                R.push(self.extract_nearest_element(q, &W_d));
            }
        }

        R
    }

    /// ALGORITHM 5
    fn search(&self, q: &Vector, K: usize, ef: usize) -> Vec<(Score, &Point)> {
        let mut W = vec![];
        let mut ep = self.get_enter_point();
        let L = self.top_layer();

        for lc in L..1 {
            W = self.search_layer(q, ep, 1, lc);
            ep = self.get_nearest_element(q, &W);
        }

        W = self.search_layer(q, ep, ef, 0);

        let mut scores: Vec<(Score, &Point)> = W
            .iter()
            .cloned()
            .map(|v| (distance(q, self.get_vector(v)), &self.points[v]))
            .collect();

        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scores.truncate(K);

        scores
    }
}

#[allow(non_snake_case)]
impl Index for HNSWIndex {
    fn insert(&mut self, point: Point) {
        todo!()
    }

    fn search(&self, query: &Vector, k: usize) -> Vec<(Score, &Point)> {
        todo!()
    }
}
