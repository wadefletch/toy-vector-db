use crate::index::Index;
use crate::point::{Point, Score};
use crate::vector::{distance, Vector};
use rand::Rng;

pub struct HNSWIndex {
    pub points: Vec<Point>, // a list of vectors, indexed by usize
    pub neighbors: Vec<Vec<usize>>,
    // a list of lists. the first index is the level. the second index is the
    // index of the point in the points
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

    fn get_nearest_element(&self, q: &Vector, ep: &Vec<usize>) -> usize {
        let mut min_distance = f32::INFINITY;
        let mut nearest_element = 0;
        for e in ep.clone() {
            let d = distance(q, &self.points[e].vector);
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
            let d = distance(q, &self.points[e].vector);
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

    fn get_top_layer(&self) -> usize {
        self.points.len() - 1
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

    fn add_bidirectional_connections(&self, q: &Point, e: Vec<usize>, lc: usize) {
        let q_index = self.points.len();
        self.points.push(q.clone());
        for e in e {
            self.neighbors[lc].push(e);
            self.neighbors[lc].push(q_index); // should be the index of q in self.points
        }
    }

    /// ALGORITHM 1
    /// Inserts a new point, `q` into the HNSW tree.
    pub fn insert(
        &self,
        q: &Point,              // new element
        M: usize,               // number of established connections
        M_max: usize,           // maximum number of connections for each element per layer
        ef_construction: usize, // size of the dyanmic candidate list
        mL: usize,              // normalization factor for level generation
    ) {
        let mut W = vec![]; // list for the currently found nearest elements
        let mut ep = self.get_enter_point(); // the enter point for the tree
        let L = self.get_top_layer(); // the highest layer of the tree (0 being the lowest)
        let l = self.get_level(mL); // the random layer to which the new element will rise

        // for every layer from the top to one above where the new element will be inserted
        for lc in L..l + 1 {
            // in the current layer, grab the closest element to q
            W = self.search_layer(&q.vector, vec![ep], 1, lc);
            // update the enter point to the closest element to q
            // it seems like this could be omitted, since the length of W is 1 (ef=1)
            ep = self.get_nearest_element(&q.vector, &W);
        }

        // for every layer from either the top or the random layer to which the new element will rise
        for lc in L.min(l)..0 {
            // in the current layer, grab the closest elements to q
            W = self.search_layer(&q.vector, vec![ep], ef_construction, lc);
            // select the elements from this level that should be neighbors to the element being inserted
            // but, aren't the results from search_layer in W already the ef_construction closest elements?
            // maybe M<ef_construction and this is a subset?
            let neighbors = self.select_neighbors_simple(&q.vector, &W, M);
            // create connections between the query vector and the neighbors on layer lc
            self.add_bidirectional_connections(q, neighbors, lc);
            // for each newly connected neighbor
            for e in neighbors {
                // get the neighbors' neighbors
                let e_conn = self.neighbors[e];
                // if the neighbor now has too many neighbors (|e_conn| > M_max)
                if e_conn.len() > M_max {
                    // reselect a properly-sized set of neighbors
                    let e_new_conn =
                        self.select_neighbors_simple(&self.points[e].vector, &e_conn, M_max);
                    self.neighbors[e] = e_new_conn;
                }
            }
        }
    }

    /// ALGORITHM 2
    /// Finds the nearest neighbors (in a given layer) to a given query vector
    pub fn search_layer(
        &self,
        q: &Vector,     // query
        ep: Vec<usize>, // enter points
        ef: usize,      // basically k
        lc: usize,      // layer number
    ) -> Vec<usize> {
        let mut v = ep.clone(); // visited points
        let mut C = ep.clone(); // candidate points
        let mut W = ep.clone(); // nearest neighbors

        // while there are candidate points left to search...
        while C.len() > 0 {
            // get the closest element from the candidate list
            let c = self.get_nearest_element(q, &C);

            // get the furthest element from the current nearest neighbors
            let f = self.get_furthest_element(q, &W);

            // if the closest candidate element is further than the furthest
            // 'nearest' element, we're done. this seems like an optimization
            // that could be removed for clarity, but it's in the original paper
            let dist_to_c = distance(&self.points[c].vector, q);
            let dist_to_f = distance(&self.points[f].vector, q);
            if dist_to_c > dist_to_f {
                break; // all elements have been evaluated
            }

            // for each neighbor of the closest unvisited element to q
            for e in self.neighbors[c].iter().cloned() {
                // if the neighbor hasn't been visited yet
                if !v.contains(&e) {
                    // mark the neighbor as visited
                    v.push(e);
                    // get the furthest element from the query from the current "nearest" neighbors
                    let f = self.get_furthest_element(q, &W);
                    // if the current neighbor is closer to the query vector
                    // than the current worst "nearest" neighbor OR the current
                    // number of "nearest" neighbors is less than the desired
                    // number of returned nearest neighbors
                    let dist_to_e = distance(&self.points[e].vector, q);
                    let dist_to_f = distance(&self.points[f].vector, q);
                    if dist_to_e < dist_to_f || W.len() < ef {
                        // add the neighbor to the candidates
                        C.push(e);
                        // add the neighbor to the "nearest" neighbors
                        W.push(e);
                        // if there are more elements in the "nearest" neighbors
                        // array than can be returned, bump the furthest from q
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
            .iter() // convert the candidate elements vector to an iterable
            .map(|&v| (distance(q, &self.points[v].vector), v)) // map each vector (item) to a tuple of (distance between q<->v, v)
            .collect(); // accumulate the iterable back into a collection
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // sort the distances from lowest to highest
        distances.truncate(M); // truncate the list to the desired number of neighbors
        distances.iter().map(|&(_, v)| v).collect() // return the indicies of the M closest elements
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
            if distance(q, &self.points[e].vector)
                < distance(
                    q,
                    &self.points[self.extract_nearest_element(&self.points[e].vector, &R)].vector,
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
        let mut W = vec![]; // set for the current nearest elements
        let mut ep = self.get_enter_point(); // enter point for the hnsw tree
        let L = self.get_top_layer(); // the highest layer of the tree (layer of enter point)

        // for each layer from the top to one above where the new element will be inserted
        for lc in L..1 {
            W = self.search_layer(q, vec![ep], 1, lc); // get the closest elements to q
            ep = self.get_nearest_element(q, &W); // update the enter point to the closest element to q
        }

        // find the ef nearest elements to q
        W = self.search_layer(q, vec![ep], ef, 0);

        let mut scores: Vec<(Score, &Point)> = W
            .iter()
            .cloned()
            .map(|v| (distance(q, &self.points[v].vector), &self.points[v]))
            .collect();

        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scores.truncate(K);

        scores
    }
}

impl Index for HNSWIndex {
    fn insert(&mut self, point: Point) {
        todo!()
    }

    fn search(&self, query: &Vector, k: usize) -> Vec<(Score, &Point)> {
        todo!()
    }
}
