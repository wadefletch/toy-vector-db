use std::collections::HashMap;

use crate::index::Index;
use crate::point::{Point, Score};
use crate::vector::{distance, Vector};
use rand::Rng;

#[allow(non_snake_case)]
pub struct HNSWIndex {
    // a list of points, indexed by usize
    pub points: Vec<Point>,

    // vec of layers containing (hashmap of point index -> list of neighbor point indicies)
    pub neighbors: Vec<HashMap<usize, Vec<usize>>>,

    // the enter point (root) of the tree
    ep: usize,

    // the highest layer of the tree (0 being the lowest)
    L: usize,

    // number of established connections
    M: usize,

    // maximum number of connections for each element per layer
    M_max: usize,

    // size of the candidate list at search time
    ef: usize,

    // size of the candidate list at construction time
    ef_construction: usize,

    // normalization factor for layer generation
    mL: f32,
}

// https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW

#[allow(non_snake_case)]
impl HNSWIndex {
    pub fn new() -> Self {
        let ep: usize = 0;

        let L = 4;
        let M = 16;
        let M_max: usize = 2 * M;
        let ef = 100;
        let ef_construction = 2 * ef;
        let mL = (L as f32).ln().recip();

        Self {
            points: vec![],
            neighbors: vec![HashMap::new(); L],
            ep,
            L,
            M,
            M_max,
            ef,
            ef_construction,
            mL,
        }
    }

    fn get_nearest_element(&self, q: &Vector, ep: &Vec<usize>) -> usize {
        let mut min_distance = 1.0; // cosine similarity maximum value
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
        let mut max_distance = -1.0; // cosine similarity minimum value
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

    fn get_layer(&self, mL: f32) -> usize {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let l: usize = (r.ln() * mL).abs().floor() as usize;
        l
    }

    fn extract_nearest_element(&self, q: &Vector, ep: &mut Vec<usize>) -> usize {
        let e = self.get_nearest_element(q, ep);
        // e is a value that should be removed, not an index
        // TODO find a cleaner way to do this.
        let i = ep.iter().position(|x| *x == e).unwrap();
        ep.remove(i);
        e
    }

    fn add_bidirectional_connections(&mut self, q: usize, e: &Vec<usize>, lc: usize) {
        for &e in e {
            self.neighbors[lc].entry(q).or_insert_with(Vec::new).push(e);
            self.neighbors[lc].entry(e).or_insert_with(Vec::new).push(q);
        }
    }

    /// ALGORITHM 1
    /// Inserts a new point, `q` into the HNSW tree.
    pub fn insert(
        &mut self,
        q: &Point, // new element
    ) {
        let mut W: Vec<usize>;
        let mut ep = self.ep.clone(); // the enter point for the tree
        let l = self.get_layer(self.mL); // the random layer to which the new element will rise

        // if the tree is empty, special case
        if self.points.is_empty() {
            self.points.push(q.clone());
            self.neighbors.iter_mut().for_each(|layer| {
                layer.insert(0, vec![]);
            });
            return;
        }

        // add the new element to the tree
        let q_index = self.points.len();
        self.points.push(q.clone());

        // for every layer from the top to one above where the new element will be inserted
        for lc in (l + 1..self.L).rev() {
            // in the current layer, grab the closest element to q
            W = self.search_layer(&q.vector, vec![ep], 1, lc);
            // update the enter point to the closest element to q
            // it seems like this could be omitted, since the length of W is 1 (ef=1)
            ep = self.get_nearest_element(&q.vector, &W);
        }

        // for every layer from either the top or the random layer to which the new element will rise
        for lc in (0..self.L.min(l)).rev() {
            // in the current layer, grab the closest elements to q
            W = self.search_layer(&q.vector, vec![ep], self.ef_construction, lc);
            // select the elements from this layer that should be neighbors to the element being inserted
            // but, aren't the results from search_layer in W already the ef_construction closest elements?
            // maybe M<ef_construction and this is a subset?
            let neighbors = self.select_neighbors_simple(&q.vector, &W, self.M);
            // create connections between the query vector and the neighbors on layer lc
            self.add_bidirectional_connections(q_index, &neighbors, lc);
            // for each newly connected neighbor
            for e in neighbors {
                // get the neighbors' neighbors
                let e_conn = self.neighbors[lc].get(&e).unwrap();
                // if the neighbor now has too many neighbors (|e_conn| > M_max)
                if e_conn.len() > self.M_max {
                    // reselect a properly-sized set of neighbors
                    let e_new_conn =
                        self.select_neighbors_simple(&self.points[e].vector, e_conn, self.M_max);
                    self.neighbors[lc].insert(e, e_new_conn);
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
            let c = self.extract_nearest_element(q, &mut C);

            // get the furthest element from the current nearest neighbors
            let f = self.get_furthest_element(q, &W);

            // if the closest candidate element is further than the furthest
            // 'nearest' element, we're done. this seems like an optimization
            // that could be removed for clarity, but it's in the original paper
            if distance(&self.points[c].vector, q) > distance(&self.points[f].vector, q) {
                break; // all elements have been evaluated
            }

            // for each neighbor of the closest unvisited element to q
            for e in self.neighbors[lc]
                .get(&c)
                .unwrap_or(&vec![])
                .iter()
                .cloned()
            {
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
                            let i = W.iter().position(|y| *y == x).unwrap();
                            W.remove(i);
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
    // fn select_neighbors_heuristic(
    //     &self,
    //     q: &Vector,                    // query vector
    //     C: &Vec<usize>,                // candidate elements
    //     M: usize,                      // number of neighbors to return
    //     lc: usize,                     // layer number
    //     extend_candidates: bool,       // flag indicating whether or not to extend candidate list
    //     keep_pruned_connections: bool, // flag indicating whether or not to add discarded elements
    // ) -> Vec<usize> {
    //     let mut R = vec![];
    //     let mut W = C.clone();

    //     if extend_candidates {
    //         for e in C.iter().cloned() {
    //             for e_adj in self.neighbors[lc].get(&e).unwrap().iter().cloned() {
    //                 if !W.contains(&e_adj) {
    //                     W.push(e_adj);
    //                 }
    //             }
    //         }
    //     }

    //     let mut W_d = vec![];

    //     while W.len() > 0 && R.len() < M {
    //         let e = self.get_nearest_element(q, &W);
    //         if distance(q, &self.points[e].vector)
    //             < distance(
    //                 q,
    //                 &self.points[self.extract_nearest_element(&self.points[e].vector, &mut R)]
    //                     .vector,
    //             )
    //         {
    //             R.push(e);
    //         } else {
    //             W_d.push(e);
    //         }
    //     }

    //     if keep_pruned_connections {
    //         while W_d.len() > 0 && R.len() < M {
    //             R.push(self.extract_nearest_element(q, &mut W_d));
    //         }
    //     }

    //     R
    // }

    /// ALGORITHM 5
    fn search(&self, q: &Vector, K: usize) -> Vec<(Score, &Point)> {
        let mut W: Vec<usize>; // set for the current nearest elements
        let mut ep = self.ep.clone(); // enter point for the hnsw tree

        // for each layer from the top to one above where the new element will be inserted
        for lc in (1..self.L).rev() {
            W = self.search_layer(q, vec![ep], 1, lc); // get the closest elements to q
            ep = self.get_nearest_element(q, &W); // update the enter point to the closest element to q
        }

        // find the ef nearest elements to q
        W = self.search_layer(q, vec![ep], self.ef, 0);

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
        self.insert(&point);
    }

    fn search(&self, query: &Vector, k: usize) -> Vec<(Score, &Point)> {
        self.search(&query, k)
    }
}
