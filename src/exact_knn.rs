use crate::index::Index;
use crate::point::{Point, Score};
use crate::vector::{distance, Vector};

// Define an Index struct to hold our points. For now, this is just a wrapper
// around a vec of points, but we will add more functionality to optimize our
// searches later.
pub struct ExactKNNIndex {
    pub points: Vec<Point>,
}

impl ExactKNNIndex {
    pub fn new() -> Self {
        Self { points: vec![] }
    }
}

impl Index for ExactKNNIndex {
    fn insert(&mut self, point: Point) {
        self.points.push(point);
    }

    fn insert_many(&mut self, points: Vec<Point>) {
        self.points.extend(points);
    }

    fn search(&self, query: &Vector, k: usize) -> Vec<(Score, &Point)> {
        let mut scores: Vec<(Score, &Point)> = self
            .points
            .iter()
            .map(|v| (distance(&query, &v.vector), v))
            .collect();

        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scores.truncate(k);

        scores
    }
}
