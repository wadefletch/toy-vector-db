use crate::point::{Point, Score};
use crate::vector::Vector;

pub trait Index {
    fn search(&self, query: &Vector, k: usize) -> Vec<(Score, &Point)>;
    fn insert(&mut self, point: Point);
    fn insert_many(&mut self, points: Vec<Point>) {
        points.into_iter().for_each(|point| self.insert(point));
    }
}
