use crate::vector::Vector;
use serde::{Deserialize, Serialize};

// Define a Point struct with the pieces of each record we want to hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub body: String,
    #[serde(rename = "text-embedding-ada-002")]
    pub vector: Vector,
}

impl PartialEq for Point {
    fn eq(&self, other: &Point) -> bool {
        self.vector == other.vector
    }
}

// Define a type alias for the score of a point.
pub type Score = f32;
