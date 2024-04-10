use crate::vector::Vector;
use serde::{Deserialize, Serialize};

// Define a Point struct with the pieces of each record we want to hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub name: String,
    pub summary: String,
    #[serde(rename = "text_embeddings")]
    pub vector: Vector,
}

// Define a type alias for the score of a point.
pub type Score = f32;
