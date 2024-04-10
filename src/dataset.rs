use crate::point::Point;
use std::fs::read_to_string;

// Next, let's read the dataset from the file. As this is an NDJSON (Newline
// Delimited JSON) file, we can use the `lines` method to read each line of the
// file, then use serde_json to parse each line into a `Point` struct.
pub fn read_dataset(filename: &str) -> Vec<Point> {
    read_to_string(filename)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(&line).unwrap())
        .collect()
}

pub fn split_dataset(dataset: Vec<Point>, ratio: f32) -> (Vec<Point>, Vec<Point>) {
    let split_at = (dataset.len() as f32 * ratio).round() as usize;
    let (train, test) = dataset.split_at(split_at);
    (train.to_vec(), test.to_vec())
}
