use crate::point::Point;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Next, let's read the dataset from the file. As this is an NDJSON (Newline
// Delimited JSON) file, we can use the `lines` method to read each line of the
// file, then use serde_json to parse each line into a `Point` struct.
pub fn read_ndjson<T: serde::de::DeserializeOwned>(filename: &str, limit: usize) -> Vec<T> {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let lines = reader.lines();
    lines
        .take(limit)
        .filter_map(|line| serde_json::from_str::<T>(&line.ok()?).ok())
        .collect()
}

pub fn split_dataset(dataset: Vec<Point>, ratio: f32) -> (Vec<Point>, Vec<Point>) {
    let split_at = (dataset.len() as f32 * ratio).round() as usize;
    let (train, test) = dataset.split_at(split_at);
    (train.to_vec(), test.to_vec())
}
