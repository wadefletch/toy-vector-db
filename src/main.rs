mod dataset;
mod exact_knn;
mod hnsw;
mod index;
mod point;
mod vector;

use dataset::{read_dataset, split_dataset};
use exact_knn::ExactKNNIndex;
use index::Index;
use std::time::Instant;

fn main() {
    // Read the dataset
    print!("Reading dataset... ");
    let start_read = Instant::now();
    let data = read_dataset("airbnb_embeddings.json");
    let (base, query) = split_dataset(data, 0.8);
    println!("Done in {:.2?}ms", start_read.elapsed().as_millis());

    // Create an index of the dataset
    print!("Creating index... ");
    let start_index = Instant::now();
    let mut index = ExactKNNIndex::new();
    index.insert_many(base.clone());
    println!("Done in {:.2?}ms", start_index.elapsed().as_millis());

    // Perform a full cosine search for a random query point.
    let query_point = &query[7];
    print!("Searching for {:?}... ", query_point.name);
    let start_search = Instant::now();
    let results = index.search(&query_point.vector, 10);
    println!("Done in {:.2?}ms", start_search.elapsed().as_millis());
    println!("Top 3:");
    println!(
        "{}",
        results
            .iter()
            .take(3)
            .map(|r| format!("- {}", r.1.name.clone()))
            .collect::<Vec<String>>()
            .join("\n")
    );
}
