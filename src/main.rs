mod dataset;
mod evaluation;
mod exact_knn;
mod hnsw;
mod index;
mod point;
mod vector;

use dataset::{read_ndjson, split_dataset};
use evaluation::{evaluate_precision, evaluate_recall};
use exact_knn::ExactKNNIndex;
use hnsw::HNSWIndex;
use index::Index;
use point::Point;
use std::time::Instant;

// Function to create an index
fn create_index<T: Index>(index: &mut T, data: Vec<Point>) {
    print!("Creating {} index... ", std::any::type_name::<T>());
    let start_index = Instant::now();
    index.insert_many(data);
    println!("Done in {:.2?}ms", start_index.elapsed().as_millis());
}

// Function to search an index
fn search_index<'a, T: Index>(
    index: &'a T,
    query_point: &'a Point,
    k: usize,
) -> Vec<(f32, &'a Point)> {
    print!(
        "Searching {} for {:?}... ",
        std::any::type_name::<T>(),
        query_point
            .body
            .split('.')
            .next()
            .unwrap_or(&query_point.body)
            .to_string()
    );
    let start_search = Instant::now();
    let results = index.search(&query_point.vector, k);
    println!("Done in {:.2?}ms", start_search.elapsed().as_millis());
    println!("Top {k} ({}):", std::any::type_name::<T>());
    println!(
        "{}",
        results
            .iter()
            .take(k)
            .map(|r| format!(
                "- {}",
                r.1.body.split('.').next().unwrap_or(&r.1.body).to_string()
            ))
            .collect::<Vec<String>>()
            .join("\n")
            .to_string()
    );
    results
}

fn main() {
    // Read the dataset
    print!("Reading dataset... ");
    let start_read = Instant::now();
    let data = read_ndjson::<Point>("wikipedia_embeddings.ndjson", 1000);
    let (base, query) = split_dataset(data, 0.95);
    println!("Done in {:.2?}ms", start_read.elapsed().as_millis());

    // Create an ExactKNN index of the dataset
    let mut exact_knn_index = ExactKNNIndex::new();
    create_index(&mut exact_knn_index, base.clone());

    // Create an HNSW index of the dataset
    let mut hnsw_index = HNSWIndex::new();
    create_index(&mut hnsw_index, base.clone());

    // Define a k value for our search
    let k = 10;

    // Search for a random query point from the query dataset.
    let query_point = &query[29];

    let hnsw_results = search_index(&hnsw_index, query_point, k);
    let hnsw_points = hnsw_results.iter().map(|(_, point)| *point).collect();

    let exact_results = search_index(&exact_knn_index, query_point, k);
    let exact_points = exact_results.iter().map(|(_, point)| *point).collect();

    let recall = evaluate_recall(&hnsw_points, &exact_points);
    println!("Recall (@{k}): {:.2?}", recall);

    let precision = evaluate_precision(&hnsw_points, &exact_points);
    println!("Precision (@{k}): {:.2?}", precision);
}
