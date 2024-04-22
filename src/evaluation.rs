use crate::point::Point;

pub fn intersection_count(a: &Vec<&Point>, b: &Vec<&Point>) -> usize {
    a.iter().filter(|point| b.contains(point)).count()
}

pub fn evaluate_recall(retrieved: &Vec<&Point>, relevant: &Vec<&Point>) -> f32 {
    // recall https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Recall
    intersection_count(&retrieved, &relevant) as f32 / relevant.len() as f32
}

pub fn evaluate_precision(retrieved: &Vec<&Point>, relevant: &Vec<&Point>) -> f32 {
    // precision https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision
    intersection_count(&retrieved, &relevant) as f32 / retrieved.len() as f32
}
