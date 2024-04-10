pub type Vector = Vec<f32>;

/// Calculate the magnitude of a Vector.
pub fn magnitude(v: &Vector) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Calculate the dot product of two Vectors.
pub fn dot(a: &Vector, b: &Vector) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Calculate the cosine similarity of two Vectors.
pub fn cosine_similarity(a: &Vector, b: &Vector) -> f32 {
    (dot(a, b) / (magnitude(a) * magnitude(b))).max(0.0)
}

/// Calculate the cosine distance of two Vectors.
pub fn cosine_distance(a: &Vector, b: &Vector) -> f32 {
    1.0 - cosine_similarity(a, b)
}

pub fn euclidean_distance(a: &Vector, b: &Vector) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powf(2.0))
        .sum::<f32>()
        .sqrt()
}

pub fn distance(a: &Vector, b: &Vector) -> f32 {
    cosine_distance(a, b)
}

// We use cosine distance because it's important to match the distance function
// between the embedding model and the search engine. (The OpenAI model was
// trained to minimize the cosine distance between similar documents.)
// https://help.openai.com/en/articles/6824809-embeddings-frequently-asked-questions
