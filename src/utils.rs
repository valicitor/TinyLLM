use ndarray::{Array2};
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub fn cross_entropy(logits: &Array2<f32>, targets: &[usize]) -> f32 {
    let mut loss = 0.0;
    for (i, &target) in targets.iter().enumerate() {
        let row = logits.row(i);
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute sum of exp shifted by max
        let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        
        // Log softmax of target index
        let log_prob = (row[target] - max) - sum_exp.ln();
        
        loss -= log_prob;
    }
    loss / targets.len() as f32
}

pub fn sample_from_logits(logits: &Array2<f32>,temperature: f32,top_k: Option<usize>) -> usize {
    let last_row = logits.row(logits.nrows() - 1);

    // Apply temperature scaling
    let scaled_logits: Vec<(usize, f32)> = last_row
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / temperature))
        .collect();

    // Filter top_k if provided
    let filtered: Vec<(usize, f32)> = if let Some(k) = top_k {
        let mut sorted = scaled_logits.clone();
        // Sort descending by logit value
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(k);
        sorted
    } else {
        scaled_logits
    };

    // Extract filtered indices and logits separately
    let indices: Vec<usize> = filtered.iter().map(|&(i, _)| i).collect();
    let logits_filtered: Vec<f32> = filtered.iter().map(|&(_, logit)| logit).collect();

    // Compute softmax probabilities
    let max_logit = logits_filtered
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits_filtered
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    // Sample from weighted distribution
    let dist = WeightedIndex::new(&probs).unwrap();
    let mut rng = thread_rng();
    let sampled_idx = dist.sample(&mut rng);

    // Map back to original vocab index
    indices[sampled_idx]
}