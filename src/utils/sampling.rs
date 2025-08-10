use ndarray::Array2;
use rand::distributions::{WeightedIndex};
use rand::{thread_rng, Rng};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SamplingError {
    #[error("Invalid temperature: {0}")]
    InvalidTemperature(f32),
    #[error("Empty distribution")]
    EmptyDistribution,
}

/// Samples from logits using temperature scaling and optional top-k/top-p filtering
pub fn sample_from_logits(
    logits: &Array2<f32>,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Result<usize, SamplingError> {
    if temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature(temperature));
    }

    let last_row = logits.row(logits.nrows() - 1);
    let mut scaled_logits: Vec<(usize, f32)> = last_row
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / temperature))
        .collect();

    // Apply top-k filtering
    if let Some(k) = top_k {
        scaled_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scaled_logits.truncate(k);
    }

    // Apply top-p (nucleus) sampling
    if let Some(p) = top_p {
        scaled_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut cumulative_prob = 0.0;
        let mut cutoff = scaled_logits.len();
        for (i, &(_, logit)) in scaled_logits.iter().enumerate() {
            let prob = logit.exp();
            cumulative_prob += prob;
            if cumulative_prob > p {
                cutoff = i + 1;
                break;
            }
        }
        scaled_logits.truncate(cutoff);
    }

    if scaled_logits.is_empty() {
        return Err(SamplingError::EmptyDistribution);
    }

    // Softmax and sample
    let max_logit = scaled_logits.iter().map(|&(_, logit)| logit).fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = scaled_logits.iter().map(|&(_, logit)| (logit - max_logit).exp()).sum();
    
    let dist = WeightedIndex::new(
        scaled_logits.iter().map(|&(_, logit)| (logit - max_logit).exp() / sum_exp)
    ).map_err(|_| SamplingError::EmptyDistribution)?;
    
    Ok(scaled_logits[thread_rng().sample(dist)].0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sample_from_logits() {
        let logits = array![[0.1, 0.2, 0.7]];
        let result = sample_from_logits(&logits, 1.0, None, None).unwrap();
        assert!(result < 3);
    }
}