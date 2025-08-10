use ndarray::{Array2, Array4, ArrayView2, ArrayViewMut2, ArrayViewMut4};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MathError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Applies causal mask to attention scores (upper triangular set to -inf)
/// # Arguments
/// * `scores` - Mutable reference to attention scores array [batch_size, num_heads, seq_len, seq_len]
pub fn apply_causal_mask(scores: &mut Array4<f32>) {
    let seq_len = scores.dim().2;
    let mut mask = Array2::<f32>::ones((seq_len, seq_len));
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[i, j]] = f32::NEG_INFINITY;
        }
    }

    for mut batch in scores.outer_iter_mut() {
        for mut head in batch.outer_iter_mut() {
            head *= &mask;
        }
    }
}

/// Computes softmax along the last dimension of a 2D array
pub fn softmax_2d(matrix: &mut ArrayViewMut2<f32>) {
    for mut row in matrix.rows_mut() {
        // Find max value, ignoring NaNs and infinities
        let max = row.fold(f32::NEG_INFINITY, |a, &b| {
            if b.is_nan() || b.is_infinite() { a } else { a.max(b) }
        });
        
        // If we got -inf (all values were non-finite), set to uniform distribution
        if max == f32::NEG_INFINITY {
            row.fill(1.0 / row.len() as f32);
            continue;
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        
        // Normalize with protection against division by zero
        sum = sum.max(1e-20); // Very small epsilon
        for val in row.iter_mut() {
            *val /= sum;
            // Final validation
            if !val.is_finite() {
                *val = 0.0; // Fallback to zero if still invalid
            }
        }
    }
}

/// Computes softmax along the last dimension of a 4D array
/// # Arguments
/// * `matrix` - Mutable reference to 4D array [batch_size, num_heads, seq_len, seq_len]
pub fn softmax_4d(matrix: &mut Array4<f32>) {
    for mut batch in matrix.outer_iter_mut() {
        for mut head in batch.outer_iter_mut() {
            softmax_2d(&mut head);
        }
    }
}

/// Applies dropout to a 4D array with optional RNG seed for reproducibility
/// # Arguments
/// * `matrix` - Mutable reference to 4D array
/// * `dropout_rate` - Probability of dropping an element (0.0 to 1.0)
/// * `training` - Whether to apply dropout (false during inference)
/// * `rng` - Optional seeded RNG for reproducibility
pub fn apply_dropout_4d(
    matrix: &mut Array4<f32>,
    dropout_rate: f32,
    training: bool,
    rng: &mut Option<SmallRng>,
) {
    if !training || dropout_rate <= 0.0 {
        return;
    }

    // Initialize RNG if not provided
    if rng.is_none() {
        *rng = Some(SmallRng::seed_from_u64(42));
    }

    let keep_prob = 1.0 - dropout_rate;
    let rng = rng.as_mut().unwrap();

    matrix.mapv_inplace(|x| {
        if rng.r#gen::<f32>() < keep_prob {
            x / keep_prob
        } else {
            0.0
        }
    });
}

/// Builder for creating a configured dropout RNG
pub fn build_dropout_rng(dropout_rate: f32, seed: Option<u64>) -> Option<SmallRng> {
    if dropout_rate > 0.0 {
        Some(SmallRng::seed_from_u64(seed.unwrap_or(42)))
    } else {
        None
    }
}

/// Computes cross-entropy loss
pub fn cross_entropy(logits: ArrayView2<f32>, targets: &[usize]) -> Result<f32, MathError> {
    if logits.nrows() != targets.len() {
        return Err(MathError::DimensionMismatch(format!(
            "Expected {} rows, got {}",
            targets.len(),
            logits.nrows()
        )));
    }

    let mut loss = 0.0;
    for (i, &target) in targets.iter().enumerate() {
        let row = logits.row(i);
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        let log_prob = (row[target] - max) - sum_exp.ln();
        loss -= log_prob;
    }

    Ok(loss / targets.len() as f32)
}

/// Computes gradient of cross-entropy loss w.r.t. logits
pub fn d_cross_entropy(
    logits: ArrayView2<f32>,
    targets: &[usize],
) -> Result<Array2<f32>, MathError> {
    let batch_size = logits.nrows();
    let vocab_size = logits.ncols();

    // Create an owned array from the view
    let mut dlogits = logits.to_owned();

    // Validate targets length matches batch size
    if targets.len() != batch_size {
        return Err(MathError::DimensionMismatch(format!(
            "Targets length {} doesn't match batch size {}",
            targets.len(),
            batch_size
        )));
    }

    for i in 0..batch_size {
        let max = dlogits.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_sum = 0.0;

        // Softmax computation
        for j in 0..vocab_size {
            dlogits[[i, j]] = (dlogits[[i, j]] - max).exp();
            exp_sum += dlogits[[i, j]];
        }

        // Normalize and compute gradient
        for j in 0..vocab_size {
            dlogits[[i, j]] /= exp_sum;
        }

        // Subtract 1 for the target class
        dlogits[[i, targets[i]]] -= 1.0;
    }

    Ok(dlogits)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2, Array4};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::utils::{apply_causal_mask, apply_dropout_4d, build_dropout_rng, cross_entropy, d_cross_entropy, softmax_2d, softmax_4d};

    #[test]
    fn test_apply_causal_mask() {
        let mut scores = Array4::<f32>::ones((1, 1, 3, 3));
        apply_causal_mask(&mut scores);

        // Break out the nested view indexing into separate bindings
        let first_batch = scores.index_axis(ndarray::Axis(0), 0);
        let first_head = first_batch.index_axis(ndarray::Axis(0), 0);

        // Now we can safely use the final view
        assert_eq!(first_head[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(first_head[[0, 2]], f32::NEG_INFINITY);
        assert_eq!(first_head[[1, 2]], f32::NEG_INFINITY);
        assert!(first_head[[1, 0]].is_finite());
        assert!(first_head[[2, 0]].is_finite());
    }

    #[test]
    fn test_softmax_2d() {
        let mut matrix = array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
        softmax_2d(&mut matrix.view_mut());
        for row in matrix.rows() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_4d() {
        let mut matrix = Array4::<f32>::ones((1, 1, 2, 3));
        softmax_4d(&mut matrix);
        for row in matrix.outer_iter() {
            for head in row.outer_iter() {
                for line in head.rows() {
                    let sum: f32 = line.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_apply_dropout_4d() {
        let mut matrix = Array4::<f32>::ones((1, 1, 2, 2));
        let mut rng = Some(SmallRng::seed_from_u64(1234));
        apply_dropout_4d(&mut matrix, 0.5, true, &mut rng);

        let count_kept = matrix.iter().filter(|&&x| x != 0.0).count();
        assert!(count_kept <= 4 && count_kept >= 1); // Expect some dropped elements
    }

    #[test]
    fn test_cross_entropy_and_gradient() {
        let logits = array![[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]];
        let targets = vec![0, 1];

        let loss = cross_entropy(logits.view(), &targets).unwrap();
        assert!(loss > 0.0);

        let grad = d_cross_entropy(logits.view(), &targets).unwrap();
        assert_eq!(grad.dim(), (2, 3));
        for row in grad.rows() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 0.0).abs() < 1e-5); // Gradient sums to 0 per row
        }
    }

    #[test]
    fn test_build_dropout_rng_none() {
        let rng = build_dropout_rng(0.0, None);
        assert!(rng.is_none());
    }

    #[test]
    fn test_build_dropout_rng_some() {
        let rng = build_dropout_rng(0.5, Some(1234));
        assert!(rng.is_some());
    }
}
