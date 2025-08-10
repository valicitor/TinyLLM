use super::super::error::ModelError;
use ndarray::{s, Array2, Array3, ArrayView3};
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};

/// Token embedding layer
#[derive(Debug, Serialize, Deserialize)]
pub struct Embedding {
    pub weight: Array2<f32>, // Shape: [vocab_size, embed_dim]
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Result<Self, ModelError> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).map_err(|e| {
            ModelError::InitializationError(e.to_string())
        })?;

        Ok(Self {
            weight: Array2::from_shape_fn((vocab_size, embed_dim), |_| {
                normal.sample(&mut rng)
            }),
        })
    }

    /// Forward pass: input shape = [batch_size, seq_len], output = [batch_size, seq_len, embed_dim]
    pub fn forward(&self, indices: &[Vec<usize>]) -> Array3<f32> {
        let batch_size = indices.len();
        let seq_len = indices[0].len(); // assumes all sequences are same length
        let embed_dim = self.weight.shape()[1];

        let mut output = Array3::zeros((batch_size, seq_len, embed_dim));

        for (b, sequence) in indices.iter().enumerate() {
            for (t, &token_id) in sequence.iter().enumerate() {
                let embedding = self.weight.row(token_id);
                output.slice_mut(s![b, t, ..]).assign(&embedding);
            }
        }

        output
    }

    /// Backward pass: Accumulate gradients into embedding weights
    /// `indices` is [batch_size][seq_len], `grad_output` is [batch_size, seq_len, embed_dim]
    pub fn backward(&mut self, indices: &[Vec<usize>], grad_output: ArrayView3<f32>) {
        for (b, sequence) in indices.iter().enumerate() {
            for (t, &token_id) in sequence.iter().enumerate() {
                let grad = grad_output.slice(s![b, t, ..]);
                let mut row = self.weight.row_mut(token_id);
                row += &grad;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq; // handy for float comparisons

    #[test]
    fn test_embedding_new() {
        let vocab_size = 100;
        let embed_dim = 16;
        let embedding = Embedding::new(vocab_size, embed_dim).unwrap();

        // Check shape is correct
        assert_eq!(embedding.weight.shape(), &[vocab_size, embed_dim]);

        // Check weights are roughly zero-centered (mean close to 0)
        let mean: f32 = embedding.weight.mean().unwrap();
        assert!(mean.abs() < 0.01, "Mean not close to zero: {}", mean);
    }

    #[test]
    fn test_forward_output_shape() {
        let vocab_size = 10;
        let embed_dim = 8;
        let embedding = Embedding::new(vocab_size, embed_dim).unwrap();

        let indices = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
        ]; // batch_size=2, seq_len=3

        let output = embedding.forward(&indices);

        // Output shape should be (2, 3, 8)
        assert_eq!(output.shape(), &[2, 3, embed_dim]);
    }

    #[test]
    fn test_forward_output_values() {
        let vocab_size = 5;
        let embed_dim = 4;
        let mut embedding = Embedding::new(vocab_size, embed_dim).unwrap();

        // Overwrite weights for predictable test
        for i in 0..vocab_size {
            for j in 0..embed_dim {
                embedding.weight[[i, j]] = (i * 10 + j) as f32;
            }
        }

        let indices = vec![vec![0, 1, 2]];
        let output = embedding.forward(&indices);

        // output[0,0,:] should equal embedding.weight[0,:]
        for j in 0..embed_dim {
            assert_abs_diff_eq!(output[[0, 0, j]], embedding.weight[[0, j]], epsilon = 1e-6);
            assert_abs_diff_eq!(output[[0, 1, j]], embedding.weight[[1, j]], epsilon = 1e-6);
            assert_abs_diff_eq!(output[[0, 2, j]], embedding.weight[[2, j]], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_backward_accumulates_gradients() {
        let vocab_size = 3;
        let embed_dim = 2;
        let mut embedding = Embedding::new(vocab_size, embed_dim).unwrap();

        // Initialize weights to zero for clarity
        embedding.weight.fill(0.0);

        // Input indices: batch=1, seq_len=2
        let indices = vec![vec![0, 1]];

        // Gradients: shape = (1, 2, 2)
        let grad_output = ndarray::arr3(&[
            [[1.0, 2.0], [3.0, 4.0]],
        ]);

        embedding.backward(&indices, grad_output.view());

        // Weight row 0 should be incremented by [1.0, 2.0]
        assert_abs_diff_eq!(embedding.weight.row(0)[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(embedding.weight.row(0)[1], 2.0, epsilon = 1e-6);

        // Weight row 1 should be incremented by [3.0, 4.0]
        assert_abs_diff_eq!(embedding.weight.row(1)[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(embedding.weight.row(1)[1], 4.0, epsilon = 1e-6);

        // Weight row 2 should remain zero
        assert_abs_diff_eq!(embedding.weight.row(2)[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(embedding.weight.row(2)[1], 0.0, epsilon = 1e-6);
    }
}