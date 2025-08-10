use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, s};
use serde::{Deserialize, Serialize};

/// Layer Normalization with learnable parameters
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    /// Creates a new LayerNorm instance
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
        }
    }
    /// Forward pass: input shape [batch_size, seq_len, embed_dim]
    pub fn forward(&self, x: ArrayView3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, embed_dim) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, embed_dim));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let x_row = x.slice(s![b, t, ..]);
                let mut y_row = output.slice_mut(s![b, t, ..]);

                let mean = x_row.mean().unwrap();
                // Calculate variance using Bessel's correction (N-1)
                let var = x_row.var(0.0); // Changed from 1.0 to 0.0
                let std = (var + self.eps).sqrt();

                for i in 0..embed_dim {
                    let normalized = (x_row[i] - mean) / std;
                    y_row[i] = normalized * self.gamma[i] + self.beta[i];
                }
            }
        }

        output
    }

    /// Backward pass for layer normalization
    pub fn backward(&mut self, x: ArrayView2<f32>, grad_output: ArrayView2<f32>) -> Array2<f32> {
        let mut grad_input = Array2::zeros(x.dim());
        let mut grad_gamma = Array1::zeros(self.gamma.dim());
        let mut grad_beta = Array1::zeros(self.beta.dim());

        for i in 0..x.nrows() {
            let x_row = x.row(i);
            let grad_row = grad_output.row(i);

            let mean = x_row.mean().unwrap();
            let var = x_row.var(1.0);
            let std = (var + self.eps).sqrt();
            let n = x_row.len() as f32;

            // Compute gradients for gamma and beta
            for j in 0..x_row.len() {
                let x_centered = x_row[j] - mean;
                let normalized = x_centered / std;
                grad_gamma[j] += grad_row[j] * normalized;
                grad_beta[j] += grad_row[j];
            }

            // Compute gradient for input
            for j in 0..x_row.len() {
                let x_centered = x_row[j] - mean;
                let mut grad = grad_row[j] * self.gamma[j] / std;
                grad -= grad_row
                    .iter()
                    .zip(&self.gamma)
                    .map(|(g, gamma)| g * gamma)
                    .sum::<f32>()
                    / (n * std);
                grad -= x_centered
                    * grad_row
                        .iter()
                        .zip(&self.gamma)
                        .map(|(g, gamma)| g * gamma * x_centered)
                        .sum::<f32>()
                    / (n * var * std);
                grad_input[[i, j]] = grad;
            }
        }

        // Update parameters (in a real scenario, this would be part of an optimizer step)
        self.gamma -= &grad_gamma;
        self.beta -= &grad_beta;

        grad_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, array};

    #[test]
    fn test_layernorm_forward_identity() {
        let layer = LayerNorm::new(3);
        let input = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer.forward(input.view());

        // Updated expected calculation
        let mean = 2.0;
        let var = (1.0f32.powi(2) + 0.0 + 1.0f32.powi(2)) / 3.0; // Population variance
        let std = (var + layer.eps).sqrt();
        let expected = array![[[
            (1.0 - mean) / std,
            (2.0 - mean) / std,
            (3.0 - mean) / std
        ]]];

        assert_eq!(output.shape(), expected.shape());
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() <= 1e-4, "output: {}, expected: {}", o, e);
        }
    }

    #[test]
    fn test_layernorm_forward_gamma_beta() {
        let mut layer = LayerNorm::new(2);
        layer.gamma = array![2.0, 0.5];
        layer.beta = array![1.0, -1.0];

        let input = Array3::from_shape_vec((1, 1, 2), vec![4.0, 6.0]).unwrap();
        let output = layer.forward(input.view());

        // Updated expected calculation
        let mean = 5.0;
        let var = (1.0f32.powi(2) + 1.0f32.powi(2)) / 2.0; // Population variance
        let std = (var + layer.eps).sqrt();
        let norm_0 = (4.0 - mean) / std;
        let norm_1 = (6.0 - mean) / std;

        let expected = array![[[
            norm_0 * 2.0 + 1.0,
            norm_1 * 0.5 - 1.0
        ]]];

        assert_eq!(output.shape(), expected.shape());
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() <= 1e-4, "output: {}, expected: {}", o, e);
        }
    }

    #[test]
    fn test_layernorm_backward_shapes_match() {
        let mut layer = LayerNorm::new(3);

        // Input: batch=2, dim=3
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Simulated gradient from next layer (same shape)
        let grad_output = array![[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]];

        let grad_input = layer.backward(input.view(), grad_output.view());

        assert_eq!(grad_input.shape(), &[2, 3]);
    }

    #[test]
    fn test_layernorm_backward_effect() {
        let mut layer = LayerNorm::new(2);

        // Save original gamma and beta
        let original_gamma = layer.gamma.clone();
        let original_beta = layer.beta.clone();

        let input = array![[1.0, 2.0]];
        let grad_output = array![[0.5, -0.5]];

        let _ = layer.backward(input.view(), grad_output.view());

        // Check that gamma and beta were updated
        assert!(
            layer.gamma != original_gamma,
            "Gamma should change after backward"
        );
        assert!(
            layer.beta != original_beta,
            "Beta should change after backward"
        );
    }
}
