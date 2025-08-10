use super::super::error::ModelError;
use ndarray::{s, Array2, Array3, ArrayView3};
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};

/// A linear (fully-connected) layer
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Array2<f32>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self, ModelError> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).map_err(|e| {
            ModelError::InitializationError(e.to_string())
        })?;

        Ok(Self {
            weight: Array2::from_shape_fn((input_dim, output_dim), |_| {
                normal.sample(&mut rng)
            }),
            bias: Array2::zeros((1, output_dim)),
        })
    }

    /// Forward pass for 3D input: [batch_size, seq_len, input_dim]
    pub fn forward(&self, x: ArrayView3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = x.dim();
        let output_dim = self.bias.shape()[1];

        let mut output = Array3::zeros((batch_size, seq_len, output_dim));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let x_row = x.slice(s![b, t, ..]);
                let mut y_row = output.slice_mut(s![b, t, ..]);
                y_row.assign(&(&x_row.dot(&self.weight) + &self.bias.row(0)));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array, Array3};

    #[test]
    fn test_linear_initialization() {
        let input_dim = 4;
        let output_dim = 3;
        let linear = Linear::new(input_dim, output_dim).unwrap();

        assert_eq!(linear.weight.shape(), &[input_dim, output_dim]);
        assert_eq!(linear.bias.shape(), &[1, output_dim]);
    }

    #[test]
    fn test_forward_output_shape() {
        let input_dim = 5;
        let output_dim = 2;
        let linear = Linear::new(input_dim, output_dim).unwrap();

        let batch_size = 2;
        let seq_len = 3;
        let input = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
        let output = linear.forward(input.view());

        assert_eq!(output.dim(), (batch_size, seq_len, output_dim));
    }

    #[test]
    fn test_forward_computation_known_weights() {
        // Define a Linear layer manually for deterministic testing
        let weight = array![[1.0, 2.0], [0.0, 1.0], [-1.0, 0.0]];
        let bias = array![[0.5, -0.5]];
        let linear = Linear { weight, bias };

        // Input: batch size 1, seq len 1, input dim 3
        let input = Array::from_shape_vec((1, 1, 3), vec![2.0, 3.0, 4.0]).unwrap();
        let output = linear.forward(input.view());

        // Manually compute: y = x · W + b = [2, 3, 4] · [[1,2],[0,1],[-1,0]] + [0.5,-0.5]
        // => y = [2*1 + 3*0 + 4*(-1) + 0.5, 2*2 + 3*1 + 4*0 - 0.5] = [-1.5, 6.5]
        let expected = Array::from_shape_vec((1, 1, 2), vec![-1.5, 6.5]).unwrap();

        for ((o, e), idx) in output.iter().zip(expected.iter()).zip(0..) {
            assert!((o - e).abs() < 1e-5, "Mismatch at index {}: got {}, expected {}", idx, o, e);
        }
    }
}