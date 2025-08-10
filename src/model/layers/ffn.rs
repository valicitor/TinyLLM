use ndarray::{s, Array2, Array3, ArrayView3};
use serde::{Serialize, Deserialize};

use crate::model::{Linear, ModelError};

#[derive(Serialize, Deserialize, Clone)]
pub struct FeedForwardNetwork {
    pub linear1: Linear,  // d_model × d_ff
    pub linear2: Linear,  // d_ff × d_model
}

impl FeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize) -> Result<Self, ModelError> {
        Ok(Self { 
            linear1: Linear::new(d_model, d_ff)?, 
            linear2: Linear::new(d_ff, d_model)?
        })
    }

    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let hidden = self.linear1.forward(x.view()).mapv(Self::gelu);
        self.linear2.forward(hidden.view())
    }

    #[inline]
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (x / std::f32::consts::SQRT_2).tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array, Array3};

    #[test]
    fn test_ffn_output_shape() {
        let d_model = 4;
        let d_ff = 8;
        let ffn = FeedForwardNetwork::new(d_model, d_ff).unwrap();

        let input = Array3::<f32>::zeros((2, 3, d_model));
        let output = ffn.forward(&input);

        assert_eq!(output.dim(), (2, 3, d_model));
    }

    #[test]
    fn test_ffn_forward_known_weights() {
        // Create a fake FFN with known weights and biases
        let linear1 = Linear {
            weight: array![[1.0, 0.0], [0.0, 1.0]].into_dyn().into_dimensionality().unwrap(),
            bias: array![[0.0, 0.0]],
        };
        let linear2 = Linear {
            weight: array![[1.0, 1.0], [-1.0, 1.0]].into_dyn().into_dimensionality().unwrap(),
            bias: array![[0.0, 0.0]],
        };

        let ffn = FeedForwardNetwork { linear1, linear2 };

        // Input: batch size 1, seq len 1, d_model = 2
        let input = Array::from_shape_vec((1, 1, 2), vec![1.0, 2.0]).unwrap();
        let output = ffn.forward(&input);

        // First layer: identity, so hidden = input => GELU(input)
        let gelu = |x: f32| -> f32 { 0.5 * x * (1.0 + (x / std::f32::consts::SQRT_2).tanh()) };
        let h0 = gelu(1.0);
        let h1 = gelu(2.0);

        // Second layer: y = h0 * 1.0 + h1 * -1.0, h0 * 1.0 + h1 * 1.0
        let expected = array![[[h0 - h1, h0 + h1]]];

        for ((o, e), i) in output.iter().zip(expected.iter()).zip(0..) {
            assert!((o - e).abs() < 1e-5, "Mismatch at index {}: got {}, expected {}", i, o, e);
        }
    }
}