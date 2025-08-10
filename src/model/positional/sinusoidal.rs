use ndarray::{Array2, Array3};
use serde::{Serialize, Deserialize};

use crate::model::{positional::PositionalEncoding};

/// Sinusoidal positional encoding
#[derive(Debug, Serialize, Deserialize)]
pub struct SinusoidalPositionalEncoding {
    pub encoding: Array2<f32>,
}

impl SinusoidalPositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut encoding = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / (10000.0f32.powf((2 * i) as f32 / d_model as f32));
                if i % 2 == 0 {
                    encoding[[pos, i]] = angle.sin();
                } else {
                    encoding[[pos, i]] = angle.cos();
                }
            }
        }

        Self { encoding }
    }
}

#[typetag::serde]
impl PositionalEncoding for SinusoidalPositionalEncoding {
    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = input.dim();
        let encoding = self.encoding.slice(ndarray::s![..seq_len, ..]);
        let mut output = input.to_owned();
        
        for mut batch in output.outer_iter_mut() {
            batch += &encoding;
        }
        
        output
    }

    fn backward(&mut self, _positions: &Array2<usize>, _grad_output: &Array3<f32>) {
        // No-op for sinusoidal encoding
    }
}