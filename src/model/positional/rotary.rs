use ndarray::{s, Array2, Array3, ArrayViewMut2, ArrayViewMut3};
use serde::{Serialize, Deserialize};

use crate::model::{positional::PositionalEncoding, ModelError};

/// Rotary Position Embedding (RoPE) implementation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RotaryPositionEmbedding {
    dim: usize,
    max_seq_len: usize,
    freq: Array2<f32>,
}

impl RotaryPositionEmbedding {
    /// Creates new rotary position embeddings
    pub fn new(dim: usize, max_seq_len: usize) -> Result<Self, ModelError> {
        if dim % 2 != 0 {
            return Err(ModelError::ConfigError(
                "Rotary embedding dimension must be even".to_string()
            ));
        }

        let mut freq = Array2::zeros((max_seq_len, dim / 2));
        for pos in 0..max_seq_len {
            for i in 0..dim/2 {
                let theta = 1.0 / (10000.0f32.powf(2.0 * i as f32 / dim as f32));
                freq[[pos, i]] = (pos as f32 * theta).sin();
            }
        }

        Ok(Self {
            dim,
            max_seq_len,
            freq,
        })
    }

    /// Applies rotary embeddings to queries or keys in-place
    pub fn apply_rotary_embeddings(
        &self,
        mut x: ArrayViewMut3<f32>,
        positions: &Array2<usize>,
    ) -> Result<(), ModelError> {
        let (batch_size, seq_len, dim) = x.dim();
        
        if dim != self.dim {
            return Err(ModelError::DimensionMismatch(format!(
                "Input dimension {} doesn't match embedding dimension {}",
                dim, self.dim
            )));
        }

        for (batch_idx, positions_row) in positions.rows().into_iter().enumerate() {
            for (seq_idx, &pos) in positions_row.iter().enumerate() {
                if pos >= self.max_seq_len {
                    return Err(ModelError::ConfigError(format!(
                        "Position {} exceeds maximum sequence length {}",
                        pos, self.max_seq_len
                    )));
                }

                let mut row = x.slice_mut(s![batch_idx, seq_idx, ..]);
                for i in 0..self.dim/2 {
                    let x0 = row[i];
                    let x1 = row[i + self.dim/2];
                    let sin = self.freq[[pos, i]];
                    let cos = self.freq[[pos, i]].cos();

                    row[i] = x0 * cos - x1 * sin;
                    row[i + self.dim/2] = x0 * sin + x1 * cos;
                }
            }
        }

        Ok(())
    }
}

#[typetag::serde]
impl PositionalEncoding for RotaryPositionEmbedding {
    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let mut output = input.to_owned();
        let (batch_size, seq_len, _) = input.dim();
        let positions = Array2::from_shape_fn((batch_size, seq_len), |(_, seq_idx)| seq_idx);
        self.apply_rotary_embeddings(output.view_mut(), &positions)
            .expect("Valid application of rotary embeddings");
        output
    }

    fn backward(&mut self, _positions: &Array2<usize>, _grad_output: &Array3<f32>) {
        // No-op for rotary embeddings
    }
}