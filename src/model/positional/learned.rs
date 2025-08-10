use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};

use crate::model::{positional::PositionalEncoding, ModelError};

/// Learned positional embeddings
#[derive(Debug, Serialize, Deserialize)]
pub struct LearnedPositionalEncoding {
    embeddings: Array2<f32>,
}

impl LearnedPositionalEncoding {
    /// Creates new learned positional embeddings
    pub fn new(max_len: usize, dim: usize) -> Result<Self, ModelError> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).map_err(|e| {
            ModelError::InitializationError(e.to_string())
        })?;

        Ok(Self {
            embeddings: Array2::from_shape_fn((max_len, dim), |_| {
                normal.sample(&mut rng)
            }),
        })
    }

    /// Gets embeddings for a sequence of positions
pub fn forward_positions(&self, positions: &Array2<usize>) -> Result<Array3<f32>, ModelError> {
        let (batch_size, seq_len) = positions.dim();
        let dim = self.embeddings.shape()[1];
        let mut output = Array3::zeros((batch_size, seq_len, dim));

        for (batch_idx, positions_row) in positions.rows().into_iter().enumerate() {
            for (seq_idx, &pos) in positions_row.iter().enumerate() {
                if pos >= self.embeddings.shape()[0] {
                    return Err(ModelError::ConfigError(format!(
                        "Position {} exceeds maximum length {}",
                        pos,
                        self.embeddings.shape()[0]
                    )));
                }
                output.slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&self.embeddings.row(pos));
            }
        }

        Ok(output)
    }

    pub fn backward_positions(&mut self, positions: &Array2<usize>, grad_output: ArrayView3<f32>) {
        for (batch_idx, positions_row) in positions.rows().into_iter().enumerate() {
            for (seq_idx, &pos) in positions_row.iter().enumerate() {
                if pos < self.embeddings.shape()[0] {
                    let mut emb = self.embeddings.row_mut(pos);
                    emb += &grad_output.slice(s![batch_idx, seq_idx, ..]);
                }
            }
        }
    }
}

#[typetag::serde]
impl PositionalEncoding for LearnedPositionalEncoding {
    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let positions = Array2::from_shape_fn((batch_size, seq_len), |(_, seq_idx)| seq_idx);
        self.forward_positions(&positions).expect("Valid positions")
    }

    fn backward(&mut self, positions: &Array2<usize>, grad_output: &Array3<f32>) {
        self.backward_positions(positions, grad_output.view());
    }
}