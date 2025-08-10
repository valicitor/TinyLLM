use ndarray::Array3;
use serde::{Serialize, Deserialize};

use crate::{model::ModelError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub dropout_rate: f32,
    // You can extend this later with attention type (e.g., causal vs full), initializer, etc.
}

#[typetag::serde]
pub trait Attention {
    fn forward(&self, x: &Array3<f32>, training: bool) -> Result<Array3<f32>, ModelError>;
}
