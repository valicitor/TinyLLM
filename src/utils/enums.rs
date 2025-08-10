use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionalEncodingType {
    /// Fixed sinusoidal patterns (non-learnable)
    Sinusoidal,
    /// Learned positional embeddings
    Learned,
    /// Rotary position embeddings (RoPE)
    Rotary,
}

impl Default for PositionalEncodingType {
    fn default() -> Self {
        Self::Sinusoidal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    MultiHead,
    // TODO: Implement other attention types
    Flash,
    Linformer,
    Performer,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::MultiHead
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    // TODO: Implement other optimizers
    SGD {
        learning_rate: f32,
        momentum: Option<f32>,
    },
    RMSProp {
        learning_rate: f32,
        decay: f32,
        epsilon: f32,
    },
}