use serde::{Serialize, Deserialize};
use crate::{model::attention::AttentionConfig, utils::{AttentionType, OptimizerType, PositionalEncodingType}};

use super::error::ModelError;

/// Configuration for the language model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // --- Architecture ---
    pub embed_dim: usize,                // d_model     Dimension of token embeddings (and all hidden states).	    512–8192 (e.g., 768)
    pub vocab_size: usize,              //              Number of unique tokens in the vocabulary.	                50k–200k (e.g., 50_257)
    pub block_size: usize,              // max_seq_len  Maximum input sequence length (for positional embeddings).  512–4096 (e.g., 1024)
    pub num_layers: usize,              // = n_layers   Number of stacked Transformer blocks.                       6–100+ (e.g., GPT-3: 96)
    pub n_heads: usize,                 //              Number of parallel attention heads in MultiHeadAttention.   8–16 (must divide d_model)
    pub d_ff: usize,                    //              Hidden dimension in the FeedForward layer.                  4 * d_model (e.g., 3072)
    pub positional_encoding: PositionalEncodingType,  // = positional_embedding

    // --- Attention ---
    pub attention_type: AttentionType,   // Your custom field
    pub attn_config: AttentionConfig, // Configuration for attention mechanism

    // --- Regularization ---
    #[serde(default = "default_dropout")]
    pub dropout: f32,                    //             Dropout rate for attention/FFN layers (prevents overfitting).	0.0–0.2
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,             //             Epsilon for numerical stability in LayerNorm.	                1e-5–1e-12

    // --- Activation ---
    //#[serde(default = "default_activation")]
    //pub activation: Activation,          // Added from TransformerConfig (e.g., GELU)

    // --- Optimizer (Training) ---
    //pub optimizer_type: OptimizerType,   // Your custom field
}

// Default values for optional fields
fn default_dropout() -> f32 { 0.1 }
fn default_layer_norm_eps() -> f32 { 1e-5 }
//fn default_activation() -> Activation { Activation::GELU }

impl ModelConfig {

}