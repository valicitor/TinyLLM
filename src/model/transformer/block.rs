//! src/model/transformer.rs

use ndarray::Array3;
use serde::{Serialize, Deserialize};
use crate::model::layers::{LayerNorm, FeedForwardNetwork};
use crate::model::attention::{Attention, AttentionConfig};
use crate::model::{ModelError, MultiHeadAttention};
use crate::utils::AttentionType;

/// A single transformer block
#[derive(Serialize, Deserialize)]
pub struct TransformerBlock {
    attention: Box<dyn Attention>,
    attn_config: AttentionConfig,
    ffn: FeedForwardNetwork, 
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(
        attn_type: &AttentionType,
        attn_config: AttentionConfig,
        embed_dim: usize,
        vocab_size: usize,
    ) -> Result<Self, ModelError> {
        let config = attn_config.clone();
        let attention = Self::build_attention(attn_type, config);
        let ffn = FeedForwardNetwork::new(embed_dim, vocab_size)?;
        let ln1 = LayerNorm::new(embed_dim);
        let ln2 = LayerNorm::new(embed_dim);

        Ok(Self {
            attention,
            attn_config,
            ffn,
            ln1,
            ln2,
        })
    }

    fn build_attention(attn_type: &AttentionType, config: AttentionConfig) -> Box<dyn Attention> {
        match attn_type {
            AttentionType::MultiHead => {
                Box::new(MultiHeadAttention::new(config))
            }
            AttentionType::Flash => todo!(),
            AttentionType::Linformer => todo!(),
            AttentionType::Performer => todo!(),
        }
    }

    pub fn forward(&self, x: Array3<f32>, training: bool) -> Result<Array3<f32>, ModelError> {
        // Self-attention + residual
        let norm1 = self.ln1.forward(x.view());
        let attention_output = self.attention.forward(&norm1, training)?;
        let x = &x + &attention_output;

        // FFN + residual
        let norm2 = self.ln2.forward(x.view());
        let x = x + self.ffn.forward(&norm2);
        Ok(x)
    }
}
 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AttentionType};  // adjust path if needed
    use ndarray::{arr3};

    // Your build_attention function (or import it)    
    fn build_attention(attn_type: &AttentionType, config: AttentionConfig) -> Box<dyn Attention> {
        match attn_type {
            AttentionType::MultiHead => {
                Box::new(MultiHeadAttention::new(config))
            }
            _ => todo!(),
        }
    }

    #[test]
    fn test_transformer_block_new_success() {
        let attn_type = AttentionType::MultiHead;

        let embed_dim = 4;
        let vocab_size = 8;
        let attn_config = AttentionConfig {
            embed_dim,
            num_heads: 2,
            dropout_rate: 0.0,
        };

        let block = TransformerBlock::new(&attn_type, attn_config, embed_dim, vocab_size);

        assert!(block.is_ok());
    }

    #[test]
    fn test_transformer_block_forward_output_shape() {
        let attn_type = AttentionType::MultiHead;
        let embed_dim = 4;
        let vocab_size = 8;
        let attn_config = AttentionConfig {
            embed_dim,
            num_heads: 2,
            dropout_rate: 0.0,
        };

        let block = TransformerBlock::new(&attn_type, attn_config, embed_dim, vocab_size).unwrap();

        // Input shape: batch_size=2, seq_len=3, embed_dim=4
        let input = arr3(&[
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8],
             [0.9, 1.0, 1.1, 1.2]],
            [[1.2, 1.1, 1.0, 0.9],
             [0.8, 0.7, 0.6, 0.5],
             [0.4, 0.3, 0.2, 0.1]],
        ]);

        let output = block.forward(input, false).unwrap();

        // Output shape should be (batch_size, seq_len, vocab_size)
        assert_eq!(output.shape(), &[2, 3, vocab_size]);
    }
}