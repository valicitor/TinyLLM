//! Neural network model implementation

mod attention;
mod config;
mod error;
mod layers;
mod positional;
mod transformer;

pub use attention::Attention;
pub use attention::MultiHeadAttention;
pub use config::ModelConfig;
pub use error::ModelError;
pub use layers::{Embedding, FeedForwardNetwork, LayerNorm, Linear};
use ndarray::Array3;
pub use positional::{
    LearnedPositionalEncoding, RotaryPositionEmbedding, SinusoidalPositionalEncoding,
};
use serde::{Deserialize, Serialize};
pub use transformer::TransformerBlock;

use crate::model::positional::PositionalEncoding;
use crate::utils::AttentionType;
use crate::utils::PositionalEncodingType;

/// Complete language model architecture
#[derive(Serialize, Deserialize)]
pub struct LanguageModel {
    pub(crate) config: ModelConfig,
    pub(crate) embedding: Embedding,
    pub(crate) positional_encoding: Box<dyn PositionalEncoding>,
    pub(crate) blocks: Vec<TransformerBlock>, // Now using the separate transformer
    pub(crate) lm_head: Linear,
}

impl LanguageModel {
    /// Creates a new model with initialized weights
    pub fn new(config: ModelConfig) -> Result<Self, ModelError> {
        //config.validate()?;

        let embedding = Embedding::new(config.vocab_size, config.embed_dim)?;
        let positional_encoding: Box<dyn PositionalEncoding> = Self::build_positional_encoding(
            &config.positional_encoding,
            config.embed_dim,
            config.block_size,
        );

        // Create the transformer blocks
        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                &config.attention_type,
                config.attn_config.clone(),
                config.embed_dim,
                config.vocab_size,
            )?); // Shared config
        }

        let lm_head = Linear::new(config.embed_dim, config.vocab_size)?;

        Ok(Self {
            config,
            embedding,
            positional_encoding,
            blocks,
            lm_head,
        })
    }

    fn build_positional_encoding(
        positional_encoding: &PositionalEncodingType,
        embed_dim: usize,
        block_size: usize,
    ) -> Box<dyn PositionalEncoding> {
        match positional_encoding {
            PositionalEncodingType::Sinusoidal => {
                Box::new(SinusoidalPositionalEncoding::new(block_size, embed_dim))
            }
            PositionalEncodingType::Learned => {
                Box::new(LearnedPositionalEncoding::new(block_size, embed_dim).unwrap())
            }
            PositionalEncodingType::Rotary => {
                Box::new(RotaryPositionEmbedding::new(embed_dim, block_size).unwrap())
            }
        }
    }

    fn forward(&self, token_ids: &[Vec<usize>]) -> Result<Array3<f32>, ModelError> {
        // Step 1: Embed tokens and positions
        let token_embeddings = self.embedding.forward(token_ids);
        let positional_embeddings = self.positional_encoding.forward(&token_embeddings);

        // Step 2: Pass through transformer blocks
        let mut hidden_states = positional_embeddings;
        for block in &self.blocks {
            hidden_states = block
                .forward(hidden_states, false)
                .expect("Block forward failed");
        }

        // Step 3: Project to vocabulary and flatten
        Ok(self.lm_head.forward(hidden_states.view()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::{
        attention::AttentionConfig, config::ModelConfig, error::ModelError, positional::{
            LearnedPositionalEncoding, RotaryPositionEmbedding, SinusoidalPositionalEncoding,
        }
    }, utils::OptimizerType};
    use ndarray::{Array3, array};

    #[test]
    fn test_model_initialization() -> Result<(), ModelError> {
        let config = ModelConfig {
            embed_dim: 64,
            vocab_size: 1000,
            block_size: 128,
            num_layers: 2,
            n_heads: 4,
            d_ff: 256,
            positional_encoding: PositionalEncodingType::Sinusoidal,
            attention_type: AttentionType::MultiHead,
            attn_config: AttentionConfig {
                embed_dim: 64,
                num_heads: 4,
                dropout_rate: 0.1,
            },
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        };

        let model = LanguageModel::new(config)?;

        assert_eq!(model.blocks.len(), 2);
        //assert_eq!(model.embedding.vocab_size(), 1000);
        //assert_eq!(model.embedding.embed_dim(), 64);
        Ok(())
    }

    #[test]
    fn test_forward_pass_shape() -> Result<(), ModelError> {
        let config = ModelConfig {
            embed_dim: 64,
            vocab_size: 1000,
            block_size: 128,
            num_layers: 2,
            n_heads: 4,
            d_ff: 256,
            positional_encoding: PositionalEncodingType::Learned,
            attention_type: AttentionType::MultiHead,
            attn_config: AttentionConfig {
                embed_dim: 64,
                num_heads: 4,
                dropout_rate: 0.1,
            },
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        };

        let model = LanguageModel::new(config)?;
        let token_ids = vec![
            vec![1, 2, 3, 4], // Batch 1
            vec![5, 6, 7, 8], // Batch 2
        ];

        let output = model.forward(&token_ids)?;

        // Should return shape [batch_size, seq_len, vocab_size]
        assert_eq!(output.shape(), &[2, 4, 1000]);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Block forward failed")]
    fn test_invalid_sequence_length() {
        let config = ModelConfig {
            embed_dim: 64,
            vocab_size: 1000,
            block_size: 8, // Small block size
            num_layers: 1,
            n_heads: 4,
            d_ff: 256,
            positional_encoding: PositionalEncodingType::Learned,
            attention_type: AttentionType::MultiHead,
            attn_config: AttentionConfig {
                embed_dim: 64,
                num_heads: 4,
                dropout_rate: 0.1,
            },
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        };

        let model = LanguageModel::new(config).unwrap();
        let token_ids = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9], // Exceeds block_size
        ];

        let _ = model.forward(&token_ids).unwrap();
    }

    #[test]
    fn test_positional_encoding_types() -> Result<(), ModelError> {
        let test_cases = vec![
            PositionalEncodingType::Sinusoidal,
            PositionalEncodingType::Learned,
            PositionalEncodingType::Rotary,
        ];

        for encoding_type in test_cases {
            let config = ModelConfig {
                embed_dim: 32,
                vocab_size: 500,
                block_size: 64,
                num_layers: 1,
                n_heads: 4,
                d_ff: 128,
                positional_encoding: encoding_type.clone(),
                attention_type: AttentionType::MultiHead,
                attn_config: AttentionConfig {
                    embed_dim: 32,
                    num_heads: 4,
                    dropout_rate: 0.1,
                },
                dropout: 0.1,
                layer_norm_eps: 1e-5,
            };

            let model = LanguageModel::new(config)?;
            let token_ids = vec![vec![1, 2, 3]];
            let output = model.forward(&token_ids)?;

            assert_eq!(output.shape(), &[1, 3, 500]);
        }
        Ok(())
    }

    #[test]
    fn test_empty_batch() -> Result<(), ModelError> {
        let config = ModelConfig {
            embed_dim: 64,
            vocab_size: 1000,
            block_size: 128,
            num_layers: 2,
            n_heads: 4,
            d_ff: 256,
            positional_encoding: PositionalEncodingType::Learned,
            attention_type: AttentionType::MultiHead,
            attn_config: AttentionConfig {
                embed_dim: 64,
                num_heads: 4,
                dropout_rate: 0.1,
            },
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        };

        let model = LanguageModel::new(config)?;
        let token_ids: Vec<Vec<usize>> = vec![];

        let output = model.forward(&token_ids)?;
        assert_eq!(output.shape(), &[0, 0, 1000]); // Empty batch
        Ok(())
    }

    #[test]
    fn test_variable_length_sequences() -> Result<(), ModelError> {
        let config = ModelConfig {
            embed_dim: 32,
            vocab_size: 500,
            block_size: 64,
            num_layers: 1,
            n_heads: 4,
            d_ff: 128,
            positional_encoding: PositionalEncodingType::Learned,
            attention_type: AttentionType::MultiHead,
            attn_config: AttentionConfig {
                embed_dim: 32,
                num_heads: 4,
                dropout_rate: 0.1,
            },
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        };

        let model = LanguageModel::new(config)?;
        let token_ids = vec![
            vec![1, 2],       // Short sequence
            vec![3, 4, 5, 6], // Medium sequence
            vec![7],          // Very short sequence
        ];

        let output = model.forward(&token_ids)?;
        assert_eq!(output.shape(), &[3, 4, 500]); // Padded to max length in batch
        Ok(())
    }
}
