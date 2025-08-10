use thiserror::Error;

use crate::tokenizer::TokenizerError;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    #[error("Initialization error: {0}")]
    InitializationError(String),
    
    #[error("Forward pass error: {0}")]
    ForwardError(String),
    
    #[error("Backward pass error: {0}")]
    BackwardError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Generation error: {0}")]
    GenerationError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Sampling error: {0}")]
    SamplingError(String),
}

impl ModelError {
    pub fn serialization<E: std::fmt::Display>(error: E) -> Self {
        ModelError::SerializationError(error.to_string())
    }
}