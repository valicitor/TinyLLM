use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Unknown token: {0}")]
    UnknownToken(String),
    
    #[error("Invalid token ID: {0}")]
    InvalidTokenId(usize),
    
    #[error("Tokenizer not initialized")]
    NotInitialized,
    
    #[error("Vocabulary size mismatch")]
    VocabularySizeMismatch,
    
    #[error("Missing Unknown Token")]
    MissingUnkToken
}

impl TokenizerError {
    pub fn serialization(e: impl fmt::Display) -> Self {
        TokenizerError::Serialization(e.to_string())
    }
}