// error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TinyLLMError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Invalid configuration: {0}")]
    Config(String),
}