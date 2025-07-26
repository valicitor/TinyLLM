use std::fmt;

#[derive(Debug)]
pub enum TinyLLMError {
    Io(std::io::Error),
    Serialization(bincode::Error),
    Training(String),
    // Add more variants as needed
}

impl fmt::Display for TinyLLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TinyLLMError::Io(e) => write!(f, "IO error: {}", e),
            TinyLLMError::Serialization(e) => write!(f, "Serialization error: {}", e),
            TinyLLMError::Training(s) => write!(f, "Training error: {}", s),
        }
    }
}

impl std::error::Error for TinyLLMError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TinyLLMError::Io(e) => Some(e),
            TinyLLMError::Serialization(e) => Some(e),
            TinyLLMError::Training(_) => None,
        }
    }
}

impl From<std::io::Error> for TinyLLMError {
    fn from(err: std::io::Error) -> Self {
        TinyLLMError::Io(err)
    }
}

impl From<bincode::Error> for TinyLLMError {
    fn from(err: bincode::Error) -> Self {
        TinyLLMError::Serialization(err)
    }
}

impl From<String> for TinyLLMError {
    fn from(err: String) -> Self {
        TinyLLMError::Training(err)
    }
}

impl From<&str> for TinyLLMError {
    fn from(err: &str) -> Self {
        TinyLLMError::Training(err.to_string())
    }
}