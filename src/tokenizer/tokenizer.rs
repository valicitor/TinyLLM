use serde::{Deserialize, Serialize};

use crate::tokenizer::TokenizerError;

/// Trait that all tokenizers must implement
#[typetag::serde]
pub trait Tokenizer {
    /// Encodes text into tokens
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError>;
    
    /// Decodes tokens back into text
    fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError>;
    
    /// Gets the vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Gets special tokens (e.g., [PAD], [UNK])
    fn special_tokens(&self) -> &[SpecialToken];
    
    /// Saves the tokenizer to disk
    fn save(&self, path: &str) -> Result<(), TokenizerError>;
}

/// Special token representation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpecialToken {
    pub id: usize,
    pub token: String,
}

impl PartialEq for SpecialToken {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}