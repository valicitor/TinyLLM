use super::{Tokenizer, SpecialToken, TokenizerError};
use std::collections::{HashMap};
use serde::{Serialize, Deserialize};
use std::fs;

/// Simple character-level tokenizer
#[derive(Debug, Serialize, Deserialize)]
pub struct SimpleTokenizer {
    stoi: HashMap<String, usize>,
    itos: HashMap<usize, String>,
    special_tokens: Vec<SpecialToken>,
}

impl SimpleTokenizer {
    /// Creates a new tokenizer from text
    pub fn from_text(text: &str) -> Result<Self, TokenizerError> {
        // Collect unique characters only once
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();

        // Safety check
        if chars.len() > usize::MAX - 10 {
            return Err(TokenizerError::VocabularySizeMismatch);
        }

        // Define special tokens
        let special_tokens = [
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
        ];

        let mut stoi = HashMap::with_capacity(chars.len() + special_tokens.len());
        let mut itos = HashMap::with_capacity(chars.len() + special_tokens.len());

        // Insert special tokens
        for &(tok, id) in &special_tokens {
            stoi.insert(tok.to_string(), id);
            itos.insert(id, tok.to_string());
        }

        // Add normal characters starting from next available ID
        let mut next_id = special_tokens.len();
        for c in chars {
            let s = c.to_string();
            // We already ensured uniqueness, so no need to check if key exists
            stoi.insert(s.clone(), next_id);
            itos.insert(next_id, s);
            next_id += 1;
        }

        // Convert to owned struct
        let special_tokens = special_tokens
            .iter()
            .map(|&(s, id)| SpecialToken { id, token: s.to_string() })
            .collect();

        Ok(Self {
            stoi,
            itos,
            special_tokens,
        })
    }
    
    /// Gets the unknown token ID
    pub fn unk_token_id(&self) -> Result<usize, TokenizerError> {
        self.stoi.get("[UNK]").copied().ok_or(TokenizerError::MissingUnkToken)
    }
}

#[typetag::serde]
impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let unk_id = self.unk_token_id()?;

        let encoded = text
            .chars()
            .map(|ch| {
                let key = ch.to_string();
                Ok(*self.stoi.get(&key).unwrap_or(&unk_id))
            })
            .collect::<Result<Vec<usize>, TokenizerError>>();

        encoded
    }
    
    fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError> {
        tokens.iter()
            .map(|id| {
                self.itos.get(id)
                    .ok_or_else(|| TokenizerError::InvalidTokenId(*id))
                    .map(|c| c.to_string())
            })
            .collect::<Result<Vec<String>, _>>()
            .map(|chars| chars.join(""))
    }
    
    fn vocab_size(&self) -> usize {
        self.stoi.len()
    }
    
    fn special_tokens(&self) -> &[SpecialToken] {
        &self.special_tokens
    }
    
    fn save(&self, path: &str) -> Result<(), TokenizerError> {
        let serialized = serde_json::to_string(self)
            .map_err(TokenizerError::serialization)?;
        fs::write(path, serialized)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_decode() {
        let text = "hello world";
        let tokenizer = SimpleTokenizer::from_text(text).unwrap();
        
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();
        
        assert_eq!(decoded, text);
    }
    
    #[test]
    fn test_unknown_token() {
        let tokenizer = SimpleTokenizer::from_text("abc").unwrap();
        let encoded = tokenizer.encode("abcd").unwrap();
        let unk_id = tokenizer.unk_token_id().unwrap();

        // Expect last token to be [UNK]
        assert_eq!(encoded.last(), Some(&unk_id));
    }
    
    #[test]
    fn test_special_tokens() {
        let tokenizer = SimpleTokenizer::from_text("abc").unwrap();
        assert_eq!(tokenizer.special_tokens().len(), 3);
        assert_eq!(tokenizer.vocab_size(), 6); // a, b, c + [PAD], [UNK], [CLS]
    }
}