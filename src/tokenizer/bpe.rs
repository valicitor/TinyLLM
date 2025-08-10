use std::collections::{HashMap, HashSet};
use std::fs;
use serde::{Serialize, Deserialize};

use crate::tokenizer::{Tokenizer, TokenizerError, SpecialToken};

/// Byte Pair Encoding tokenizer
#[derive(Debug, Serialize, Deserialize)]
pub struct BPETokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: HashMap<usize, String>,
    merge_ranks: HashMap<(usize, usize), usize>,
    special_tokens: Vec<SpecialToken>,
    vocab_size: usize,
    max_token_length: usize,
    lowercase: bool,
}

impl BPETokenizer {
    pub fn train(
        text: &str,
        vocab_size: usize,
        special_tokens: Vec<SpecialToken>,
        lowercase: bool,
    ) -> Result<Self, TokenizerError> {
        let normalized_text = if lowercase { text.to_lowercase() } else { text.to_string() };
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut inv_vocab: HashMap<usize, String> = HashMap::new();
        let mut word_counts: HashMap<Vec<usize>, u32> = HashMap::new();

        let mut next_id = 0;
        for token in &special_tokens {
            vocab.insert(token.token.clone(), token.id);
            inv_vocab.insert(token.id, token.token.clone());
            next_id = next_id.max(token.id + 1);
        }

        let chars: HashSet<char> = normalized_text.chars().collect();
        for c in chars {
            let ch = c.to_string();
            if !vocab.contains_key(&ch) {
                vocab.insert(ch.clone(), next_id);
                inv_vocab.insert(next_id, ch);
                next_id += 1;
            }
        }

        let words: Vec<&str> = normalized_text.split_whitespace().collect();
        for word in words {
            let word_ids: Vec<usize> = word.chars().map(|c| vocab[&c.to_string()]).collect();
            *word_counts.entry(word_ids).or_insert(0) += 1;
        }

        let mut merge_ranks: HashMap<(usize, usize), usize> = HashMap::new();
        let mut merge_order = 0;

        while vocab.len() < vocab_size {
            // Pre-allocate pair_counts with a reasonable capacity
            let mut pair_counts: HashMap<(usize, usize), u32> = HashMap::with_capacity(1024);

            // Count pairs
            for (word, &count) in &word_counts {
                for pair in word.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_insert(0) += count;
                }
            }

            // Find the best pair (early exit if no pairs left)
            // std::option::Option<(&(usize, usize), &u32)>
            let Some((&best_pair, _)) = pair_counts.iter().max_by_key(|&count| count) else {
                break;
            };

            let best_pair = best_pair; // Dereference since we know it exists

            // Create the new token
            let part1 = &inv_vocab[&best_pair.0];
            let part2 = &inv_vocab[&best_pair.1];
            let new_token = format!("{}{}", part1, part2);

            // Update vocab and inverse vocab
            vocab.insert(new_token.clone(), next_id);
            inv_vocab.insert(next_id, new_token);
            merge_ranks.insert(best_pair, merge_order);
            merge_order += 1;

            // Process words to apply the merge
            let mut updated_word_counts = HashMap::with_capacity(word_counts.len());
            for (word, count) in word_counts {
                let mut new_word = Vec::with_capacity(word.len()); // Pre-allocate
                let mut i = 0;

                while i < word.len() {
                    if i + 1 < word.len() && word[i] == best_pair.0 && word[i + 1] == best_pair.1 {
                        new_word.push(next_id);
                        i += 2;
                    } else {
                        new_word.push(word[i]);
                        i += 1;
                    }
                }

                *updated_word_counts.entry(new_word).or_insert(0) += count;
            }

            word_counts = updated_word_counts;
            next_id += 1;
        }

        Ok(Self {
            vocab,
            inv_vocab,
            merge_ranks,
            special_tokens,
            vocab_size,
            max_token_length: 16,
            lowercase,
        })
    }

    fn apply_merges(&self, tokens: &[usize]) -> Vec<usize> {
        let mut tokens = tokens.to_vec();
        loop {
            let mut best_rank = None;
            let mut best_index = 0;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_index = i;
                    }
                }
            }

            if let Some(_) = best_rank {
                let pair = (tokens[best_index], tokens[best_index + 1]);
                let new_token = self.vocab[&format!("{}{}", self.inv_vocab[&pair.0], self.inv_vocab[&pair.1])];
                tokens.splice(best_index..best_index + 2, [new_token]);
            } else {
                break;
            }
        }
        tokens
    }
}

#[typetag::serde]
impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let normalized = if self.lowercase { text.to_lowercase() } else { text.to_string() };
        let mut result = Vec::new();

        for word in normalized.split_whitespace() {
            let mut char_ids = Vec::new();
            for c in word.chars() {
                let ch = c.to_string();
                if let Some(&id) = self.vocab.get(&ch) {
                    char_ids.push(id);
                } else {
                    char_ids.push(self.special_tokens.iter().find(|t| t.token == "[UNK]")
                        .ok_or(TokenizerError::UnknownToken(ch))?.id);
                }
            }
            result.extend(self.apply_merges(&char_ids));
        }

        Ok(result)
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError> {
        let mut result = String::new();
        for (i, &id) in tokens.iter().enumerate() {
            if i > 0 { result.push(' '); } // simple separator
            let token = self.inv_vocab.get(&id).ok_or(TokenizerError::InvalidTokenId(id))?;
            result.push_str(token);
        }
        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn special_tokens(&self) -> &[SpecialToken] {
        &self.special_tokens
    }

    fn save(&self, path: &str) -> Result<(), TokenizerError> {
        let serialized = serde_json::to_string(self)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;
        fs::write(path, serialized)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{tokenizer::{BPETokenizer, SpecialToken}, Tokenizer};
    fn setup_basic_tokenizer() -> BPETokenizer {
        let special_tokens = vec![
            SpecialToken { token: "[PAD]".into(), id: 0 },
            SpecialToken { token: "[UNK]".into(), id: 1 },
        ];

        BPETokenizer::train(
            "hello hello world world",
            50,
            special_tokens,
            true,
        ).unwrap()
    }

    #[test]
    fn test_train_vocab_size() {
        let tokenizer = setup_basic_tokenizer();
        assert!(tokenizer.vocab_size() <= 50);
        assert!(tokenizer.vocab.contains_key("h"));
        assert!(tokenizer.vocab.contains_key("e"));
        assert!(tokenizer.vocab.contains_key("hello"));
    }

    #[test]
    fn test_encode_decode_symmetry() {
        let tokenizer = setup_basic_tokenizer();
        let text = "hello world";
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert!(
            decoded.replace(" ", "").contains("hello") &&
            decoded.replace(" ", "").contains("world"),
            "Decoded: {}",
            decoded
        );
    }

    #[test]
    fn test_special_token_unknown() {
        let tokenizer = setup_basic_tokenizer();
        let encoded = tokenizer.encode("💥").unwrap(); // emoji not in base vocab
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 1); // [UNK] id
    }

    #[test]
    fn test_lowercase_preprocessing() {
        let tokenizer = BPETokenizer::train(
            "Test TeSt",
            20,
            vec![
                SpecialToken { token: "[PAD]".into(), id: 0 },
                SpecialToken { token: "[UNK]".into(), id: 1 },
            ],
            true,
        ).unwrap();

        let encoded1 = tokenizer.encode("test").unwrap();
        let encoded2 = tokenizer.encode("TEST").unwrap();
        assert_eq!(encoded1, encoded2);
    }

    #[test]
    fn test_merge_application() {
        let tokenizer = setup_basic_tokenizer();
        let word = "hello";
        let ids = tokenizer.encode(word).unwrap();
        assert!(ids.len() < word.len()); // should have merged some chars
    }

    #[test]
    fn test_vocab_consistency() {
        let tokenizer = setup_basic_tokenizer();
        for (&id, token) in &tokenizer.inv_vocab {
            assert_eq!(tokenizer.vocab[token], id);
        }
    }

    #[test]
    fn test_special_tokens_round_trip() {
        let mut tokenizer = BPETokenizer::new(HashMap::new(), HashMap::new());
        tokenizer.add_special_token_auto("[PAD]");
        tokenizer.add_special_token_auto("[EOS]");

        let encoded = tokenizer.encode("[PAD] hello [EOS]").unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert!(decoded.contains("[PAD]"));
        assert!(decoded.contains("[EOS]"));
    }
}