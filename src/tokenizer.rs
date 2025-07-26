use std::collections::HashMap;

#[derive(Debug)]
pub struct Tokenizer {
    pub stoi: HashMap<char, usize>,
    pub itos: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let stoi: HashMap<char, usize> = chars.iter().cloned().enumerate().map(|(i, c)| (c, i)).collect();
        let itos: HashMap<usize, char> = stoi.iter().map(|(c, i)| (*i, *c)).collect();
        let vocab_size = stoi.len();

        Tokenizer { stoi, itos, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.stoi.get(&c).unwrap_or(&0)).collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&i| self.itos.get(&i).unwrap_or(&'?')).collect()
    }
}