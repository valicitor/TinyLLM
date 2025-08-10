use crate::tokenizer::Tokenizer;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Dataset error: {0}")]
    Generic(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Dataset for training
pub struct Dataset {
    tokens: Vec<usize>,
    seq_len: usize,
}

impl Dataset {
    pub fn from_file(path: &str, tokenizer: &Box<dyn Tokenizer>, seq_len: usize) -> Result<(), DatasetError> {
        // Load and tokenize data
        // ...
        Ok(())
    }

    pub fn batch_iter(&self, batch_size: usize) -> BatchIterator {
        BatchIterator::new(&self.tokens, batch_size, self.seq_len)
    }
}

/// Iterator over training batches
pub struct BatchIterator<'a> {
    tokens: &'a [usize],
    batch_size: usize,
    seq_len: usize,
    current_pos: usize,
}

impl<'a> BatchIterator<'a> {
    pub fn new(tokens: &'a [usize], batch_size: usize, seq_len: usize) -> Self {
        Self {
            tokens,
            batch_size,
            seq_len,
            current_pos: 0,
        }
    }

    pub fn step(&self) -> usize {
        self.current_pos
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        // Return None if we've reached the end
        if self.current_pos + self.seq_len + 1 >= self.tokens.len() {
            return None;
        }

        // Create batch
        let mut batch = Batch {
            input: Vec::with_capacity(self.batch_size * self.seq_len),
            target: Vec::with_capacity(self.batch_size * self.seq_len),
        };

        // Fill batch (simplified example)
        for _ in 0..self.batch_size {
            if self.current_pos + self.seq_len + 1 >= self.tokens.len() {
                break;
            }

            batch.input.extend_from_slice(
                &self.tokens[self.current_pos..self.current_pos + self.seq_len]
            );
            batch.target.extend_from_slice(
                &self.tokens[self.current_pos + 1..self.current_pos + self.seq_len + 1]
            );

            self.current_pos += self.seq_len;
        }

        if batch.input.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// A single training batch
pub struct Batch {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
}