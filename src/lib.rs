//! TinyLLM - A minimal but capable LLM implementation in Rust
//!
//! This library provides:
//! - Transformer-based language model implementation
//! - Multiple tokenizer options
//! - Training utilities
//! - Model serialization

pub mod model;
pub mod tokenizer;
pub mod training;
pub mod utils;
pub mod error;

// Re-export commonly used types at the crate root
pub use model::LanguageModel;
pub use tokenizer::{Tokenizer, SimpleTokenizer};
//pub use training::{Trainer, TrainingConfig};

use crate::model::ModelError;

// Core generation trait
// pub trait TextGenerator {
//     fn generate(
//         &self,
//         tokenizer: Box<dyn Tokenizer>,
//         prompt: &str,
//         max_length: usize,
//         temperature: f32,
//         top_k: Option<usize>,
//         top_p: Option<f32>,
//     ) -> Result<String, ModelError>;
// }

// impl TextGenerator for LanguageModel {
//     fn generate(
//         &self,
//         tokenizer: Box<dyn Tokenizer>,
//         prompt: &str,
//         max_length: usize,
//         temperature: f32,
//         top_k: Option<usize>,
//         top_p: Option<f32>,
//     ) -> Result<String, ModelError> {
//         // Delegate to the model's implementation
//         self.generate_text(tokenizer, prompt, max_length, temperature, top_k, top_p)
//     }
// }