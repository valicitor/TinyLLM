mod attention;
mod model;
mod tokenizer;
mod checkpoint;
mod train;
mod utils;
mod error;

use std::{env, fs};
use model::ModelState;
use tokenizer::Tokenizer;
use train::train_from_file;

/// Configuration for the LLM model
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub block_size: usize,      // Context window size
    pub embed_dim: usize,       // Embedding dimension
    pub learning_rate: f32,     // Default learning rate
    pub dropout_rate: f32,      // Dropout probability
    pub temperature: f32,       // Sampling temperature
    pub top_k: Option<usize>,   // Top-k sampling (None for no limit)
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            embed_dim: 512,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            temperature: 0.8,
            top_k: Some(40),
        }
    }
}

/// Main CLI interface for the LLM
struct TinyLLM {
    model: ModelState,
    tokenizer: Tokenizer,
    config: ModelConfig,
}

impl TinyLLM {
    /// Initialize or load a model
    pub fn new(config: ModelConfig, data_path: &str) -> Self {
        let text = fs::read_to_string(data_path)
            .expect("Failed to load dataset");
        let tokenizer = Tokenizer::from_text(&text);
        
        let model = ModelState::load("model.bin")
            .unwrap_or_else(|| {
                println!("Initializing new model...");
                ModelState::new(
                    config.block_size,
                    tokenizer.vocab_size,
                    config.embed_dim,
                )
            });

        Self { model, tokenizer, config }
    }

    /// Train the model
    pub fn train(&mut self, epochs: usize, seq_len: usize) {
        train_from_file(
            &mut self.model,
            &self.tokenizer,
            "data/tiny_shakespeare.txt",
            epochs,
            self.config.learning_rate,
            seq_len,
            self.config.temperature,
            self.config.top_k,
            10, // checkpoint interval
        ).expect("Training failed");
        
        self.model.save("model.bin")
            .expect("Failed to save model");
    }

    /// Generate text from prompt
    pub fn generate(&self, prompt: &str, length: usize) -> String {
        self.model.generate(
            &self.tokenizer,
            prompt,
            length,
            self.config.temperature,
            self.config.top_k,
        )
    }

    /// Interactive generation session
    pub fn interactive(&self) {
        println!("Starting interactive session (type 'quit' to exit)");
        loop {
            let mut prompt = String::new();
            println!("\nEnter prompt:");
            std::io::stdin().read_line(&mut prompt).unwrap();
            
            let prompt = prompt.trim();
            if prompt.eq_ignore_ascii_case("quit") {
                break;
            }

            let output = self.generate(prompt, 100);
            println!("\nGenerated:\n{}", output);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = ModelConfig::default();

    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            let epochs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            let mut tinyllm = TinyLLM::new(config, "data/tiny_shakespeare.txt");
            tinyllm.train(epochs, 32);
        }
        Some("generate") => {
            let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("To be, or");
            let tinyllm = TinyLLM::new(config, "data/tiny_shakespeare.txt");
            let output = tinyllm.generate(prompt, 200);
            println!("Generated:\n{}", output);
        }
        Some("interactive") => {
            let tinyllm = TinyLLM::new(config, "data/tiny_shakespeare.txt");
            tinyllm.interactive();
        }
        _ => {
            println!("Usage: cargo run -- [train|generate|interactive]");
            println!("Example commands:");
            println!("  cargo run -- train 50");
            println!("  cargo run -- generate \"Your prompt\"");
            println!("  cargo run -- interactive");
        }
    }
}