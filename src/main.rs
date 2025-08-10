use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tinyllm::{
    model::{LanguageModel, ModelConfig},
    tokenizer::{Tokenizer, SimpleTokenizer, BPETokenizer},
    utils::io,
};

// #[derive(Parser)]
// #[command(name = "TinyLLM")]
// #[command(version = "0.1.0")]
// #[command(about = "A minimal LLM implementation in Rust", long_about = None)]
// struct Cli {
//     #[command(subcommand)]
//     command: Commands,
// }

//#[derive(Subcommand)]
//enum Commands {
    /// Train a new model
    // Train {
    //     /// Path to training data
    //     #[arg(short, long)]
    //     data_path: PathBuf,
        
    //     /// Path to save trained model
    //     #[arg(short, long)]
    //     output_path: Option<PathBuf>,
        
    //     /// Use BPE tokenizer instead of simple tokenizer
    //     #[arg(long)]
    //     use_bpe: bool,
        
    //     /// Number of training epochs
    //     #[arg(short, long, default_value_t = 10)]
    //     epochs: usize,
    // },
    
    /// Generate text from a prompt
    // Generate {
    //     /// Path to trained model
    //     #[arg(short, long)]
    //     model_path: PathBuf,
        
    //     /// Input prompt text
    //     prompt: String,
        
    //     /// Number of tokens to generate
    //     #[arg(short, long, default_value_t = 50)]
    //     length: usize,
        
    //     /// Temperature for sampling
    //     #[arg(short, long, default_value_t = 0.8)]
    //     temperature: f32,
    // },
    
    // /// Start an interactive session
    // Interactive {
    //     /// Path to trained model
    //     #[arg(short, long)]
    //     model_path: PathBuf,
        
    //     /// Temperature for sampling
    //     #[arg(short, long, default_value_t = 0.8)]
    //     temperature: f32,
    // },
    
    // /// Tokenize text using the model's tokenizer
    // Tokenize {
    //     /// Path to model
    //     #[arg(short, long)]
    //     model_path: PathBuf,
        
    //     /// Text to tokenize
    //     text: String,
    // },
//}

fn main() -> anyhow::Result<()> {
    //let cli = Cli::parse();
    
    //match cli.command {
        // Commands::Train {
        //     data_path,
        //     output_path,
        //     use_bpe,
        //     epochs,
        // } => {
        //     train_model(data_path, output_path, use_bpe, epochs)
        // }
        
        // Commands::Generate {
        //     model_path,
        //     prompt,
        //     length,
        //     temperature,
        // } => {
        //     generate_text(model_path, prompt, length, temperature)
        // }
        
        // Commands::Interactive {
        //     model_path,
        //     temperature,
        // } => {
        //     interactive_session(model_path, temperature)
        // }
        
        // Commands::Tokenize { model_path, text } => {
        //     tokenize_text(model_path, text)
        // }
    //}j
    Ok(())
}

// fn train_model(
//     data_path: PathBuf,
//     output_path: Option<PathBuf>,
//     use_bpe: bool,
//     epochs: usize,
// ) -> anyhow::Result<()> {
//     println!("Loading training data...");
//     let text = io::read_to_string(&data_path)?;
    
//     // Initialize tokenizer
//     println!("Initializing tokenizer...");
//     let tokenizer: Box<dyn Tokenizer> = if use_bpe {
//         println!("Training BPE tokenizer...");
//         let special_tokens = vec![
//             SpecialToken::new("[PAD]", 0),
//             SpecialToken::new("[UNK]", 1),
//             SpecialToken::new("[CLS]", 2),
//         ];
//         Box::new(BPETokenizer::train(&text, 5000, special_tokens, true)?)
//     } else {
//         println!("Using simple tokenizer...");
//         Box::new(SimpleTokenizer::from_text(&text)?)
//     };
    
//     // Model configuration
//     let config = ModelConfig {
//         vocab_size: tokenizer.vocab_size(),
//         embed_dim: 256,
//         num_heads: 8,
//         block_size: 512,
//         dropout_rate: 0.1,
//         layer_norm_eps: 1e-5,
//     };
    
//     // Initialize model
//     println!("Initializing model...");
//     let model = LanguageModel::new(config)?;
    
//     // Training configuration
//     let train_config = TrainingConfig {
//         epochs,
//         batch_size: 32,
//         learning_rate: 0.001,
//         checkpoint_interval: 100,
//     };
    
//     // Initialize trainer
//     println!("Starting training...");
//     let mut trainer = Trainer::new(
//         model,
//         Box::new(AdamOptimizer::new(train_config.learning_rate)),
//         Metrics::new("logs")?,
//         train_config,
//     );
    
//     // Train the model
//     trainer.train(&text, &*tokenizer)?;
    
//     // Save the model
//     let output_path = output_path.unwrap_or_else(|| PathBuf::from("model.bin"));
//     println!("Saving model to {}...", output_path.display());
//     trainer.save_model(&output_path)?;
    
//     // Save tokenizer if it was trained
//     if use_bpe {
//         let tokenizer_path = output_path.with_extension("tokenizer.json");
//         tokenizer.save(&tokenizer_path)?;
//     }
    
//     println!("Training completed successfully!");
//     Ok(())
// }

// fn generate_text(
//     model_path: PathBuf,
//     prompt: String,
//     length: usize,
//     temperature: f32,
// ) -> anyhow::Result<()> {
//     println!("Loading model...");
//     let model = io::deserialize_from_file::<LanguageModel>(&model_path)?;
    
//     // Try to load BPE tokenizer first, fall back to simple
//     let tokenizer_path = model_path.with_extension("tokenizer.json");
//     let tokenizer: Box<dyn Tokenizer> = if tokenizer_path.exists() {
//         Box::new(BPETokenizer::load(&tokenizer_path)?)
//     } else {
//         println!("No BPE tokenizer found, using simple tokenizer");
//         Box::new(SimpleTokenizer::load_default()?)
//     };
    
//     println!("Generating text...");
//     let output = model.generate(&*tokenizer, &prompt, length, temperature, None)?;
    
//     println!("\nGenerated text:");
//     println!("{}", output);
//     Ok(())
// }

// fn interactive_session(
//     model_path: PathBuf,
//     temperature: f32,
// ) -> anyhow::Result<()> {
//     println!("Loading model...");
//     let model = io::deserialize_from_file::<LanguageModel>(&model_path)?;
    
//     // Load tokenizer
//     let tokenizer_path = model_path.with_extension("tokenizer.json");
//     let tokenizer: Box<dyn Tokenizer> = if tokenizer_path.exists() {
//         Box::new(BPETokenizer::load(&tokenizer_path)?)
//     } else {
//         Box::new(SimpleTokenizer::load_default()?)
//     };
    
//     println!("Starting interactive session. Type 'quit' to exit.");
//     let mut rl = rustyline::Editor::<()>::new()?;
    
//     loop {
//         let readline = rl.readline(">> ");
//         match readline {
//             Ok(line) if line.trim().eq_ignore_ascii_case("quit") => break,
//             Ok(line) => {
//                 let output = model.generate(&*tokenizer, &line, 100, temperature, None)?;
//                 println!("{}", output);
//             },
//             Err(_) => break,
//         }
//     }
    
//     println!("Goodbye!");
//     Ok(())
// }

// fn tokenize_text(model_path: PathBuf, text: String) -> anyhow::Result<()> {
//     // Load tokenizer
//     let tokenizer_path = model_path.with_extension("tokenizer.json");
//     let tokenizer: Box<dyn Tokenizer> = if tokenizer_path.exists() {
//         Box::new(BPETokenizer::load(&tokenizer_path)?)
//     } else {
//         Box::new(SimpleTokenizer::load_default()?)
//     };
    
//     let tokens = tokenizer.encode(&text)?;
//     println!("Token IDs: {:?}", tokens);
    
//     // Show token -> ID mapping
//     println!("\nToken mapping:");
//     for (i, token_id) in tokens.iter().enumerate() {
//         let token = tokenizer.decode(&[*token_id])?;
//         println!("{:4}: {:6} -> '{}'", i, token_id, token);
//     }
    
//     Ok(())
// }