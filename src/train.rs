use crate::model::ModelState;
use crate::tokenizer::Tokenizer;
use crate::checkpoint::{Checkpoint, save_checkpoint, load_checkpoint};
use crate::utils::cross_entropy;
use crate::attention::Attention;
use ndarray::Array2;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Adam optimizer state for a single parameter matrix
#[derive(Serialize, Deserialize, Clone)]
struct AdamState {
    m: Array2<f32>,  // First moment estimate
    v: Array2<f32>,  // Second moment estimate
}

impl AdamState {
    fn new(shape: (usize, usize)) -> Self {
        Self {
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
        }
    }
}

/// Complete Adam optimizer state for all model parameters
#[derive(Serialize, Deserialize)]
pub struct OptimizerState {
    embed: AdamState,
    linear: AdamState,
    attention: AttentionOptimizerState,
    step: usize,
}

/// Optimizer state specific to attention parameters
#[derive(Serialize, Deserialize)]
struct AttentionOptimizerState {
    wq: AdamState,
    wk: AdamState,
    wv: AdamState,
    wo: AdamState,
}

impl OptimizerState {
    /// Get a complete copy of the optimizer state
    pub fn get_state(&self) -> Self {
        OptimizerState {
            embed: AdamState {
                m: self.embed.m.clone(),
                v: self.embed.v.clone(),
            },
            linear: AdamState {
                m: self.linear.m.clone(),
                v: self.linear.v.clone(),
            },
            attention: AttentionOptimizerState {
                wq: AdamState {
                    m: self.attention.wq.m.clone(),
                    v: self.attention.wq.v.clone(),
                },
                wk: AdamState {
                    m: self.attention.wk.m.clone(),
                    v: self.attention.wk.v.clone(),
                },
                wv: AdamState {
                    m: self.attention.wv.m.clone(),
                    v: self.attention.wv.v.clone(),
                },
                wo: AdamState {
                    m: self.attention.wo.m.clone(),
                    v: self.attention.wo.v.clone(),
                },
            },
            step: self.step,
        }
    }

    /// Restore optimizer state from saved state
    pub fn set_state(&mut self, state: OptimizerState) {
        self.embed.m = state.embed.m;
        self.embed.v = state.embed.v;
        
        self.linear.m = state.linear.m;
        self.linear.v = state.linear.v;
        
        self.attention.wq.m = state.attention.wq.m;
        self.attention.wq.v = state.attention.wq.v;
        
        self.attention.wk.m = state.attention.wk.m;
        self.attention.wk.v = state.attention.wk.v;
        
        self.attention.wv.m = state.attention.wv.m;
        self.attention.wv.v = state.attention.wv.v;
        
        self.attention.wo.m = state.attention.wo.m;
        self.attention.wo.v = state.attention.wo.v;
        
        self.step = state.step;
    }

    pub fn new(model: &ModelState) -> Self {
        Self {
            embed: AdamState::new(model.embed.value.dim()),
            linear: AdamState::new(model.linear.value.dim()),
            attention: AttentionOptimizerState {
                wq: AdamState::new(model.attention.wq.value.dim()),
                wk: AdamState::new(model.attention.wk.value.dim()),
                wv: AdamState::new(model.attention.wv.value.dim()),
                wo: AdamState::new(model.attention.wo.value.dim()),
            },
            step: 0,
        }
    }
}

/// Main training function
pub fn train_from_file(
    model: &mut ModelState,
    tokenizer: &Tokenizer,
    filepath: &str,
    epochs: usize,
    lr: f32,
    seq_len: usize,
    temperature: f32,
    top_k: Option<usize>,
    checkpoint_interval: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut optimizer = OptimizerState::new(model);
    let mut rng = rand::thread_rng();

    // Attempt to load from checkpoint
    let start_epoch = load_checkpoint_if_exists(model, &mut optimizer)?;

    // Prepare file reader
    let file = File::open(filepath)?;
    let mut reader = BufReader::new(file);
    let file_len = reader.get_ref().metadata()?.len();

    // Training loop
    for epoch in start_epoch..epochs {
        let (input, target) = get_training_batch(&mut reader, &mut rng, file_len, tokenizer, seq_len)?;
        
        let (loss, grad_norm) = train_step(model, &input, &target);
        
        adam_optimizer_step(model, &mut optimizer, lr, 0.9, 0.999, 1e-8);
        
        log_progress(epoch, loss, grad_norm);
        
        handle_checkpoints(epoch, model, &optimizer, loss, checkpoint_interval)?;
        
        generate_samples_if_needed(epoch, model, tokenizer, temperature, top_k);
    }

    if epochs > 0 {
        cleanup_checkpoints()?;
    }

    Ok(())
}

/// Helper function for a single training step
fn train_step(model: &mut ModelState, input: &[usize], target: &[usize]) -> (f32, f32) {
    let cache = model.forward(input);
    let loss = cross_entropy(&cache.logits, target);
    
    model.zero_grad();
    model.backward(&cache, target);
    
    let grad_norm = model.gradient_norm();
    (loss, grad_norm)
}

/// Adam optimizer update step
fn adam_optimizer_step(
    model: &mut ModelState,
    optimizer: &mut OptimizerState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    optimizer.step += 1;
    let t = optimizer.step as f32;
    
    // Update embeddings
    update_parameter(
        &mut model.embed.value,
        &mut model.embed.grad,
        &mut optimizer.embed.m,
        &mut optimizer.embed.v,
        lr, beta1, beta2, epsilon, t
    );
    
    // Update linear layer
    update_parameter(
        &mut model.linear.value,
        &mut model.linear.grad,
        &mut optimizer.linear.m,
        &mut optimizer.linear.v,
        lr, beta1, beta2, epsilon, t
    );
    
    // Update attention parameters
    update_attention_parameters(
        &mut model.attention,
        &mut optimizer.attention,
        lr, beta1, beta2, epsilon, t
    );
}

/// Update a single parameter matrix using Adam
fn update_parameter(
    param: &mut Array2<f32>,
    grad: &mut Array2<f32>,
    m: &mut Array2<f32>,
    v: &mut Array2<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: f32,
) {
    // Update first moment estimate (m)
    *m = &*m * beta1 + &*grad * (1.0 - beta1);
    
    // Update second moment estimate (v)
    *v = &*v * beta2 + grad.mapv(|g| g * g) * (1.0 - beta2);
    
    // Compute bias-corrected estimates
    let m_hat = &*m / (1.0 - beta1.powf(t));
    let v_hat = &*v / (1.0 - beta2.powf(t));
    
    // Compute and apply update
    let denom = v_hat.mapv(|v| (v.sqrt() + epsilon).max(1e-8));
    let update = (m_hat / denom) * lr;
    *param -= &update;
    
    // Reset gradients
    grad.fill(0.0);
}

/// Update all attention parameters
fn update_attention_parameters(
    attention: &mut Attention,
    optimizer: &mut AttentionOptimizerState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: f32,
) {
    update_parameter(
        &mut attention.wq.value,
        &mut attention.wq.grad,
        &mut optimizer.wq.m,
        &mut optimizer.wq.v,
        lr, beta1, beta2, epsilon, t
    );
    
    update_parameter(
        &mut attention.wk.value,
        &mut attention.wk.grad,
        &mut optimizer.wk.m,
        &mut optimizer.wk.v,
        lr, beta1, beta2, epsilon, t
    );
    
    update_parameter(
        &mut attention.wv.value,
        &mut attention.wv.grad,
        &mut optimizer.wv.m,
        &mut optimizer.wv.v,
        lr, beta1, beta2, epsilon, t
    );
    
    update_parameter(
        &mut attention.wo.value,
        &mut attention.wo.grad,
        &mut optimizer.wo.m,
        &mut optimizer.wo.v,
        lr, beta1, beta2, epsilon, t
    );
}

/// Helper functions for training pipeline
fn load_checkpoint_if_exists(
    model: &mut ModelState,
    optimizer: &mut OptimizerState,
) -> Result<usize, Box<dyn std::error::Error>> {
    let checkpoint_path = "checkpoint_latest.bin";
    if Path::new(checkpoint_path).exists() {
        let checkpoint = load_checkpoint(checkpoint_path)?;
        model.set_state(checkpoint.model_state);
        optimizer.set_state(checkpoint.optimizer_state);
        Ok(checkpoint.epoch + 1)
    } else {
        Ok(0)
    }
}

fn get_training_batch(
    reader: &mut BufReader<File>,
    rng: &mut impl Rng,
    file_len: u64,
    tokenizer: &Tokenizer,
    seq_len: usize,
) -> Result<(Vec<usize>, Vec<usize>), Box<dyn std::error::Error>> {
    let offset = rng.gen_range(0..(file_len - (seq_len as u64 + 1)));
    reader.seek(SeekFrom::Start(offset))?;
    let tokens = read_tokens(reader, tokenizer, seq_len + 1)?;
    Ok((tokens[0..seq_len].to_vec(), tokens[1..seq_len + 1].to_vec()))
}

fn log_progress(epoch: usize, loss: f32, grad_norm: f32) {
    println!("Epoch {}: loss = {:.4}, grad norm = {:.4}", epoch + 1, loss, grad_norm);
}

fn handle_checkpoints(
    epoch: usize,
    model: &ModelState,
    optimizer: &OptimizerState,
    loss: f32,
    checkpoint_interval: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if (epoch + 1) % checkpoint_interval == 0 {
        let checkpoint = Checkpoint {
            model_state: model.get_state(),
            optimizer_state: optimizer.get_state(),
            epoch,
            loss,
        };
        
        save_checkpoint("checkpoint_latest.bin", &checkpoint)?;
        
        let path = format!("checkpoint_epoch_{}.bin", epoch + 1);
        save_checkpoint(&path, &checkpoint)?;
    }
    Ok(())
}

fn generate_samples_if_needed(
    epoch: usize,
    model: &ModelState,
    tokenizer: &Tokenizer,
    temperature: f32,
    top_k: Option<usize>,
) {
    if (epoch + 1) % 10 == 0 {
        let prompt = "To be, or";
        let sample = model.generate(tokenizer, prompt, 50, temperature, top_k);
        println!("Sample generation at epoch {}:\n{}", epoch + 1, sample);
    }
}

fn cleanup_checkpoints() -> Result<(), Box<dyn std::error::Error>> {
    // Remove latest checkpoint
    let _ = fs::remove_file("checkpoint_latest.bin");
    
    // Remove all epoch checkpoints
    delete_matching_files("checkpoint_epoch_", ".bin")?;
    
    Ok(())
}

fn delete_matching_files(prefix: &str, suffix: &str) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(".")? {
        let entry = entry?;
        let path = entry.path();
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename.starts_with(prefix) && filename.ends_with(suffix) {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

// Helper functions for token reading (unchanged from original)
fn read_tokens(reader: &mut BufReader<File>, tokenizer: &Tokenizer, n: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut buf = vec![0u8; n * 4];
    let read_bytes = reader.read(&mut buf)?;
    let text = clean_sample(&String::from_utf8_lossy(&buf[..read_bytes]));
    let tokens = tokenizer.encode(&text);
    Ok(tokens[..n.min(tokens.len())].to_vec())
}

fn clean_sample(sample: &str) -> String {
    sample.chars()
        .filter(|c| c.is_ascii_graphic() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}