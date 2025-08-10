use crate::{training::TrainingMetrics};
use crate::{utils, LanguageModel};
use super::{optimizer::Optimizer, metrics::Metrics, dataset::Dataset};
use anyhow::Ok;
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Invalid training configuration: {0}")]
    Config(String),
    #[error("IO error during training: {0}")]
    Io(#[from] std::io::Error),
    #[error("Training interrupted: {0}")]
    Interrupted(String),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub checkpoint_interval: usize,
    pub epochs: usize,
}

/// Main trainer struct that handles the training loop
pub struct Trainer {
    model: LanguageModel,
    optimizer: Box<dyn Optimizer>,
    metrics: Metrics,
    config: TrainingConfig,
}

impl Trainer {
    /// Creates a new trainer with the given components
    pub fn new(
        model: LanguageModel,
        optimizer: Box<dyn Optimizer>,
        metrics: Metrics,
        config: TrainingConfig,
    ) -> Self {
        Self {
            model,
            optimizer,
            metrics,
            config,
        }
    }

    /// Main training loop
    pub fn train(&mut self, dataset: &Dataset) -> Result<(), TrainingError> {
        for epoch in 0..self.config.epochs {
            let mut batch_iter = dataset.batch_iter(self.config.batch_size);
            
            while let Some(batch) = batch_iter.next() {
                let (loss, grad_norm) = self.train_step(&batch.input, &batch.target)?;
                
                self.metrics.record(TrainingMetrics {
                    epoch,
                    step: batch_iter.step(),
                    loss,
                    grad_norm,
                    learning_rate: self.optimizer.learning_rate(),
                    timestamp: std::time::SystemTime::now(),
                });
                
                // Checkpointing
                if batch_iter.step() % self.config.checkpoint_interval == 0 {
                    self.save_checkpoint(epoch)?;
                }
            }
            
            // End-of-epoch processing
            self.metrics.finalize_epoch(epoch);
        }
        
        Ok(())
    }

    /// Performs a single training step
    fn train_step(
        &mut self,
        input: &[Vec<usize>],
        target: &[usize],
    ) -> Result<(f32, f32), TrainingError> {        
        // 4. Forward pass through rest of the model (if needed)
        let logits = self.model.forward(input, true);
        
        // 5. Calculate loss
        let loss = utils::math::cross_entropy(&logits.view(), target)?;
        
        // 6. Backprop and optimizer step
        self.model.zero_grad();
        //elf.model.backward(/* pass necessary cache or activations */, target);
        let grad_norm = self.model.gradient_norm();
        self.optimizer.step(&mut self.model);
        
        Ok((loss, grad_norm))
    }

    /// Saves training checkpoint
    fn save_checkpoint(&self, epoch: usize) -> Result<(f32), TrainingError> {
        // ... implementation ...
        Ok(())
    }
}