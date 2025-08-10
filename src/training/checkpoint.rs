use super::{Optimizer};
use crate::{training::AdamOptimizer, utils::io::{self, IoError}, LanguageModel};
use serde::{Serialize, Deserialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub model: LanguageModel,
    #[serde(skip)]  // We'll handle optimizer separately
    pub optimizer: Box<dyn Optimizer>,
    pub trainer_state: TrainerState,
}

#[derive(Serialize, Deserialize)]
pub struct TrainerState {
    pub epoch: usize,
    pub step: usize,
    // ... other training state
}

/// Error type for checkpoint operations
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("IO error: {0}")]
    Io(#[from] IoError),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Optimizer state error: {0}")]
    OptimizerState(String),
}

/// Saves a training checkpoint
pub fn save_checkpoint(
    path: impl AsRef<Path>,
    checkpoint: &Checkpoint,
) -> Result<(), CheckpointError> {
    let path = path.as_ref();
    
    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(IoError::StdIo)?;
    }

    // Save main checkpoint (without optimizer)
    let main_path = path.with_extension("bin");
    let checkpoint_data = CheckpointData {
        model: &checkpoint.model,
        trainer_state: &checkpoint.trainer_state,
    };
    io::serialize_to_file(&main_path, &checkpoint_data)?;

    // Save optimizer state separately
    if let Some(optimizer) = checkpoint.optimizer.as_serializable() {
        let optimizer_path = path.with_extension("optim.bin");
        io::serialize_to_file(&optimizer_path, &optimizer)?;
    }

    Ok(())
}

/// Loads a training checkpoint
pub fn load_checkpoint(
    path: impl AsRef<Path>,
    optimizer_factory: impl Fn() -> Box<dyn Optimizer>,
) -> Result<Checkpoint, CheckpointError> {
    let path = path.as_ref();

    // Load main checkpoint
    let main_path = path.with_extension("bin");
    let checkpoint_data: CheckpointData = io::deserialize_from_file(&main_path)?;

    // Load optimizer state
    let optimizer_path = path.with_extension("optim.bin");
    let optimizer = if optimizer_path.exists() {
        let state = io::deserialize_from_file(&optimizer_path)
            .map_err(|e| CheckpointError::OptimizerState(e.to_string()))?;
        optimizer_factory().load_state(state)
    } else {
        optimizer_factory()
    };

    Ok(Checkpoint {
        model: checkpoint_data.model,
        optimizer,
        trainer_state: checkpoint_data.trainer_state,
    })
}

/// Helper struct for serialization (without optimizer)
#[derive(Serialize, Deserialize)]
struct CheckpointData<'a> {
    model: &'a LanguageModel,
    trainer_state: &'a TrainerState,
}

/// Trait extension for serializable optimizers
pub trait OptimizerSerde: Optimizer {
    fn as_serializable(&self) -> Option<Box<dyn erased_serde::Serialize>>;
    fn load_state(&mut self, state: Box<dyn erased_serde::Serialize>) -> Result<(), String>;
}