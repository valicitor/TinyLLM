use std::fs::File;
use std::io::{BufWriter, BufReader};
use serde::{Serialize, Deserialize};

use crate::error::TinyLLMError;
use crate::model::ModelState;
use crate::train::OptimizerState; // Assuming you use serde for serialization

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub model_state: ModelState,       // Your model parameters/state
    pub optimizer_state: OptimizerState, // Your optimizer parameters/state
    pub epoch: usize,
    pub loss: f32,
}

// Save checkpoint
pub fn save_checkpoint(path: &str, checkpoint: &Checkpoint) -> Result<(), TinyLLMError>{
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, checkpoint)?;
    Ok(())
}

// Load checkpoint
pub fn load_checkpoint(path: &str) -> Result<Checkpoint, TinyLLMError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let checkpoint = bincode::deserialize_from(reader)?;
    Ok(checkpoint)
}