use crate::{model::MultiHeadAttention, LanguageModel};
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("Optimization failed: {0}")]
    Optimization(String),
}

/// State for attention layer optimization
#[derive(Debug, Serialize, Deserialize)]
pub struct AttentionOptimizerState {
    // Query matrix state
    pub wq_m: Array2<f32>,  // First moment estimate
    pub wq_v: Array2<f32>,  // Second moment estimate
    
    // Key matrix state
    pub wk_m: Array2<f32>,
    pub wk_v: Array2<f32>,
    
    // Value matrix state
    pub wv_m: Array2<f32>,
    pub wv_v: Array2<f32>,
    
    // Output matrix state
    pub wo_m: Array2<f32>,
    pub wo_v: Array2<f32>,
}

impl AttentionOptimizerState {
    /// Creates new optimizer state initialized with zeros
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let qkv_shape = (embed_dim, num_heads * head_dim);
        let output_shape = (num_heads * head_dim, embed_dim);
        
        Self {
            wq_m: Array2::zeros(qkv_shape),
            wq_v: Array2::zeros(qkv_shape),
            wk_m: Array2::zeros(qkv_shape),
            wk_v: Array2::zeros(qkv_shape),
            wv_m: Array2::zeros(qkv_shape),
            wv_v: Array2::zeros(qkv_shape),
            wo_m: Array2::zeros(output_shape),
            wo_v: Array2::zeros(output_shape),
        }
    }
    
    /// Resets all moments to zero
    pub fn reset(&mut self) {
        self.wq_m.fill(0.0);
        self.wq_v.fill(0.0);
        self.wk_m.fill(0.0);
        self.wk_v.fill(0.0);
        self.wv_m.fill(0.0);
        self.wv_v.fill(0.0);
        self.wo_m.fill(0.0);
        self.wo_v.fill(0.0);
    }
}

/// Trait for optimization algorithms
#[typetag::serde]
pub trait Optimizer {
    /// Performs a parameter update step
    fn step(&mut self, model: &mut LanguageModel) -> Result<(), OptimizerError>;

    /// Returns the current learning rate
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate
    fn set_learning_rate(&mut self, lr: f32);

    fn update_parameter(
        &mut self,
        param: &mut Array2<f32>,
        grad: &mut Array2<f32>,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: f32,
    );

    fn update_attention_parameters(
        &mut self,
        attention: &mut MultiHeadAttention,
        lr: f32,
        t: f32,
    );
}


