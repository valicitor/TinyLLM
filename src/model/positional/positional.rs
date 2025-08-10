use ndarray::{Array2, Array3};

/// Trait for all positional encoding implementations
#[typetag::serde]
pub trait PositionalEncoding {
    /// Apply positional encoding to input embeddings
    /// Input shape: (batch_size, seq_len, embedding_dim)
    fn forward(&self, input: &Array3<f32>) -> Array3<f32>;
    
    /// Update during backpropagation (for learned embeddings)
    /// positions shape: (batch_size, seq_len)
    fn backward(&mut self, positions: &Array2<usize>, grad_output: &Array3<f32>);
}