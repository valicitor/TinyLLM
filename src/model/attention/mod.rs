//! Attention mechanisms
mod attention;
pub use attention::{AttentionConfig, Attention};

mod multihead;
pub use multihead::MultiHeadAttention;