use ndarray::{Array2, Axis, s};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use crate::attention::{Attention, WeightedValue};
use crate::utils::sample_from_logits;
use crate::tokenizer::Tokenizer;

/// Cache for storing intermediate values during forward pass
#[derive(Debug)]
pub struct ForwardCache {
    pub input_indices: Vec<usize>,
    pub embeddings: Array2<f32>,      // embeddings + pos encoding
    pub q: Array2<f32>,
    pub k: Array2<f32>,
    pub v: Array2<f32>,
    pub attn_weights: Array2<f32>,    // after softmax
    pub attention_output: Array2<f32>,
    pub logits: Array2<f32>,
}

/// Complete model state including parameters and configuration
#[derive(Serialize, Deserialize)]
pub struct ModelState {
    // Embedding layer (vocab_size x embed_dim)
    pub embed: WeightedValue<Array2<f32>>,
    
    // Final projection layer (embed_dim x vocab_size)
    pub linear: WeightedValue<Array2<f32>>,
    
    // Attention mechanism
    pub attention: Attention,
    
    // Model configuration
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub block_size: usize,
    
    // Positional encoding (block_size x embed_dim)
    pub positional_encoding: Array2<f32>,
}

impl ModelState {
    /// Get a complete copy of the model state (parameters + config)
    pub fn get_state(&self) -> Self {
        ModelState {
            embed: WeightedValue {
                value: self.embed.value.clone(),
                grad: Array2::zeros(self.embed.value.dim()), // Don't clone gradients
            },
            linear: WeightedValue {
                value: self.linear.value.clone(),
                grad: Array2::zeros(self.linear.value.dim()),
            },
            attention: Attention {
                wq: WeightedValue {
                    value: self.attention.wq.value.clone(),
                    grad: Array2::zeros(self.attention.wq.value.dim()),
                },
                wk: WeightedValue {
                    value: self.attention.wk.value.clone(),
                    grad: Array2::zeros(self.attention.wk.value.dim()),
                },
                wv: WeightedValue {
                    value: self.attention.wv.value.clone(),
                    grad: Array2::zeros(self.attention.wv.value.dim()),
                },
                wo: WeightedValue {
                    value: self.attention.wo.value.clone(),
                    grad: Array2::zeros(self.attention.wo.value.dim()),
                },
                embed_dim: self.attention.embed_dim,
            },
            vocab_size: self.vocab_size,
            embed_dim: self.embed_dim,
            block_size: self.block_size,
            positional_encoding: self.positional_encoding.clone(),
        }
    }

    /// Restore model state from saved state
    pub fn set_state(&mut self, state: ModelState) {
        self.embed.value = state.embed.value;
        self.linear.value = state.linear.value;
        
        self.attention.wq.value = state.attention.wq.value;
        self.attention.wk.value = state.attention.wk.value;
        self.attention.wv.value = state.attention.wv.value;
        self.attention.wo.value = state.attention.wo.value;
        
        self.vocab_size = state.vocab_size;
        self.embed_dim = state.embed_dim;
        self.block_size = state.block_size;
        self.positional_encoding = state.positional_encoding;
        
        // Reset gradients
        self.zero_grad();
    }

    /// Create a new model with initialized weights
    pub fn new(block_size: usize, vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            embed: WeightedValue::new(vocab_size, embed_dim),
            linear: WeightedValue::new(embed_dim, vocab_size),
            attention: Attention::new(embed_dim),
            vocab_size,
            embed_dim,
            block_size,
            positional_encoding: Self::positional_encoding(block_size, embed_dim),
        }
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &self)?;
        Ok(())
    }

    /// Load model from file
    pub fn load(path: &str) -> Option<Self> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);

        let mut model: ModelState = bincode::deserialize_from(reader).ok()?;

        // Re-init gradients skipped during serialization
        model.zero_grad();
        Some(model)
    }

    /// Forward pass through the model
    pub fn forward(&self, inputs: &[usize]) -> ForwardCache {
        let seq_len = inputs.len();
        assert!(seq_len <= self.block_size, "Input sequence too long");

        // 1. Embed tokens
        let embeddings = self.embed_tokens(inputs);
        
        // 2. Add positional encoding
        let embeddings_with_pos = &embeddings + &self.positional_encoding.slice(s![..seq_len, ..]);
        
        // 3. Attention mechanism
        let (q, k, v) = self.attention.forward(&embeddings_with_pos);
        let (attn_weights, attention_output) = self.attention.scaled_dot_product_attention(&q, &k, &v);
        
        // 4. Final output projection
        let projected = attention_output.dot(&self.attention.wo.value);
        
        // 5. Compute logits (seq_len x vocab_size)
        let logits = projected.dot(&self.linear.value);

        ForwardCache {
            input_indices: inputs.to_vec(),
            embeddings: embeddings_with_pos,
            q,
            k,
            v,
            attn_weights,
            attention_output: projected,
            logits,
        }
    }

    /// Backward pass through the model
    pub fn backward(&mut self, cache: &ForwardCache, targets: &[usize]) {
        // 1. Compute gradient of cross-entropy loss w.r.t. logits
        let dlogits = self.d_cross_entropy(&cache.logits, targets);
        
        // 2. Linear layer gradient
        let linear_grad_update = cache.attention_output.t().dot(&dlogits);
        self.linear.grad += &linear_grad_update;
        
        let d_attention_output = dlogits.dot(&self.linear.value.t());
        
        // 3. Backprop through attention mechanism
        let d_embeddings = self.attention.backward(
            &cache.embeddings,
            &cache.q,
            &cache.k,
            &cache.v,
            &cache.attn_weights,
            &d_attention_output,
        );
        
        // 4. Backprop to embedding layer
        self.backprop_to_embeddings(&cache.input_indices, &d_embeddings);
    }

    /// Zero out all gradients
    pub fn zero_grad(&mut self) {
        self.embed.zero_grad();
        self.linear.zero_grad();
        self.attention.zero_grads();
    }

    pub fn gradient_norm(&self) -> f32 {
        let mut total = 0.0;
        
        total += self.embed.grad.mapv(|x| x * x).sum();
        total += self.linear.grad.mapv(|x| x * x).sum();
        total += self.attention.wq.grad.mapv(|x| x * x).sum();
        total += self.attention.wk.grad.mapv(|x| x * x).sum();
        total += self.attention.wv.grad.mapv(|x| x * x).sum();
        total += self.attention.wo.grad.mapv(|x| x * x).sum();
        
        total.sqrt()
    }

    /// Generate text from prompt
    pub fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        length: usize,
        temperature: f32,
        top_k: Option<usize>
    ) -> String {
        let mut tokens = tokenizer.encode(prompt);

        for _ in 0..length {
            let context = if tokens.len() > self.block_size {
                &tokens[tokens.len() - self.block_size..]
            } else {
                &tokens[..]
            };

            let cache = self.forward(context);
            let last_logits = cache.logits.slice(s![-1, ..]).to_owned().insert_axis(Axis(0));
            let next_token = sample_from_logits(&last_logits, temperature, top_k);
            tokens.push(next_token);
        }

        tokenizer.decode(&tokens)
    }

    // --- Helper methods ---

    fn embed_tokens(&self, inputs: &[usize]) -> Array2<f32> {
        let rows: Vec<_> = inputs.iter()
            .map(|&i| self.embed.value.row(i))
            .collect();
        ndarray::stack(Axis(0), &rows).unwrap()
    }

    fn d_cross_entropy(&self, logits: &Array2<f32>, targets: &[usize]) -> Array2<f32> {
        let batch_size = logits.nrows();
        let vocab_size = logits.ncols();
        let mut dlogits = logits.clone();

        for i in 0..batch_size {
            let max = dlogits.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            
            for j in 0..vocab_size {
                dlogits[[i, j]] = (dlogits[[i, j]] - max).exp();
                exp_sum += dlogits[[i, j]];
            }
            
            for j in 0..vocab_size {
                dlogits[[i, j]] /= exp_sum;
            }
            
            dlogits[[i, targets[i]]] -= 1.0;
        }
        
        dlogits
    }

    fn backprop_to_embeddings(&mut self, input_indices: &[usize], d_embeddings: &Array2<f32>) {
        for (i, &token_idx) in input_indices.iter().enumerate() {
            for j in 0..self.embed_dim {
                self.embed.grad[[token_idx, j]] += d_embeddings[[i, j]];
            }
        }
    }

    fn positional_encoding(block_size: usize, embed_dim: usize) -> Array2<f32> {
        let mut pe = Array2::<f32>::zeros((block_size, embed_dim));

        for pos in 0..block_size {
            for i in 0..(embed_dim / 2) {
                let div_term = (10000f32).powf((2 * i) as f32 / embed_dim as f32);
                pe[[pos, 2 * i]] = (pos as f32 / div_term).sin();
                pe[[pos, 2 * i + 1]] = (pos as f32 / div_term).cos();
            }
        }

        pe
    }
}