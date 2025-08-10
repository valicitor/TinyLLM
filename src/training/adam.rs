use serde::{Serialize, Deserialize};

use crate::{training::{optimizer::{AttentionOptimizerState, OptimizerError}, Optimizer}, LanguageModel};

/// Adam optimizer implementation
#[derive(Debug, Serialize, Deserialize)]
pub struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: usize,
    // Add attention state
    pub attention_state: AttentionOptimizerState,
}

impl AdamOptimizer {
    pub fn new(
        learning_rate: f32,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 0,
            attention_state: AttentionOptimizerState::new(
                embed_dim,
                num_heads,
                head_dim,
            ),
        }
    }
}

#[typetag::serde]
impl Optimizer for AdamOptimizer {
    fn step(&mut self, model: &mut LanguageModel) -> Result<(), OptimizerError> {
        self.step += 1;
        let t = self.step as f32;
        
        // Update embeddings
        self.update_parameter(
            &mut model.embed.value,
            &mut model.embed.grad,
            &mut self.embed.m,
            &mut self.embed.v,
            self.lr, self.beta1, self.beta2, self.epsilon, t
        );
        
        // Update linear layer
        self.update_parameter(
            &mut model.linear.value,
            &mut model.linear.grad,
            &mut self.linear.m,
            &mut self.linear.v,
            self.learning_rate, self.beta1, self.beta2, self.epsilon, t
        );

        Self::update_parameter(
            &mut model.linear.value,
            &mut model.linear.grad,
            &mut self.linear.m,
            &mut self.linear.v,
            self.learning_rate, self.beta1, self.beta2, self.epsilon, t
        );
        
        // Update attention parameters
        Self::update_attention_parameters(
            &mut self,
            &mut model.attention,
            self.learning_rate, 
            t
        );

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
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
        &mut self,
        attention: &mut MultiHeadAttention,
        lr: f32,
        t: f32,
    ) {
        // Update query matrix
        Self::update_parameter(
            &mut attention.w_q.value,
            &mut attention.w_q.grad,
            &mut self.attention_state.wq_m,
            &mut self.attention_state.wq_v,
            lr,
            self.beta1,
            self.beta2,
            self.epsilon,
            t,
        );
        
        // Update key matrix
        Self::update_parameter(
            &mut attention.w_k.value,
            &mut attention.w_k.grad,
            &mut self.attention_state.wk_m,
            &mut self.attention_state.wk_v,
            lr,
            self.beta1,
            self.beta2,
            self.epsilon,
            t,
        );
        
        // Update value matrix
        Self::update_parameter(
            &mut attention.w_v.value,
            &mut attention.w_v.grad,
            &mut self.attention_state.wv_m,
            &mut self.attention_state.wv_v,
            lr,
            self.beta1,
            self.beta2,
            self.epsilon,
            t,
        );
        
        // Update output matrix
        Self::update_parameter(
            &mut attention.w_o.value,
            &mut attention.w_o.grad,
            &mut self.attention_state.wo_m,
            &mut self.attention_state.wo_v,
            lr,
            self.beta1,
            self.beta2,
            self.epsilon,
            t,
        );
    }
}