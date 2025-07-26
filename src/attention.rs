// attention.rs
use ndarray::{Array2};
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct WeightedValue<T> {
    pub value: T,
    pub grad: T,
}

impl WeightedValue<Array2<f32>> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        Self {
            value: Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng)),
            grad: Array2::zeros((rows, cols)),
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Attention {
    pub wq: WeightedValue<Array2<f32>>,
    pub wk: WeightedValue<Array2<f32>>,
    pub wv: WeightedValue<Array2<f32>>,
    pub wo: WeightedValue<Array2<f32>>,
    pub embed_dim: usize,
}

impl Attention {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            wq: WeightedValue::new(embed_dim, embed_dim),
            wk: WeightedValue::new(embed_dim, embed_dim),
            wv: WeightedValue::new(embed_dim, embed_dim),
            wo: WeightedValue::new(embed_dim, embed_dim),
            embed_dim,
        }
    }

    pub fn zero_grads(&mut self) {
        self.wq.zero_grad();
        self.wk.zero_grad();
        self.wv.zero_grad();
        self.wo.zero_grad();
    }

    pub fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // Project to Q, K, V
        let q = x.dot(&self.wq.value);
        let k = x.dot(&self.wk.value);
        let v = x.dot(&self.wv.value);
        
        (q, k, v)
    }

    pub fn scaled_dot_product_attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let seq_len = q.nrows();
        let dk = self.embed_dim as f32;

        // Scores = Q * K^T / sqrt(dk)
        let mut scores = q.dot(&k.t()) / dk.sqrt();

        // Apply causal mask
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        // Softmax
        let mut attn_weights = scores.clone();
        for mut row in attn_weights.rows_mut() {
            let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = row.iter_mut().map(|x| {
                *x = (*x - max).exp();
                *x
            }).sum();
            row.mapv_inplace(|x| x / exp_sum);
        }

        // Output = weights * V
        let output = attn_weights.dot(v);
        
        (attn_weights, output)
    }

    pub fn backward(
        &mut self,
        x: &Array2<f32>,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        attn_weights: &Array2<f32>,
        grad_output: &Array2<f32>,
    ) -> Array2<f32> {
        // Backprop through output projection
        let d_wo = x.t().dot(grad_output);
        self.wo.grad += &d_wo;
        let d_attention = grad_output.dot(&self.wo.value.t());

        // Backprop through attention
        let d_v = attn_weights.t().dot(&d_attention);
        let d_attn_weights = d_attention.dot(&v.t());

        // Backprop through softmax
        let mut d_scores = Array2::zeros(attn_weights.raw_dim());
        for i in 0..attn_weights.nrows() {
            let attn_row = attn_weights.row(i);
            let d_attn_row = d_attn_weights.row(i);
            let dot = attn_row.dot(&d_attn_row);
            for j in 0..attn_weights.ncols() {
                d_scores[[i, j]] = attn_row[j] * (d_attn_row[j] - dot);
            }
        }

        // Backprop through scaling
        let scale = (self.embed_dim as f32).sqrt();
        let d_q = d_scores.dot(k) / scale;
        let d_k = d_scores.t().dot(q) / scale;

        // Backprop through Q, K, V projections
        let d_wq = x.t().dot(&d_q);
        let d_wk = x.t().dot(&d_k);
        let d_wv = x.t().dot(&d_v);
        
        self.wq.grad += &d_wq;
        self.wk.grad += &d_wk;
        self.wv.grad += &d_wv;

        // Return gradient to be propagated back
        d_q.dot(&self.wq.value.t()) + d_k.dot(&self.wk.value.t()) + d_v.dot(&self.wv.value.t())
    }
}