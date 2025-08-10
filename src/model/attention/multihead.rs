use ndarray::{
    Array, Array2, Array3, Array4, ArrayBase, Axis, Dim, OwnedRepr, linalg::general_mat_mul, s,
};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::{
    model::{Attention, ModelError, attention::AttentionConfig},
    utils::{ParameterWithGrad, apply_causal_mask, apply_dropout_4d, softmax_4d},
};

/// Multi-head attention layer
#[derive(Debug, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub embed_dim: usize,
    pub head_dim: usize,
    pub q: ParameterWithGrad<Array2<f32>, Array2<f32>>, // [embed_dim, embed_dim]
    pub k: ParameterWithGrad<Array2<f32>, Array2<f32>>,
    pub v: ParameterWithGrad<Array2<f32>, Array2<f32>>,
    pub o: ParameterWithGrad<Array2<f32>, Array2<f32>>, // output projection
    pub dropout_rate: f32,
}

#[typetag::serde]
impl Attention for MultiHeadAttention {
    fn forward(&self, x: &Array3<f32>, training: bool) -> Result<Array3<f32>, ModelError> {
        self.forward(x, training)
    }
}

impl MultiHeadAttention {
    pub fn new(config: AttentionConfig) -> Self {
        let head_dim = config.embed_dim / config.num_heads;
        assert!(
            config.embed_dim % config.num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );

        // Scale initialization differently for Q/K/V
        let q_std_dev = (2.0 / (config.embed_dim as f32)).sqrt();
        let kv_std_dev = (1.0 / (config.embed_dim as f32)).sqrt();
        let o_std_dev = (1.0 / ((config.num_heads * head_dim) as f32)).sqrt();

        // Initialize query, key, value weights
        let qkv_shape = (config.embed_dim, config.num_heads * head_dim);
        let q = Array::random(qkv_shape, Normal::new(0.0, q_std_dev).unwrap());
        let k = Array::random(qkv_shape, Normal::new(0.0, kv_std_dev).unwrap());
        let v = Array::random(qkv_shape, Normal::new(0.0, kv_std_dev).unwrap());

        // Initialize output weights
        let output_shape = (config.num_heads * head_dim, config.embed_dim);
        let o = Array::random(output_shape, Normal::new(0.0, o_std_dev).unwrap());

        Self {
            num_heads: config.num_heads,
            embed_dim: config.embed_dim,
            head_dim,
            q: ParameterWithGrad::new(q),
            k: ParameterWithGrad::new(k),
            v: ParameterWithGrad::new(v),
            o: ParameterWithGrad::new(o),
            dropout_rate: config.dropout_rate,
        }
    }

    pub fn project_input(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>, // Accept reference to owned array
        weights: &Array2<f32>,
    ) -> Array3<f32> {
        let (batch_size, seq_len, embed_dim) = x.dim();
        let (_, proj_dim) = weights.dim();

        // Efficient batched matrix multiplication
        let x_2d = x
            .view()
            .into_shape((batch_size * seq_len, embed_dim))
            .unwrap();
        let mut result = Array2::zeros((batch_size * seq_len, proj_dim));
        general_mat_mul(1.0, &x_2d, weights, 0.0, &mut result);

        result.into_shape((batch_size, seq_len, proj_dim)).unwrap()
    }

    fn project_input_heads(&self, x: &Array3<f32>, weights: &Array2<f32>) -> Array4<f32> {
        let projected = self.project_input(x, weights);
        self.split_heads(projected) //[batch_size, seq_len, proj_dim]
    }

    pub fn forward(
        &self,
        x: &Array3<f32>, // [batch_size, seq_len, embed_dim]
        training: bool,
    ) -> Result<Array3<f32>, ModelError> {
        // 1. Project inputs to Q, K, V
        let q_heads = self.project_input_heads(x, &self.q.weight);
        let k_heads = self.project_input_heads(x, &self.k.weight);
        let v_heads = self.project_input_heads(x, &self.v.weight);

        // 3. Compute scaled dot-product attention
        let (_, output) = self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, training);

        // 4. Merge heads and project output
        let merged = self.merge_heads(output);
        let output = self.project_input(&merged, &self.o.weight);

        Ok(output)
    }

    pub fn split_heads(&self, x: Array3<f32>) -> Array4<f32> {
        let (b, s, d) = x.dim();
        let head_dim = d / self.num_heads;
        x.into_shape((b, s, self.num_heads, head_dim))
            .unwrap()
            //.permuted_axes([0, 2, 1, 3])
    }

    pub fn merge_heads(&self, x: Array4<f32>) -> Array3<f32> {
        let (b, h, s, d) = x.dim();
        x.into_shape((b, s, h * d))
            .unwrap()
            //.permuted_axes([0, 2, 1, 3])
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Array4<f32>, // [batch_size, seq_len, num_heads, head_dim ]
        k: &Array4<f32>,
        v: &Array4<f32>,
        training: bool,
    ) -> (Array4<f32>, Array4<f32>) {
        let (batch_size, seq_len, num_heads, head_dim) = q.dim();
        let scale = (head_dim as f32).sqrt();

        // 1. Compute attention scores (QK^T / sqrt(d_k))
        let mut scores = Array4::<f32>::zeros((batch_size, num_heads, seq_len, seq_len));

        // Batch matrix multiplication with numerical stability checks
        for b in 0..batch_size {
            let q_batch = q.slice(s![b, .., .., ..]);
            let k_batch = k.slice(s![b, .., .., ..]);
            let mut scores_batch = scores.slice_mut(s![b, .., .., ..]);

            for h in 0..num_heads {
                let q_head = q_batch.slice(s![h, .., ..]);
                let k_head = k_batch.slice(s![h, .., ..]);
                let mut s_head = scores_batch.slice_mut(s![h, .., ..]);

                // QK^T operation with scaling
                let mut dot = q_head.dot(&k_head.t());
                dot.mapv_inplace(|x| {
                    let scaled = x / scale;
                    // Clip extremely large values to prevent overflow
                    if scaled > 100.0 {
                        100.0
                    } else if scaled < -100.0 {
                        -100.0
                    } else {
                        scaled
                    }
                });

                // Ensure no infinities or NaNs
                assert!(
                    dot.iter().all(|v| v.is_finite()),
                    "Attention scores contain non-finite values before masking"
                );

                s_head.assign(&dot);
            }
        }

        // 2. Apply causal mask
        apply_causal_mask(&mut scores);

        // 3. Softmax with numerical stability
        let mut attn_weights = scores;
        softmax_4d(&mut attn_weights);

        // 4. Apply dropout if training
        if training && self.dropout_rate > 0.0 {
            apply_dropout_4d(&mut attn_weights, self.dropout_rate, training, &mut None);
        }

        // 5. Compute weighted sum (attention output)
        let mut output = Array4::<f32>::zeros((batch_size, num_heads, seq_len, head_dim));
        for b in 0..batch_size {
            let weights_batch = attn_weights.slice(s![b, .., .., ..]);
            let v_batch = v.slice(s![b, .., .., ..]);
            let mut out_batch = output.slice_mut(s![b, .., .., ..]);

            for h in 0..num_heads {
                let weights = weights_batch.slice(s![h, .., ..]);
                let values = v_batch.slice(s![h, .., ..]);
                out_batch
                    .slice_mut(s![h, .., ..])
                    .assign(&weights.dot(&values));
            }
        }

        (attn_weights, output)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use approx::assert_abs_diff_eq;
//     use ndarray::{Array3, Array4, arr3}; // For float comparisons (add approx crate in dev-dependencies)

//     // Helper to create a MultiHeadAttention instance with minimal config
//     fn create_mha(embed_dim: usize, num_heads: usize) -> MultiHeadAttention {
//         MultiHeadAttention::new(AttentionConfig {
//             embed_dim,
//             num_heads,
//             dropout_rate: 0.0,
//         })
//     }

//     // --- Tests for `split_heads` ---
//     #[test]
//     fn test_split_heads_shape() {
//         let mha = create_mha(8, 2); // embed_dim=8, num_heads=2
//         let input = Array3::<f32>::zeros((2, 3, 8)); // batch=2, seq_len=3, embed_dim=8 
//         let split = mha.split_heads(input);
//         println!("[DEBUG] Split heads shape: {:?}", split.shape());
//         assert_eq!(split.shape(), &[2, 3, 2, 4]); // [batch_size, num_heads, seq_len, head_dim]
//     }

//     #[test]
//     fn test_split_heads_data_integrity() {
//         let mha = create_mha(6, 3); // embed_dim=6, num_heads=3 (head_dim=2)
//         let input = arr3(&[
//             [[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]], // batch=1, seq_len=2
//         ]);
//         let split = mha.split_heads(input.clone());

//         println!("[DEBUG] Full split tensor:\n{:?}", split);

//         // Check manual splits for first sequence element
//         println!(
//             "[DEBUG] Split heads data: {} Expected: {}",
//             split[[0, 0, 0, 0]],
//             1.0
//         );
//         assert_abs_diff_eq!(split[[0, 0, 0, 0]], 1.0); // Head 0, dim 0
//         println!(
//             "[DEBUG] Split heads data: {} Expected: {}",
//             split[[0, 0, 0, 1]],
//             2.0
//         );
//         assert_abs_diff_eq!(split[[0, 0, 0, 1]], 2.0); // Head 0, dim 1
//         println!(
//             "[DEBUG] Split heads data: {} Expected: {}",
//             split[[0, 0, 1, 0]],
//             3.0
//         );
//         assert_abs_diff_eq!(split[[0, 0, 1, 0]], 3.0); // Head 1, dim 0
//         println!(
//             "[DEBUG] Split heads data: {} Expected: {}",
//             split[[0, 0, 1, 1]],
//             4.0
//         );
//         assert_abs_diff_eq!(split[[0, 0, 1, 1]], 4.0); // Head 1, dim 1
//         println!(
//             "[DEBUG] Split heads data: {} Expected: {}",
//             split[[0, 0, 2, 0]],
//             5.0
//         );
//         assert_abs_diff_eq!(split[[0, 0, 2, 0]], 5.0); // Head 2, dim 0
//         println!(
//             "[DEBUG] Split heads data: {} Expected: {}",
//             split[[0, 0, 2, 1]],
//             6.0
//         );
//         assert_abs_diff_eq!(split[[0, 0, 2, 1]], 6.0); // Head 2, dim 1
//     }

//     #[test]
//     fn test_split_heads_single_head() {
//         let mha = create_mha(4, 1); // embed_dim=4, num_heads=1 (head_dim=4)
//         let input = Array3::<f32>::from_elem((1, 2, 4), 1.0); // batch=1, seq_len=2
//         let split = mha.split_heads(input);
//         assert_eq!(split.shape(), &[1, 2, 1, 4]); // Single head retains full dim
//     }

//     // --- Tests for `merge_heads` ---
//     #[test]
//     fn test_merge_heads_shape() {
//         let mha = create_mha(8, 2); // embed_dim=8, num_heads=2
//         let input = Array4::<f32>::zeros((2, 2, 3, 4)); // [batch, seq_len, num_heads, head_dim]
//         let merged = mha.merge_heads(input);
//         println!("[DEBUG] Merged heads shape: {:?}", merged.shape());
//         assert_eq!(merged.shape(), &[2, 3, 8]); // [batch, seq_len, embed_dim]
//     }

//     #[test]
//     fn test_merge_heads_data_integrity() {
//         let mha = create_mha(4, 2); // embed_dim=6, num_heads=3
//         let input =
//             Array4::from_shape_vec((1, 2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
//         let merged = mha.merge_heads(input);

//         // Check merged output matches original layout
//         println!(
//             "[DEBUG] Merge heads data: {} Expected: {}",
//             merged[[0, 0, 0]],
//             1.0
//         );
//         assert_abs_diff_eq!(merged[[0, 0, 0]], 1.0); // Head 0, dim 0
//         println!(
//             "[DEBUG] Merge heads data: {} Expected: {}",
//             merged[[0, 0, 1]],
//             2.0
//         );
//         assert_abs_diff_eq!(merged[[0, 0, 1]], 2.0); // Head 0, dim 1
//         println!(
//             "[DEBUG] Merge heads data: {} Expected: {}",
//             merged[[0, 0, 2]],
//             3.0
//         );
//         assert_abs_diff_eq!(merged[[0, 0, 2]], 3.0); // Head 1, dim 0
//         println!(
//             "[DEBUG] Merge heads data: {} Expected: {}",
//             merged[[0, 0, 3]],
//             4.0
//         );
//         assert_abs_diff_eq!(merged[[0, 0, 3]], 4.0); // Head 1, dim 1
//         println!(
//             "[DEBUG] Merge heads data: {} Expected: {}",
//             merged[[0, 0, 4]],
//             5.0
//         );
//         assert_abs_diff_eq!(merged[[0, 0, 4]], 5.0); // Head 2, dim 0
//         println!(
//             "[DEBUG] Merge heads data: {} Expected: {}",
//             merged[[0, 0, 5]],
//             6.0
//         );
//         assert_abs_diff_eq!(merged[[0, 0, 5]], 6.0); // Head 2, dim 1
//     }

//     #[test]
//     fn test_merge_heads_single_sequence() {
//         let mha = create_mha(4, 2); // embed_dim=4, num_heads=2
//         let input = Array4::<f32>::from_elem((1, 1, 2, 2), 1.0); // batch=1, seq_len=1
//         let merged = mha.merge_heads(input);
//         assert_eq!(merged.shape(), &[1, 1, 4]); // Merged to single sequence element
//     }

//     // --- Roundtrip Test (Split → Merge) ---
//     #[test]
//     fn test_split_merge_roundtrip() {
//         let mha = create_mha(12, 3); // embed_dim=12, num_heads=3
//         let input = Array3::<f32>::random((2, 5, 12), Normal::new(0.0, 1.0).unwrap());
//         let split = mha.split_heads(input.clone());
//         let merged = mha.merge_heads(split);
//         //assert_abs_diff_eq!(merged, input, epsilon = 1e-6); // Should be identical
//         for (o, e) in merged.iter().zip(input.iter()) {
//             assert!((o - e).abs() <= 1e-6, "output: {}, expected: {}", o, e);
//         }
//     }

//     #[test]
//     fn test_initialize_attention_weights_shapes() {
//         let embed_dim = 8;
//         let num_heads = 2;
//         let head_dim = embed_dim / num_heads;

//         let (q, k, v, o) = MultiHeadAttention::init(embed_dim, num_heads, head_dim);

//         assert_eq!(q.dim(), (embed_dim, num_heads * head_dim));
//         assert_eq!(k.dim(), (embed_dim, num_heads * head_dim));
//         assert_eq!(v.dim(), (embed_dim, num_heads * head_dim));
//         assert_eq!(o.dim(), (num_heads * head_dim, embed_dim));
//     }

//     #[test]
//     fn test_project_input_basic() {
//         let embed_dim = 4;
//         let mha = create_mha(embed_dim, 2);

//         // Create a simple input: batch_size=1, seq_len=2, embed_dim=4
//         let x = arr3(&[[[1., 0., 0., 0.], [0., 1., 0., 0.]]]);

//         // Project input with q weights
//         let projected = mha.project_input(&x, &mha.q.weight);

//         // Check shape: should be (1, 2, embed_dim)
//         assert_eq!(projected.shape(), &[1, 2, embed_dim]);

//         // Since weights are random, just test no NaNs and finite values
//         assert!(projected.iter().all(|v| v.is_finite()));
//     }

//     #[test]
//     fn test_split_and_merge_heads_consistency() {
//         let embed_dim = 4;
//         let num_heads = 2;
//         let mha = create_mha(embed_dim, num_heads);

//         // Create input with shape (batch_size=1, seq_len=3, embed_dim=8)
//         let x = arr3(&[[
//             [1., 2., 3., 4., 5., 6., 7., 8.],
//             [9., 10., 11., 12., 13., 14., 15., 16.],
//             [17., 18., 19., 20., 21., 22., 23., 24.],
//         ]]);

//         let projected = mha.project_input(&x, &mha.q.weight);
//         let split = mha.split_heads(projected.clone());
//         assert_eq!(split.dim(), (1, 3, num_heads, embed_dim / num_heads));
//         let merged = mha.merge_heads(split);
//         assert_eq!(merged.dim(), (1, 3, embed_dim));

//         for (o, e) in merged.iter().zip(projected.iter()) {
//             assert!((o - e).abs() <= 1e-6, "output: {}, expected: {}", o, e);
//         }
//     }

//     #[test]
//     fn test_scaled_dot_product_attention_shapes_and_values() {
//         let embed_dim = 4;
//         let num_heads = 2;
//         let head_dim = embed_dim / num_heads;
//         let mha = create_mha(embed_dim, num_heads);

//         // Create Q, K, V of shape (batch_size=1, num_heads=2, seq_len=2, head_dim=2)
//         let q = Array4::<f32>::ones((1, num_heads, 2, head_dim));
//         let k = Array4::<f32>::ones((1, num_heads, 2, head_dim));
//         let v = Array4::<f32>::from_shape_fn((1, num_heads, 2, head_dim), |(_, _, i, j)| {
//             (i + j) as f32
//         });

//         let (attn_weights, output) = mha.scaled_dot_product_attention(&q, &k, &v, false);

//         // Shapes
//         assert_eq!(attn_weights.dim(), (1, num_heads, 2, 2));
//         assert_eq!(output.dim(), (1, num_heads, 2, head_dim));

//         // Attention weights rows should sum to ~1 (softmax)
//         for b in 0..1 {
//             for h in 0..num_heads {
//                 for i in 0..2 {
//                     let row = attn_weights.slice(s![b, h, i, ..]);
//                     let sum: f32 = row.sum();
//                     assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
//                 }
//             }
//         }
//     }

//     #[test]
//     fn test_forward_output_shape_and_finiteness() {
//         let embed_dim = 8;
//         let num_heads = 2;
//         let mha = create_mha(embed_dim, num_heads);

//         let batch_size = 2;
//         let seq_len = 3;
//         // Input random values
//         let x = Array3::<f32>::random(
//             (batch_size, seq_len, embed_dim),
//             Normal::new(0.0, 1.0).unwrap(),
//         );

//         let output = mha.forward(&x, false).expect("Forward pass failed");

//         // Output shape should be (batch_size, seq_len, embed_dim)
//         assert_eq!(output.shape(), &[batch_size, seq_len, embed_dim]);

//         // Values should be finite
//         assert!(output.iter().all(|v| v.is_finite()));
//     }

//     #[test]
//     fn test_merge_heads_roundtrip_with_synthetic_data() {
//         let batch_size = 2;
//         let seq_len = 4;
//         let num_heads = 2;
//         let head_dim = 3;
//         let embed_dim = num_heads * head_dim;

//         let mut counter = 0.0;
//         let input = Array::from_shape_fn((batch_size, num_heads, seq_len, head_dim), |_| {
//             counter += 1.0;
//             counter
//         });

//         let mha = create_mha(embed_dim, num_heads);

//         let merged = mha.merge_heads(input.clone());
//         let restored = mha.split_heads(merged);

//         for (o, e) in restored.iter().zip(input.iter()) {
//             assert!((o - e).abs() <= 1e-6, "output: {}, expected: {}", o, e);
//         }
//     }
// }
