# TinyLLM
# 🧠 TinyLLM

**TinyLLM** is a minimalist character-level Transformer language model implemented in Rust. It's built as a learning project to understand how large language models (LLMs) work under the hood — from tokenization and training to inference.

> 🚀 Inspired by projects like [nanoGPT](https://github.com/karpathy/nanoGPT), but written in idiomatic Rust with a focus on clarity and control.

---

## 📦 Features

- ✅ Pure Rust implementation — no Python or external ML frameworks
- ✅ Character-level tokenizer
- ✅ Transformer architecture with self-attention
- ✅ Adam optimizer with weight decay and bias correction
- ✅ Checkpointing for training state
- ✅ Simple sampling/generation loop
- ✅ Grad norm + loss tracking
- ✅ Tests for core components

---

## 🛠️ Project Structure
src/
├── model.rs # Transformer blocks, attention, and parameters
├── train.rs # Training loop and Adam optimizer
├── generate.rs # Text sampling from the trained model
├── tokenizer.rs # Character-level tokenizer
├── checkpoint.rs # Save/load model + optimizer state
├── main.rs # CLI entry point (train/generate)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/valicitor/tinyllm.git
cd tinyllm
```
