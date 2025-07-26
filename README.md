# TinyLLM

**TinyLLM** is a minimal character-level Transformer language model written in Rust.  
This project is an experiment in building a full-stack LLM pipeline from scratch, including tokenization, training, checkpointing, and inference.

---

## Features

- Transformer architecture implemented in pure Rust
- Adam optimizer and backpropagation
- Character-level tokenization
- Interactive CLI interface
- Checkpointing and model loading/saving via `bincode`
- Built for experimentation and learning

---

## Usage

TinyLLM has a simple command-line interface with three main commands:

### Train the model

Train the model from scratch on a character-level dataset (default: Tiny Shakespeare).

```bash
cargo run --release -- train [EPOCHS]
```

- `EPOCHS`: Optional number of training epochs (default: `100`)

Example:

```bash
cargo run --release -- train 50
```

This will:
- Load `data/tiny_shakespeare.txt` as training data
- Tokenize it at the character level
- Initialize a transformer model (or load a previous checkpoint from `model.bin`)
- Save checkpoints every 10 epochs

---

### Generate text from a prompt

Generate text from the latest saved model:

```bash
cargo run --release -- generate "Once upon a time"
```

- Generates 200 characters of text using the given prompt
- Uses top-k sampling and temperature settings defined in `ModelConfig`

---

### Interactive mode

Chat with the model directly in your terminal:

```bash
cargo run --release -- interactive
```

Example:

```text
Enter prompt:
To be, or not to be

Generated:
To be, or not to be:
And so the queen did stay
Upon her words of fate and fire...
```

Type `quit` to exit.

---

## Dataset

By default, the model expects a text file at:

```
data/tiny_shakespeare.txt
```

You can replace this file with your own text corpus (e.g. poetry, code, legal text, etc).  
Just make sure it's a **plain UTF-8 `.txt` file**, and it will be tokenized at the character level.

---

## Configuration

Model hyperparameters can be configured via the `ModelConfig` struct in `main.rs`:

```rust
pub struct ModelConfig {
    pub block_size: usize,
    pub embed_dim: usize,
    pub learning_rate: f32,
    pub dropout_rate: f32,
    pub temperature: f32,
    pub top_k: Option<usize>,
}
```

These control the context size, embedding dimension, sampling behavior, and more.

---

## 📦 Dependencies

- [`ndarray`](https://crates.io/crates/ndarray)
- [`rand`](https://crates.io/crates/rand)
- [`bincode`](https://crates.io/crates/bincode)
- [`serde`](https://crates.io/crates/serde)

---

## 📜 License

MIT License. See [LICENSE](./LICENSE) for details.

---

## 🛠️ TODO

- Add multi-head attention
- Add dropout and masking
- Implement LayerNorm
- Save training metrics to file
- Improve sampling diversity
- Support inference via WebAssembly?

---

## 🙏 Credits

This project was inspired by [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), reimagined in safe, performant Rust.

---
