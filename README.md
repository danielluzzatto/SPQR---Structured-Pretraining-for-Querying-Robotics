
# SPQR - Structured Pretraining for Querying Robotics Implementation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A PyTorch implementation of GPT-2 inspired by Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series. This project includes support for multi-GPU training, flash attention, and a focus on scalability.

---

## Features
- **GPT-2 Architecture**: Follows the "Attention Is All You Need" paper.
- **Flash Attention**: Optimized attention mechanism for faster training.
- **Multi-GPU Support**: Distributed training using PyTorch's `DistributedDataParallel`.
- **Scalable**: Works on single GPU, multi-GPU, and CPU (not recommended for training).

---

## Hyperparameters
- `block_size`: 1024 (context length)
- `vocab_size`: 50304
- `n_layer`: 12 (number of transformer layers)
- `n_head`: 12 (number of attention heads)
- `n_embd`: 768 (embedding dimension)
- `total_batch_size`: 524288
- `B`: 64 (batch size)
- `T`: 1024 (sequence length)

---

## Architecture
```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
    wpe = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
    ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # language model head
```

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)

```bash
pip install torch numpy tiktoken
```

---

## Usage

### Training
```bash
# Single GPU
python train.py --batch_size 64 --context_length 1024

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train.py --batch_size 64 --context_length 1024 --ddp
```


---

## Acknowledgments
- Inspired by Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series.
- Uses flash attention for optimized performance.
- Based on the architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
```

### Notes:
1. Replace `LICENSE` with your actual license file if you have one.
2. If you have a diagram or screenshot of the architecture, add it under the **Features** section.
3. If you want to include a link to your dataset or pretrained weights, add a **Datasets** or **Pretrained Models** section.

Let me know if you need further adjustments!
