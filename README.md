# Custom NLP Question-Answering Pipeline using PyTorch

An end-to-end Deep Learning pipeline built using foundational **PyTorch** components to solve a closed-domain Question-Answering (QA) task. 

Rather than relying on high-level orchestration wrappers (like Hugging Face or PyTorch Lightning) or pre-trained models, this project implements the fundamental core building blocks of a Natural Language Processing (NLP) system—including custom word-level tokenization, manual vocabulary mapping, a custom PyTorch `Dataset` wrapper, and a Vanilla Recurrent Neural Network (`nn.RNN`) architecture.

## 🚀 Key Features Implemented

- **Custom Text Preprocessing:** Custom rule-based string cleaning and word-level tokenization.
- **Dynamic Vocabulary Builder:** Automatically constructs a single dictionary corpus mapping words to unique numerical indices, including an explicit Out-Of-Vocabulary (`<UNK>`) fallback token.
- **PyTorch Structural Data Pipeline:** Tailored `QADataset` subclass and structured `DataLoader` abstractions handling data indexing and multi-dimensional tensor construction on the fly.
- **Vanilla Recurrent Neural Network:** Built an explicit `nn.Module` using custom input lookup sequence tracking, an embedding layer, a recurrent network tracking long-range step vectors, and a fully connected linear classification head.
- **Robust Training Loop Optimization:** Monitored cross-entropy convergence step-by-step using Adam optimization and implemented probability threshold checking for inference confidence.

---
