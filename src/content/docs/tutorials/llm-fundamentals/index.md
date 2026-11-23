---
title: "LLM Fundamentals"
description: "When a language model sees \"One day\", how does it decide what to write next? It turns those words into tokens and assigns a probability to each possible next token."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-13
---

When a language model sees "One day", how does it decide what to write next? It turns those words into tokens and assigns a probability to each possible next token. We train a small model to see how those probabilities change and evaluate how well it follows a story prompt.

This tutorial is built around a working language model we implement from scratch. We build a tokenizer and an 11-million-parameter Transformer, train it on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), and compare its responses before and after fine-tuning. That same implementation connects the chapters, from how text becomes tokens to how a model generates answers and how those answers are evaluated. Newcomers can follow the chapters in order, while engineers and researchers can revisit individual topics and explore the deep dives.

<aside class="cm-model-card cm-notebook-card" id="story-model">
<div class="cm-notebook-kicker" id="story-model-project">Project</div>
<p class="cm-notebook-title">Language model for storytelling</p>

The project builds a small language model from scratch to continue short children’s stories from an opening line. After fine-tuning, it can also write a story from a simple plot description and include requested words or features such as dialogue. Note that this small model may skip a requested word or leave out part of the plot due to its limited capacity.

For post-training, we use supervised fine-tuning (SFT) on [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct). An optional Direct Preference Optimization (DPO) lab takes a copy of the SFT model and runs one update to favor a story that uses every requested word over one that misses a word. Pretraining, SFT, and the DPO lab can all run on a single GPU.

<div class="cm-notebook-specs" role="region" aria-label="Model specifications" tabindex="0">

| Part | Value | Explained in |
| --- | --- | --- |
| Vocabulary | 4,096 tokens from a byte-level BPE (256 bytes, 4 special tokens, 3,836 merges) | [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/#vocabulary-size) |
| Input and output tables | one shared table, 1.57 million numbers | [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/#weight-tying) |
| Context window | 512 tokens | [Attention](/tutorials/llm-fundamentals/attention-and-positional-information/#quadratic-cost-and-the-key-value-cache) |
| Attention heads | 6 per block, 64 numbers wide, sharing 2 key-value heads | [Attention](/tutorials/llm-fundamentals/attention-and-positional-information/#multi-head-attention) |
| Positions | rotary position embedding | [Attention](/tutorials/llm-fundamentals/attention-and-positional-information/#rotary-position-embedding) |
| Width | 384 numbers per token | [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/#the-residual-stream) |
| Blocks | 6 | [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/#the-decoder-block) |
| Feed-forward width | 1,024, with SwiGLU | [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/#swiglu) |
| Parameters | 11.0 million | [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/#parameter-counting) |
| Weights in memory | 44.1 MB in float32, 22.0 MB in bf16 | [Pretraining](/tutorials/llm-fundamentals/pretraining-at-scale/#mixed-precision) |
| Pretraining data | 45.4 million tokens of TinyStories | [Pretraining](/tutorials/llm-fundamentals/pretraining-at-scale/#the-compute-budget) |
| Instruction tuning data | 60,000 [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) pairs | [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#supervised-fine-tuning) |

</div>

![Story model architecture](/images/llm-fundamentals/15-story-model-architecture.png)

The diagram follows the model from an input token to its next-token scores. Each block updates the same 384-dimensional vector. At the end, the output head turns that vector into 4,096 scores, one for each token in the vocabulary.

<div class="cm-notebook-actions"><a href="/notebooks/llm-fundamentals/source/#readme" data-notebook-expand="llm-fundamentals">Open code project</a></div>
</aside>

## Hardware requirement

The `story` profile uses one GPU on Apple Silicon or NVIDIA. TinyStories and TinyStories-Instruct stream over the network, and both training stages use the same 11-million-parameter model and 512-token context. The memory figures below are starting estimates for the default settings.

<div class="cm-hw">
<fieldset class="cm-hw-switch">
  <legend class="sr-only">Hardware platform</legend>
  <label>
    <input class="sr-only" type="radio" name="hardware-platform" value="apple" aria-controls="hw-apple-panel" checked>
    <span>Apple Silicon</span>
  </label>
  <label>
    <input class="sr-only" type="radio" name="hardware-platform" value="nvidia" aria-controls="hw-nvidia-panel">
    <span>NVIDIA</span>
  </label>
</fieldset>
<section class="cm-hw-card" id="hw-apple-panel" aria-labelledby="hw-mac">
<h3 id="hw-mac">Apple Silicon Mac</h3>
<dl>
<dt>Minimum requirement</dt><dd>Any M-series chip (M1 or later). The run uses the GPU through PyTorch's MPS backend.</dd>
<dt>Memory</dt><dd><strong>8 GB</strong> unified memory is a starting estimate. At the end of the longer M4 Max run, PyTorch reported 1.2 GiB of GPU driver memory. Python, the dataset stream and macOS also use unified memory.</dd>
<dt>System</dt><dd>macOS 15 or later for MPS with <a href="https://pytorch.org/blog/pytorch-2-15-release-blog/#deprecations-and-backwards-incompatible-changes">PyTorch 2.15</a>.</dd>
<dt>Precision</dt><dd>float32. MPS runs the same code without bf16 autocast.</dd>
<dt>Measured training time</dt><dd>About 21 minutes for pretraining and 9 minutes for fine-tuning on an M4 Max. Streaming, tokenizer training and evaluation take additional time.</dd>
</dl>
</section>
<section class="cm-hw-card" id="hw-nvidia-panel" aria-labelledby="hw-nvidia">
<h3 id="hw-nvidia">NVIDIA GPU</h3>
<dl>
<dt>Minimum requirement</dt><dd>One NVIDIA GPU with CUDA support and an estimated <strong>4 GB</strong> of GPU memory. Desktop examples include the GeForce RTX 3060 (12 GB) and GeForce RTX 4060 (8 GB).</dd>
<dt>System</dt><dd>Linux or Windows with a <a href="https://pytorch.org/get-started/locally/">CUDA-enabled PyTorch build</a> and a compatible NVIDIA driver.</dd>
<dt>Precision</dt><dd>bf16 autocast for matrix products when PyTorch reports support. The model uses float32 otherwise. Weights and optimizer state stay in float32.</dd>
<dt>Training measurements</dt><dd>The notebook prints peak GPU memory and throughput after pretraining. The runs reported here used Apple MPS.</dd>
</dl>
</section>
</div>

Both platforms also need Python 3.12, a network connection to stream the datasets, and free disk for PyTorch and two checkpoints of 132 MB each. That is about 2 GB on a Mac. On Linux or Windows the CUDA build of PyTorch alone takes several gigabytes. Both datasets are streamed, so they need little local disk space. Without a GPU, the default `cpu` profile still runs every stage on a small generated corpus. It needs no network and finishes in about 11 seconds on one CPU thread.

## Reading order and prerequisites

The first four chapters explain what the model predicts and how it makes a prediction. Language Modeling starts with the probability of a short sentence. Tokenization explains why the model might see more tokens than words in that sentence. Attention and the Transformer Block then follow those tokens through the model.

Pretraining and Post-Training show how the model is trained and fine-tuned to produce the two checkpoints compared in the notebook. The last three chapters use a trained model to explain how text is generated, how applications give the model context and tools, and how developers check the results. The chapter examples can be followed by hand. The notebook runs the same ideas in Python.

The following topics are recommended prerequisites for this tutorial.

- **Machine learning basics.** Training usually adjusts a model's parameters by gradient descent to lower a loss on training data. The model is judged on data it has not seen. Overfitting, train and test splits, and the learning rate come up throughout.
- **Neural networks.** A layer multiplies its input by a weight matrix and applies a nonlinearity, and backpropagation finds the gradient of the loss for every weight.
- **Probability and statistics.** Conditional probability, distributions, expected value, variance and likelihood. Language modeling rests on the chain rule of probability and on maximum likelihood.
- **Linear algebra.** Vectors, matrices, dot products and matrix multiplication, which make up nearly every computation inside a transformer.
- **Calculus.** Derivatives and the chain rule, enough to follow how a gradient flows back through a layer.

## Topics

| Chapter | What it covers |
|---|---|
| [Language Modeling](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/) | How a model predicts the next token, and how cross entropy, perplexity and bits per byte measure the prediction. |
| [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/) | How byte pair encoding cuts text into tokens, and how each token becomes an embedding vector. |
| [Attention](/tutorials/llm-fundamentals/attention-and-positional-information/) | How queries, keys and values let tokens read each other, and how position information is added back. |
| [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/) | How normalization, attention and the feed-forward network stack into a full model, with its parameter and FLOP counts. |
| [Pretraining](/tutorials/llm-fundamentals/pretraining-at-scale/) | How a data corpus, the AdamW optimizer and a compute budget produce a trained base model. |
| [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/) | How fine-tuning, preference learning and reinforcement learning turn a base model into an assistant. |
| [Inference](/tutorials/llm-fundamentals/inference-and-decoding/) | How a model generates text, how sampling picks each token, and what makes serving fast and cheap. |
| [LLM Applications](/tutorials/llm-fundamentals/using-llms-in-applications/) | How prompting, retrieval, tools and fine-tuning adapt a finished model to a task. |
| [Evaluation and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/) | How models are measured, and how hallucination, contamination and attacks are checked before release. |

[LLM Deep Dive](/tutorials/llm-deep-dive/) follows the same ideas into larger systems, where long prompts, GPU memory, and serving many users change the design. [Training Tool-Using Agent With RL](/tutorials/rl-for-llm-agents/) picks up after post-training and shows how to train and evaluate a model that makes several tool calls to complete a task.
