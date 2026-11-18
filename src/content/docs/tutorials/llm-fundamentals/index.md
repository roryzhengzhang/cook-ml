---
title: "LLM Fundamentals"
description: "Large language models went from research projects to everyday tools in a few years, and a lot of what gets said about them is guesswork."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-13
---

Large language models went from research projects to everyday tools in a few years, and a lot of what gets said about them is guesswork. Why does a model that tops a benchmark disappoint on your own data? Why does each word of an answer cost more to produce than each word of the prompt? Why does an assistant sometimes agree with you when you are wrong? None of these answers is a secret. They follow from a small number of mechanisms, and once you can see those mechanisms, choices that look arbitrary in a product decision or a research paper start to make sense.

This tutorial is built around a working language model we implement from scratch. In the [project notebook](/tutorials/llm-fundamentals/notebook/), we build a tokenizer and an 11-million-parameter Transformer, train it on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), and compare its responses before and after fine-tuning. That same implementation connects the chapters, from how text becomes tokens to how a model generates answers and how those answers are evaluated. Newcomers can follow the chapters in order, while engineers and researchers can revisit individual topics and explore the deep dives.

<aside class="cm-model-card cm-notebook-card" id="story-model">
<div class="cm-notebook-kicker" id="story-model-project">Project</div>
<p class="cm-notebook-title">Language model for storytelling</p>

The project builds a small Transformer from scratch and trains it to write short stories. During pretraining, the model learns to predict the next token in [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Given an opening such as "One day", the goal is to continue it as a coherent story.

For post-training, we use supervised fine-tuning (SFT) on [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct). A request can ask the model to use specific words, include a sentence, follow a plot summary, or add features such as dialogue and a plot twist. An optional Direct Preference Optimization (DPO) lab takes a copy of the SFT model and runs one update to favor a story that uses every requested word over one that misses a word. The notebook gives the base and SFT models the same held-out requests. It checks words and sentences automatically and shows both stories for reviewing plot and features. Pretraining, SFT, and the DPO lab can all run on a single GPU.

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

<div class="cm-notebook-actions"><a href="/tutorials/llm-fundamentals/notebook/">Browse the notebook <span aria-hidden="true">↗</span></a></div>
</aside>

![Story model architecture](/images/llm-fundamentals/15-story-model-architecture.png)

The figure above shows the overall architecture of the language model we’ll implement in the project notebook.

## Reading order and prerequisites

The chapters follow the order in which a model is built and then used. Language modeling sets the goal of predicting the next token, tokenization decides what a token is, and attention and the transformer block assemble the network that makes the prediction. Those four form the foundation, and the rest of the series uses their terms without defining them again. Pretraining and post-training then turn that network into an assistant, and inference, applications and evaluation cover what happens once the model is finished. These later topics lean on the foundation more than on each other. Pretraining and inference both turn the model's parameter count into time and memory, while applications and evaluation draw on everything before them.

The series builds on the basics of machine learning rather than teaching them. Every example is still worked by hand rather than in code, so no programming is needed. The chapters recall a term from these subjects briefly where they first use it, but they do not teach the subjects themselves:

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
