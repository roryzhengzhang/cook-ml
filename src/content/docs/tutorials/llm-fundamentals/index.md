---
title: "LLM Fundamentals"
description: "When a language model sees \"One day\", how does it decide what to write next? This series follows that choice from the text the model reads to the probabilities it produces."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-13
---

When a language model sees "One day", how does it decide what to write next? This series follows that choice from the text the model reads to the probabilities it produces. The same small model gives us a way to see how training changes those probabilities and how to tell whether its answers have improved.

This tutorial is built around a working language model we implement from scratch. In the [project notebook](/tutorials/llm-fundamentals/notebook/), we build a tokenizer and an 11-million-parameter Transformer, train it on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), and compare its responses before and after fine-tuning. The project is designed to run on a single consumer GPU, including the built-in GPU on an Apple Silicon Mac. That same implementation connects the chapters, from how text becomes tokens to how a model generates answers and how those answers are evaluated. Newcomers can follow the chapters in order, while engineers and researchers can revisit individual topics and explore the deep dives.

<aside class="cm-model-card cm-notebook-card" id="story-model">
<div class="cm-notebook-kicker" id="story-model-project">Project</div>
<p class="cm-notebook-title">Language model for storytelling</p>

The project builds a small language model from scratch.

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

The diagram follows the model from an input token to its next-token scores. Each block updates the same 384-number vector. At the end, the output head turns that vector into 4,096 scores, one for each token in the vocabulary.

## Reading order and prerequisites

The first four chapters explain what the model predicts and how it makes a prediction. Language Modeling starts with the probability of a short sentence. Tokenization explains why the model might see more tokens than words in that sentence. Attention and the Transformer Block then follow those tokens through the network.

Pretraining and Post-Training use that network to build the two checkpoints compared in the notebook. The last three chapters use a trained model to explain how text is generated, how applications give the model context and tools, and how teams check the results. The chapter examples can be followed by hand. The notebook runs the same ideas in Python.

Some familiarity with the following topics will make the worked examples easier to follow.

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
