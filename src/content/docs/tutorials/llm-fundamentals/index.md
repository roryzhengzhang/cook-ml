---
title: "LLM Fundamentals"
description: "Large language models went from research projects to everyday tools in a few years, and a lot of what gets said about them is guesswork."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-13
---

Large language models went from research projects to everyday tools in a few years, and a lot of what gets said about them is guesswork. Why does a model that tops a benchmark disappoint on your own data? Why does each word of an answer cost more to produce than each word of the prompt? Why does an assistant sometimes agree with you when you are wrong? None of these answers is a secret. They follow from a small number of mechanisms, and once you can see those mechanisms, choices that look arbitrary in a product decision or a research paper start to make sense.

This tutorial is for anyone who wants to understand how these models work, from readers new to large language models to engineers and researchers who already build with them. For newcomers, the chapters build on each other in order and cover the key ideas behind large language models step by step, from how text becomes tokens to how a finished model is trained, served and evaluated. Each idea starts with a small example before any math appears, so every step can be followed and checked by hand. For engineers and researchers who know the basics, the series works as a refresher: every chapter and section is one click away in the sidebar, and the deep dives hold the extra detail.

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
