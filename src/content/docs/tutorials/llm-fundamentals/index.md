---
title: "LLM Fundamentals"
description: "Large language models went from research projects to everyday tools in a few years, and a lot of what gets said about them is guesswork."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-13
---

Large language models went from research projects to everyday tools in a few years, and a lot of what gets said about them is guesswork. Why does a model that tops a benchmark disappoint on your own data? Why does each word of an answer cost more to produce than each word of the prompt? Why does an assistant sometimes agree with you when you are wrong? None of these answers is a secret. They follow from a small number of mechanisms, and once you can see those mechanisms, choices that look arbitrary in a product decision or a research paper start to make sense.

This tutorial is for anyone who wants to understand how these models work, from newcomers to the field to engineers and researchers who already build with them. For newcomers, the chapters build on each other in order and cover the key ideas behind large language models step by step, from how text becomes tokens to how a finished model is trained, served and evaluated. Each idea starts with a small example before any math appears, so every step can be followed and checked by hand. For engineers and researchers who know the basics, the series works as a refresher: every chapter and section is one click away in the sidebar, and the deep dives hold the extra detail.

## Topics

### [Language Modeling](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/)

How the chain rule turns a sentence into a product of next-token probabilities, why counting word sequences breaks down, and how a neural network shares statistical strength across contexts. Softmax, cross-entropy, and perplexity are computed by hand, bits per byte makes models with different tokenizers comparable, and maximum likelihood turns out to match the data distribution. The chapter closes with teacher forcing and the mismatch between training and generation.

### [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/)

The choice between characters, words, and subwords, and byte pair encoding run merge by merge alongside WordPiece and Unigram. Byte fallback, vocabulary size, fertility across languages, and the special tokens and chat templates that frame every conversation come next. The second half covers the embedding lookup that turns tokens into vectors, similarity, tied input and output tables, and the failures tokenization causes in counting, spelling, and arithmetic.

### [Attention](/tutorials/llm-fundamentals/attention-and-positional-information/)

Attention computed from queries, keys, and values on a three-token example, with the square-root scaling, the causal mask, and the split of the width into heads. Attention by itself ignores order, so sinusoidal positions, rotary position embedding, and ALiBi put it back, and the chapter shows what breaks beyond the trained context. It closes with the quadratic cost and the key-value cache, which Efficient Attention Mechanisms develops in depth.

### [Transformer Block](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/)

The residual stream traced through normalization, attention, and the feed-forward network, with LayerNorm set against RMSNorm and pre-norm against post-norm. SwiGLU's width is derived, the decoder block assembled, and all 6.74 billion parameters of the reference model counted. The chapter then derives 2N forward and 6N training FLOPs per token, compares the encoder, decoder, and encoder-decoder families with their objectives, and separates total from active parameters in a mixture of experts.

### [Pretraining](/tutorials/llm-fundamentals/pretraining-at-scale/)

How a corpus is built with deduplication, filtering, and mixing, and how documents are packed into training windows. AdamW goes step by step, followed by the 16 bytes of training state per parameter and the choice of learning-rate schedule and bf16 precision. Loss spikes, a run's cost from $C\approx6ND$, and why compute-optimal scaling recommends a larger model than the ones people actually deploy come next, and the chapter closes with mid-training, the short final stage that switches to a curated mixture and longer contexts.

### [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/)

What a base model does, and how supervised fine-tuning changes it with loss only on response tokens. A reward model learns from pairwise preferences, a KL penalty keeps reinforcement learning in check, and direct preference optimization offers a simpler route. Verifiable rewards with GRPO, reasoning models and test-time compute, distillation, reward hacking and sycophancy follow, along with how the failure you observe points to a recipe.

### [Inference](/tutorials/llm-fundamentals/inference-and-decoding/)

Prompt processing and token generation run at very different speeds, and the chapter sizes the key-value cache to show why generation is limited by memory bandwidth. Greedy and beam search sit beside temperature, top-k, top-p, and min-p sampling, with repetition control and grammar-constrained output on top. Continuous batching, paged caches, quantization, and speculative decoding finish the chapter, measured by time to first token and time per output token.

### [LLM Applications](/tutorials/llm-fundamentals/using-llms-in-applications/)

System prompts, in-context examples, and chain of thought steer a frozen model without touching its weights. Retrieval and vector search add knowledge, long contexts have limits worth knowing, and tools plus an agent loop extend what the model can do, with reliability that compounds at every step. LoRA and QLoRA adapt the weights cheaply, and the failure being fixed decides between prompting, retrieval, and fine-tuning.

### [Evaluation and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/)

Loss, benchmark accuracy, and human preference measure different things, and the chapter covers an unbiased estimate of pass@k and how large an evaluation set must be to detect a given difference. Contamination checks and a trustworthy LLM judge come next, then where hallucination comes from, how calibration is measured, and how claims are checked against evidence. Jailbreaks and prompt injection, memorization and bias, interpretability basics, and the release gates that decide whether a model ships complete the picture.
