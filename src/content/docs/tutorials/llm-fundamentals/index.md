---
title: "LLM Fundamentals"
description: "A large language model is a program that reads text as a sequence of small pieces and repeatedly predicts the next piece."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-13
---

A large language model is a program that reads text as a sequence of small pieces and repeatedly predicts the next piece. Everything an interviewer asks about it follows that one prediction step through its life: how text becomes pieces, how the network turns earlier pieces into a guess, how the guess is trained on trillions of examples, how a raw predictor becomes a helpful assistant, how answers are produced quickly enough to serve, how applications put the model to work, and how anyone can tell whether it is good and safe. These nine notes cover that path in order, from a single probability to a released model.

The intended reader is new to the field, so every note opens each concept with a plain example before any notation. The depth still targets senior and staff interviews. A senior answer explains the mechanism, computes its cost, and proposes a measurement. A staff answer also states which constraint decides the choice, what evidence would reverse it, and who owns that evidence. These are study goals rather than any company's hiring rubric.

## Overview

The opening of each note introduces the problem and one figure. Numbered sections then build the mechanism from a plain explanation to the mathematics, with at least one worked example computed to actual numbers.

Two running examples recur across the notes. A toy model with two- or three-number vectors and a tiny vocabulary keeps every mechanism checkable by hand. A reference 7B decoder modeled on Llama 2 7B supplies real-scale costs: 32 layers, width 4,096, 32 attention heads of width 128, a 32,000-token vocabulary, a feed-forward width of 11,008, untied embeddings, and 2-byte bf16 weights. Its parameter count, training budget, cache size, and decode latency appear in several notes, and they use the same numbers everywhere so you can cross-check them.

## Topics

### [Language Modeling and Next Token Prediction](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/)

Turn a sentence into a product of next-token probabilities with the chain rule, then see why counting word sequences breaks down and how a neural network shares statistical strength across contexts. Compute softmax, cross-entropy, and perplexity by hand, convert to bits per byte so models with different tokenizers can be compared, and connect maximum likelihood to matching the data distribution. Close with teacher forcing and the mismatch between training and generation.

### [Tokenization and Embeddings](/tutorials/llm-fundamentals/tokenization-and-embeddings/)

Choose between characters, words, and subwords, then run byte pair encoding merge by merge and compare it with WordPiece and Unigram. Study byte fallback, vocabulary size, fertility across languages, and the special tokens and chat templates that frame every conversation. Turn tokens into vectors with an embedding lookup, measure similarity, tie input and output tables, and explain the failures that tokenization causes in counting, spelling, and arithmetic.

### [Attention and Positional Information](/tutorials/llm-fundamentals/attention-and-positional-information/)

Compute attention from queries, keys, and values on a three-token example, explain the square-root scaling and the causal mask, and split the width into heads. Show that attention by itself ignores order, then add sinusoidal positions, rotary position embedding, and ALiBi, and explain what breaks beyond the trained context. Close with the quadratic cost and the key-value cache, which Efficient Attention Mechanisms develops in depth.

### [The Transformer Block and Model Anatomy](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/)

Follow the residual stream through normalization, attention, and the feed-forward network, and compare LayerNorm with RMSNorm and pre-norm with post-norm. Derive SwiGLU's width, assemble the decoder block, and count all 6.74 billion parameters of the reference model. Derive 2N forward and 6N training FLOPs per token, compare encoder, decoder, and encoder-decoder families with their objectives, and separate total from active parameters in a mixture of experts.

### [Pretraining at Scale](/tutorials/llm-fundamentals/pretraining-at-scale/)

Build a corpus with deduplication, filtering, and mixing, then pack documents into training windows. Step through AdamW, account for 16 bytes of training state per parameter, and choose a learning-rate schedule and bf16 precision. Handle loss spikes, compute a run's cost with $C\approx6ND$, and explain why compute-optimal scaling recommends a larger model than the ones people actually deploy.

### [Post Training from Base Model to Assistant](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/)

Explain what a base model does, then add supervised fine-tuning with loss only on response tokens. Train a reward model on pairwise preferences, regularize reinforcement learning with a KL penalty, and compare it with direct preference optimization. Cover verifiable rewards with GRPO, reasoning models and test-time compute, distillation, reward hacking and sycophancy, and how to choose a recipe from the failure you observe.

### [Inference and Decoding](/tutorials/llm-fundamentals/inference-and-decoding/)

Separate prompt processing from token generation, size the key-value cache, and show why generation is limited by memory bandwidth. Compare greedy and beam search with temperature, top-k, top-p, and min-p sampling, then add repetition control and grammar-constrained output. Finish with continuous batching, paged caches, quantization, and speculative decoding, measured by time to first token and time per output token.

### [Using LLMs in Applications](/tutorials/llm-fundamentals/using-llms-in-applications/)

Steer a frozen model with system prompts, in-context examples, and chain of thought. Add knowledge with retrieval and vector search, use long contexts carefully, and connect the model to tools and an agent loop whose reliability compounds with every step. Adapt weights cheaply with LoRA and QLoRA, and choose between prompting, retrieval, and fine-tuning from the failure you are trying to fix.

### [Evaluation Reliability and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/)

Separate loss, benchmark accuracy, and human preference, estimate pass@k without bias, and size an evaluation set for the difference you need to detect. Detect contamination and make an LLM judge trustworthy. Explain where hallucination comes from, measure calibration, and check that claims are supported by evidence. Cover jailbreaks and prompt injection, memorization and bias, interpretability basics, and the release gates that decide whether a model ships.

## Cross cutting quick reference

The table maps a problem you might observe to the first thing to measure, the action to try, and the evidence you need before committing to it. The paragraphs after it collect the formulas that the nine notes use most often, each worked on the reference 7B model.

| Observed problem | First measurement | Candidate action | Evidence needed before committing |
|---|---|---|---|
| Two models with different tokenizers report different perplexity | Bits per byte on the same text | Compare on a tokenizer-independent unit | Ranking holds on downstream tasks |
| Non-English users pay more and fit less context | Tokens per word by language | Retrain or extend the tokenizer | Lower fertility without a quality loss |
| Training runs out of memory | Bytes per parameter by state and activations | Sharding, recomputation, or parameter-efficient tuning | The same run fits at acceptable step time |
| Loss spikes mid-run | Gradient norm and batch contents near the spike | Clipping, lower learning rate, skip data, restart | Loss recovers and stays stable |
| Base model rambles or ignores instructions | Output format audit on held-out prompts | Supervised fine-tuning | Better held-out instruction following |
| Reward rises while answers get worse | Human or rule-based audit of high-reward samples | Stronger KL penalty or better reward signal | True quality tracks the reward |
| Generation is slow at batch one | Time per output token against the bandwidth bound | Quantization or speculative decoding | Faster tokens at matched quality |
| Too few users fit per GPU | Cache bytes per active token | Grouped key-value heads, paged cache, cache quantization | More requests meeting the same latency target |
| Answers miss fresh or private facts | Retrieval recall and answer accuracy with oracle evidence | Retrieval-augmented generation | Correct and supported answers on new facts |
| Confident wrong answers | Calibration and supported-claim rate | Grounding, abstention, citation checks | Fewer unsupported claims at acceptable coverage |
| Benchmark gain does not show up for users | Contamination check and fresh held-out set | Rebuild the evaluation | Gain persists on uncontaminated data |

**Probability and loss.** A sentence's probability is the product of next-token probabilities. Cross-entropy is the average of $-\ln p$ over tokens, and perplexity is $e$ raised to that average, so a loss of 2.3 nats per token is a perplexity of about 10. See [Language Modeling and Next Token Prediction](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/).

**Parameters.** One decoder layer of the reference model has $4d^2$ attention parameters plus $3\cdot d\cdot d_{ff}$ feed-forward parameters, where $d$ is the width and $d_{ff}$ is the feed-forward width. That is 202.4 million per layer and 6.74 billion in total with the two embedding tables, or 13.48 GB in bf16. See [The Transformer Block and Model Anatomy](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/).

**Compute.** Training costs about $6N$ FLOPs per token for $N$ parameters, so $C\approx6ND$ for $D$ training tokens. The reference model on 2 trillion tokens needs about $8.1\times10^{22}$ FLOPs. Compute-optimal scaling at roughly 20 tokens per parameter would spend the same budget on a model of about 26 billion parameters. See [Pretraining at Scale](/tutorials/llm-fundamentals/pretraining-at-scale/).

**Training memory.** Mixed-precision AdamW keeps about 16 bytes per parameter before activations, or about 108 GB for the reference model. QLoRA keeps the frozen base in 4 bits, about 3.4 GB, and trains only small adapters. See [Pretraining at Scale](/tutorials/llm-fundamentals/pretraining-at-scale/) and [Using LLMs in Applications](/tutorials/llm-fundamentals/using-llms-in-applications/).

**Serving.** The key-value cache holds $2\cdot L\cdot h_{kv}\cdot d_h\cdot s$ bytes per token, where $L$ is layer count, $h_{kv}$ is key-value head count, $d_h$ is head width, and $s$ is bytes per number. That is 0.5 MiB per token for the reference model, or 2 GiB for a 4,096-token sequence. At batch one, each new token reads all 13.48 GB of weights, about 4 ms at 3.35 TB/s. See [Inference and Decoding](/tutorials/llm-fundamentals/inference-and-decoding/).

**Adaptation.** LoRA of rank $r$ on a matrix of shape $d_{in}\times d_{out}$ trains $r(d_{in}+d_{out})$ parameters, or 131,072 instead of 16.8 million at rank 16 on a 4,096 by 4,096 matrix. See [Using LLMs in Applications](/tutorials/llm-fundamentals/using-llms-in-applications/).

**Evaluation.** With $n$ samples of which $c$ are correct, unbiased pass@k is $1-\binom{n-c}{k}/\binom{n}{k}$. Accuracy measured on $m$ items has a standard error of about $\sqrt{p(1-p)/m}$ for true accuracy $p$, which is about 2.0 points at 500 items and 70% accuracy. See [Evaluation Reliability and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/).

## Answers that consistently score well

Each habit below separates a strong answer from one that is merely correct, and each links to the note that develops it.

- Name the unit before comparing numbers. Perplexity depends on the tokenizer, and throughput depends on whether you mean prompt or generated tokens. [Language Modeling and Next Token Prediction](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/) and [Inference and Decoding](/tutorials/llm-fundamentals/inference-and-decoding/)
- Trace a failure back to the tokenizer before blaming the model when it involves spelling, digits, whitespace, or non-English text. [Tokenization and Embeddings](/tutorials/llm-fundamentals/tokenization-and-embeddings/)
- Count parameters and FLOPs from the configuration rather than quoting a model name. [The Transformer Block and Model Anatomy](/tutorials/llm-fundamentals/the-transformer-block-and-model-anatomy/)
- Give a training budget as tokens, FLOPs, accelerator days, and bytes of state, with the utilization assumption stated. [Pretraining at Scale](/tutorials/llm-fundamentals/pretraining-at-scale/)
- Diagnose whether a model lacks knowledge, behavior, or reasoning before choosing retrieval, supervised tuning, or reinforcement learning. [Post Training from Base Model to Assistant](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/) and [Using LLMs in Applications](/tutorials/llm-fundamentals/using-llms-in-applications/)
- Say which resource limits generation, weights or cache bandwidth or compute, before proposing an optimization. [Inference and Decoding](/tutorials/llm-fundamentals/inference-and-decoding/)
- Report an evaluation result with its uncertainty, its contamination check, and the judge's measured agreement with people. [Evaluation Reliability and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/)

## Integrated design exercise

Practice this scenario: you own a 7B assistant from data to release. You have a fixed budget of 1,024 accelerators for one month, a small set of high-quality instruction and preference data, a larger unlabeled corpus, and a product that must answer questions about internal documents with citations at a latency target of 300 ms to first token and 30 ms per output token.

Start with the budget. Compute training FLOPs and days for the planned token count, the bytes of training state, and the cache and weight memory at the expected number of concurrent users. Decide whether pretraining, continued pretraining, or only post-training fits, and say what you would buy with the remainder.

Then design the behavior. Choose the tokenizer and chat template, the supervised and preference data, and whether verifiable rewards apply to any part of the task. Put retrieval in front for document knowledge rather than training facts into weights. Define the evaluation before training: a contamination-checked held-out set, a supported-claim rate for citations, a judge validated against human labels, and a red-team pass for prompt injection through retrieved documents.

Finish with serving and release. Pick quantization, batching, and speculative decoding from the measured bottleneck, set per-risk release thresholds, and name the owners of data refresh, evaluation, and rollback. At staff level, also state the measurement that would make you abandon the chosen design.

## Suggested study order

First pass: read the opening of all nine notes in order. Explain each opening figure aloud without equations.

Second pass: work through chapters 1 to 4 in one sitting, because each builds directly on the previous one. Recompute softmax, cross-entropy, a BPE merge, an attention output, and the 7B parameter count without looking. Then study chapters 5 and 7 together, since both are about budgets, and recompute $6ND$, optimizer memory, cache size, and the decode bound.

Third pass: study chapters 6, 8, and 9. For each answer, volunteer one number or one measurement. Rehearse the integrated exercise once with a modeling emphasis and once with a systems emphasis.

If time is short, prioritize cross-entropy and perplexity, BPE, the attention calculation, parameter and FLOP counting, $6ND$ with Chinchilla, the key-value cache and bandwidth bound, SFT versus DPO versus RL, RAG versus fine-tuning, and hallucination with calibration.
