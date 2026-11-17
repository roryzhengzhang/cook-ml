---
title: "LLM Deep Dive"
description: "The reference Llama 2 model from LLM Fundamentals fits on a single 80 GB GPU for inference. Training it does not fit, because AdamW state alone is 108 GB before a single activation is stored."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-16
---

The [reference Llama 2 model](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/#reference-llama-2-model) from LLM Fundamentals fits on a single 80 GB GPU for inference. Training it does not fit, because AdamW state alone is 108 GB before a single activation is stored. Its 70B sibling needs about 1.1 TB of training state, so it has to be split across many GPUs. Serving it to a few hundred concurrent users is a scheduling problem as much as a modeling one. LLM Fundamentals explains what each part of a large language model does. This series covers what happens when those parts meet real limits, such as contexts of a hundred thousand tokens, thousands of accelerators, a budget that pays for 13 billion active parameters out of 47 billion, a domain the base model has barely seen, and a latency target on every request.

Each chapter takes one topic and goes a level down, into the mechanism inside, its cost worked out to actual numbers, the ways it fails and the design decision it supports. Each mechanism is also placed among the choices current open models make, such as latent attention in DeepSeek-V3, hybrid linear attention in Qwen3-Next and interleaved local and global layers in Gemma 3.

Three reference models recur across the chapters, and their numbers agree everywhere. The [7B decoder](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/#reference-llama-2-model) from Fundamentals, shaped like Llama 2 7B, stays the default. A 70B decoder shaped like Llama 2 70B is the model that does not fit on one GPU, so the parallelism and serving chapters size it. It has 80 layers, width 8,192 and 8 grouped key-value heads. A mixture-of-experts model shaped like Mixtral 8x7B carries the expert chapter. It has 46.7 billion parameters, of which 12.9 billion are active per token. Hardware numbers assume H100 GPUs, eight to a node. Plots are either exact calculations under stated assumptions or explicitly marked teaching schematics, not measured benchmark results.

## Reading order and prerequisites

The chapters move from the model to the machines and then to the product. Efficient attention and long context come first, because the cost of attention is what limits context length and shapes everything about long documents. Parallelism and numerics follow. They cover how a model too large for one GPU is trained, how the work is split, and how low-precision arithmetic keeps a long run stable. Mixture of experts comes next, since it grows the parameter count without growing the cost per token and relies on the communication patterns introduced just before it. Efficient adaptation and domain adaptation turn a trained model into one that fits a specialized task. Serving systems closes the series by putting the finished model in front of users at scale.

All nine chapters of LLM Fundamentals are assumed, and the chapters here link back to them where they build on a specific section.

## Topics

| Chapter | What it covers |
|---|---|
| [Efficient Attention](/tutorials/llm-deep-dive/efficient-attention-mechanisms/) | Where attention's time and memory go, and how grouped and latent key-value heads, local and trained sparse attention, FlashAttention, and hybrid linear-attention and state space layers reduce them. |
| [Long Context](/tutorials/llm-deep-dive/long-context-and-document-models/) | How RoPE scaling, staged long-context training and mixed positional schemes extend a model past its training length, and whether it actually uses evidence spread across a long document. |
| [Parallelism](/tutorials/llm-deep-dive/distributed-training-and-parallelism/) | How data, tensor, pipeline, context and expert parallelism and sharded optimizer state split one training run across thousands of GPUs, as in the Llama 3 and DeepSeek-V3 layouts. |
| [Numerics and Stability](/tutorials/llm-deep-dive/numerics-and-training-stability/) | How bf16, FP8 and microscaled 4-bit arithmetic, recomputation, QK-norm and other stabilizers, newer optimizers such as Muon, and checkpointing keep a long training run fast and on track. |
| [Mixture of Experts](/tutorials/llm-deep-dive/mixture-of-experts-at-scale/) | How a router sends each token to a few of many fine-grained experts, how the load is balanced with or without an auxiliary loss, and what expert parallelism costs. |
| [Efficient Adaptation](/tutorials/llm-deep-dive/parameter-efficient-adaptation/) | How adapters, LoRA and its variants, prefix tuning and quantized adaptation share one frozen base across many tasks. |
| [Domain Adaptation](/tutorials/llm-deep-dive/domain-adaptation-and-fine-tuning/) | How to diagnose what a model lacks in a new domain, and how continued pretraining, fine-tuning and evaluation over time fix it. |
| [Serving Systems](/tutorials/llm-deep-dive/inference-systems-at-scale/) | How rooflines, chunked prefill, prefill-decode disaggregation, prefix and offloaded caches, low-bit inference and speculative decoding serve large models within latency targets. |
