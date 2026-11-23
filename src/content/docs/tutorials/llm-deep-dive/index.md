---
title: "LLM Deep Dive"
description: "The story model from LLM Fundamentals has 11 million parameters, and its whole training state takes 176 MB. At larger scales the same calculations lead to different choices."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-16
---

The [story model](/tutorials/llm-fundamentals/#story-model) from LLM Fundamentals has 11 million parameters, and its whole training state takes 176 MB. At larger scales the same calculations lead to different choices. Llama 2 7B fits on a single 80 GB GPU for inference, while its AdamW training state alone takes 108 GB before activations. Llama 2 70B needs about 1.1 TB of training state, so training spans many GPUs. Serving a model that size to a few hundred concurrent users also takes careful scheduling. LLM Fundamentals explains what each part of a large language model does. LLM Deep Dive shows how those parts behave under real limits, such as contexts of a hundred thousand tokens, thousands of accelerators, a budget that pays for 13 billion active parameters out of 47 billion, a domain the base model has barely seen, and a latency target on every request.

Each chapter takes one topic and goes a level down, into the mechanism inside, its cost worked out to actual numbers, the ways it fails and the design decision it supports. Each mechanism is also placed among the choices current open models make, such as latent attention in DeepSeek-V3, hybrid linear attention in Qwen3-Next and interleaved local and global layers in Gemma 3.

LLM Fundamentals follows its 11-million-parameter story model, which trains on one GPU in minutes. Three larger models show where memory, communication and latency change the design. The 7B model shows the limits of one GPU, the 70B model shows what changes when weights span several GPUs, and Mixtral 8x7B shows how the total and active parameter counts differ in a mixture of experts. Hardware numbers assume H100 GPUs, eight to a node. Plots are either exact calculations under stated assumptions or explicitly marked teaching schematics, not measured benchmark results.

<aside class="cm-model-card" id="scale-reference-models">
<p class="cm-model-title">Scale reference models</p>

Like the [story model](/tutorials/llm-fundamentals/#story-model), these three decoders use pre-norm blocks with RMSNorm, rotary positions and SwiGLU. Their attention layouts, vocabulary sizes and feed-forward layers differ. Their shapes copy Meta's openly released [Llama 2](https://www.llama.com/llama2/) 7B and 70B and Mistral's [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts), whose designs are published, so every number can be checked. Whenever a chapter says "the reference 7B", "the 70B" or "the MoE model", it means one of these.

| Part | 7B | 70B | MoE |
| --- | --- | --- | --- |
| Blocks | 32 | 80 | 32 |
| Width | 4,096 | 8,192 | 4,096 |
| Query heads | 32 of width 128 | 64 of width 128 | 32 of width 128 |
| Key-value heads | 32 (full multi-head) | 8 (grouped) | 8 (grouped) |
| Feed-forward width | 11,008, SwiGLU | 28,672, SwiGLU | 8 experts of 14,336, top-2 |
| Vocabulary | 32,000, untied tables | 32,000, untied tables | 32,000, untied tables |
| Context | 4,096 tokens | 4,096 tokens | 32,768 tokens |
| Parameters | 6.74 billion | 69.0 billion | 46.7 billion, 12.9 billion active |
| Weights in bf16 | 13.48 GB | 137.95 GB | 93.41 GB |
| KV cache per token | 512 KiB | 320 KiB | 128 KiB |
| AdamW training state | 108 GB | 1.10 TB | 747 GB |

</aside>

## Reading order and prerequisites

The chapters move from the model to the machines and then to the product. Efficient attention and long context come first, because the cost of attention is what limits context length and shapes everything about long documents. Parallelism and numerics follow. They cover how a model too large for one GPU is trained, how the work is split, and how low-precision arithmetic keeps a long run stable. Mixture of experts comes next, since it grows the parameter count without growing the cost per token and relies on the communication patterns introduced just before it. Efficient adaptation and domain adaptation turn a trained model into one that fits a specialized task. The final chapter explains how serving systems put the finished model in front of users at scale.

The attention chapters build on [queries, keys and values](/tutorials/llm-fundamentals/attention-and-positional-information/#queries-keys-and-values) and the [KV cache](/tutorials/llm-fundamentals/inference-and-decoding/#the-key-value-cache). The training chapters use [AdamW](/tutorials/llm-fundamentals/pretraining-at-scale/#adamw) and [mixed precision](/tutorials/llm-fundamentals/pretraining-at-scale/#mixed-precision). The serving chapter starts from [prefill and decode](/tutorials/llm-fundamentals/inference-and-decoding/#prefill-and-decode-phases).

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
