---
title: Welcome to Cook ML
date: 2026-11-15T12:00:00Z
authors: zheng
tags: [announcements]
excerpt: Why this site exists, how the tutorials are written, and what is coming after LLM Fundamentals.
---

Cook ML is a new home for long-form machine learning tutorials and shorter notes. This first post explains what the site is for and how to use it.

## Why another ML site

There are many good explanations of language models. Most of them pick one level: an intuitive story, a math derivation, or code. Learning tends to stick when the three are connected, so each tutorial here moves through them in order. It starts with an everyday example, then explains the mechanism, then writes the equation, and finally computes it on real numbers.

## Start here

The first tutorial is [LLM Fundamentals](/tutorials/llm-fundamentals/). It has nine chapters:

1. Language modeling and next-token prediction
2. Tokenization and embeddings
3. Attention and positional information
4. The transformer block and model anatomy
5. Pretraining at scale
6. Post-training, from base model to assistant
7. Inference and decoding
8. Using LLMs in applications
9. Evaluation, reliability, and safety

The chapters share one reference model, a 7B decoder with the shape of Llama 2 7B. Parameter counts, training budgets, cache sizes, and latencies all use the same numbers, so you can cross-check them from chapter to chapter.

## What comes next

Planned tutorials cover transformers in depth, retrieval-augmented generation, and ML systems. Blog posts will cover papers and practical lessons in between.
