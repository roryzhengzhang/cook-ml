---
title: "Train a story model from scratch"
description: "An 11-million-parameter storytelling model trained from scratch on TinyStories and fine-tuned on TinyStories-Instruct."
slug: tutorials/llm-fundamentals/notebook
---

The [story model from LLM Fundamentals](/tutorials/llm-fundamentals/#story-model) learns to continue children's stories during pretraining. Fine-tuning on [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) teaches a copy to write stories that follow requests. The notebook compares both versions on held-out requests and links its code to the chapters that explain it.

<div class="cm-notebook-actions cm-notebook-page-actions">
  <a class="cm-notebook-download" href="/notebooks/llm-fundamentals.ipynb" download>Download .ipynb</a>
</div>

## What the notebook trains

The notebook has three main stages. Pretraining learns to continue stories, instruction tuning learns from request and story pairs, and evaluation checks responses to unseen requests. Each stage uses its own slice of the data. Normalized story hashes keep exact duplicates out of the held-out sets, although they cannot detect every near-duplicate.

### Next-token pretraining on TinyStories

[TinyStories](https://arxiv.org/abs/2305.07759) is a public set of about 2.1 million short stories that GPT-3.5 and GPT-4 wrote with words a typical three- or four-year-old knows. The vocabulary is small and the grammar is simple, so a model with 11 million parameters can learn to write stories that hang together. On ordinary web text a model this size mostly produces fragments. A typical story starts "Once upon a time, there was a little girl named Lily." and runs about a hundred words.

To compare runs, the notebook needs to give each model the same stories. It reads a fixed version of TinyStories, called a pinned revision, and removes exact duplicates. Before the tokenizer sees any text, it sets aside 1,500 stories from the official validation split for measuring the base model. It also hashes the training stories and leaves about 18% of story IDs available for instruction tuning. The hash is based on the story text, so the same story gets the same ID in both datasets. The tokenizer is a byte-level BPE trained on 20,000 pretraining stories. It starts from the 256 byte values, reserves 4 special tokens, and learns 3,836 merges to reach 4,096 IDs.

Pretraining packs 50 million next-token targets into 141,062 rows of 512 slots. The notebook shuffles those rows and uses 16 per update. It processes four rows at a time and accumulates their gradients to limit GPU memory. Over 8,000 updates the model predicts 45.4 million real tokens.

The longer run took about 21 minutes to pretrain on an M4 Max. On 1,500 TinyStories validation stories, the base model reached a perplexity of 2.86, compared with 3.85 after 2,500 updates in the shorter run. A bigram model reaches 23.8 on the same stories. Given the opening "One day", greedy decoding begins the story below. The rest of the continuation repeats the tree and the hole.

> One day, a little girl named Lily went to the park with her mom. She saw a big tree with a hole in it. She wanted to see what was inside. Her mom said, "Lily, be careful. The hole is very deep. It can hurt you."

### Instruction tuning on TinyStories-Instruct

[TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) pairs short stories with requests. Each request can combine a word list, a sentence to include anywhere in the story, a plot summary and features such as dialogue, foreshadowing or a twist. The notebook keeps 60,000 complete pairs from the training file. It uses the reserved story IDs and skips pairs that exceed 512 tokens or whose reference story fails its word or sentence check.

Each pair becomes a short chat. The user message contains the request and the assistant message contains the story. The loss counts the story tokens and the end token. A copy of the base model trains for 3,500 updates at half the pretraining learning rate.

### Evaluation on held-out prompts

The notebook keeps 400 pairs from the TinyStories-Instruct validation file and checks their story IDs against both training stages and the base validation set. To see what instruction tuning changed, both checkpoints answer the same requests. They use greedy decoding, choosing the most likely token at each step until they reach the end token or the context limit. The notebook generates answers for 64 requests with word lists and 64 with specified sentences. It counts exact matches and reports average word coverage. It also scores response loss on all 400 held-out pairs and measures TinyStories perplexity for both checkpoints.

Requests that specify a plot or story features are evaluated by reading the responses. A short rubric asks whether the story follows the summary's main events and outcome, and where the story shows each requested feature. The notebook prints responses from both checkpoints for those reviews.

Fine-tuning took about 9 minutes on the M4 Max. These results compare the checkpoints from the longer run.

| Held-out metric | Base | Tuned |
| --- | ---: | ---: |
| TinyStories perplexity, 1,500 stories | 2.86 | 2.94 |
| TinyStories-Instruct response loss, 400 pairs | 1.194 nats per token | 0.855 nats per token |
| All requested words used, out of 64 | 1 | 11 |
| Average share of requested words used | 12.5% | 54.7% |
| Requested sentence appears, out of 64 | 0 | 6 |

The shorter run scored 1.267 nats per token on the same 400 responses. It fine-tuned on 10,000 pairs for 400 updates and matched 0 of 24 complete word lists and 2 of 24 exact sentences. The longer run uses a larger evaluation sample. On a request about Timmy helping the police find a lost puppy, the tuned story follows that main event. For the prompt about Lily meeting a friendly ghost, the tuned model includes the ghost and uses “lend.” It misses “mighty,” and the ghost never turns into a superhero. The example shows how word coverage can improve while a requested plot twist is still missed.

### Optional experiments

The last stage of the notebook holds smaller experiments that the later chapters discuss. These include preference pairs with DPO, group-relative policy updates, a reward head, distillation, beam search, int8 quantization, LoRA, retrieval and a bounded tool loop. The experiments that train or sample extra models run only when `COOK_ML_LABS=1` is set. The others define small, checked functions and run in seconds.

## Running it

<p id="hardware-requirement">The <a href="/tutorials/llm-fundamentals/#hardware-requirement">hardware requirements</a> are on the LLM Fundamentals overview.</p>

The commands below install the dependencies and start Jupyter on Apple Silicon. On NVIDIA, use the [PyTorch installer](https://pytorch.org/get-started/locally/) to select a CUDA build for your system. The other Python dependencies are the same.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install torch jupyterlab "datasets==3.6.0" "huggingface_hub<1"
COOK_ML_PROFILE=story jupyter lab llm-fundamentals.ipynb
```

The `story` profile picks CUDA when it is available and Apple MPS otherwise. `COOK_ML_DEVICE=cuda` or `COOK_ML_DEVICE=mps` chooses one explicitly. Checkpoints go to `runs/llm-fundamentals/`, one for the base model and one for the fine-tuned copy. `COOK_ML_RESUME=1` resumes a run whose settings and data match the saved checkpoint. `COOK_ML_LABS=1` turns on the optional experiments. Running all cells from the top executes the stages in the order shown below.

## The notebook

<!-- notebook -->
