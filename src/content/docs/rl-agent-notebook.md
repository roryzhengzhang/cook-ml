---
title: "Train an order exchange agent with RL"
description: "Build a store simulator, verify exchanges, and train a policy from its own multi-turn attempts."
slug: tutorials/rl-for-llm-agents/notebook
---

Order O-17 holds a small shirt, and the customer wants a medium. An inventory timeout turns a straightforward exchange into a sequence of decisions. The project notebook builds the store and checks the outcome of each attempt. It then trains a small policy to retry and complete the exchange.

<div class="cm-notebook-actions cm-notebook-page-actions">
  <a class="cm-notebook-download" href="/notebooks/rl-agent.ipynb" download>Download .ipynb</a>
</div>

## The CPU project

The notebook runs on CPU with Python and PyTorch. It uses a small policy that learns the probability of retrying, committing an exchange, and escalating a reported defect. The environment and verifier are the ones used throughout the series. The policy is small enough to inspect each probability before and after an update.

The code uses PyTorch tensors and automatic differentiation. All tasks are generated locally, and the notebook runs without a model download, external API, or account. Python 3.12 and PyTorch 2.15 were used for the reference execution.

```sh
python -m pip install torch jupyterlab
jupyter lab rl-agent.ipynb
```

### Held-out results

One good exchange shows that the agent can solve the task. Repeating it from the same starting conditions shows how often it succeeds. The evaluation uses four saved task setups, called fixtures, and runs each one in 32 groups of eight attempts. Every checkpoint is evaluated on the same task fixtures and per-attempt seeds. The CPU reference run used training seed 17.

| CPU policy | Successful attempts | Groups with eight successes |
| --- | ---: | ---: |
| Supervised warm start | 605 / 1,024 | 34 / 128 |
| Rejection fine-tuning | 738 / 1,024 | 42 / 128 |
| GRPO | 770 / 1,024 | 45 / 128 |

After the policy change, the existing checkpoint scored 75.2% on the original suite and 74.0% on the new suite. Continuing on new tasks alone gave 75.2% and 94.5%. The 50/50 task mix gave 81.2% and 90.0%. Each action context in this small policy has its own learned logit. The release exercise shows how to measure the tradeoff on both suites. A language model with shared weights can also forget an older behavior, so the GPU experiment needs its own comparison.

### Reasoning before a retry

When the stock lookup times out, a second CPU policy first chooses whether to treat the stock count as unknown or assume it is zero. The retry probability depends on that choice. GRPO trains both decisions from the final exchange reward. In the seeded run, the probability of treating the timeout as unknown rose from 50.0% to 59.3%, and this policy's held-out success rose from 58.8% to 67.1%. The two thoughts are fixed choices in this exercise. The GPU variant generates its own text. [Reasoning before tool calls](/tutorials/rl-for-llm-agents/multi-turn-rollouts/#reasoning-before-tool-calls)

## The language-model pilot

The GPU recipe targets a single 48 GB L40S with a 1.5B dense base model. In the proposed colocated run, SGLang generates attempts, Megatron updates the model, and the next batch begins after weight synchronization. A smaller configuration on a 24 GB RTX 4090 would be tested separately. The hardware figures in the training systems chapter are worked estimates until the model run has measured memory, throughput, and held-out success. [slime placement guide](https://thudm.github.io/slime/get_started/experiment-guide.html#choose-gpu-ownership-then-scheduling)

### Pilot files

[Store adapter](/notebooks/rl-agent/slime_store.py) · [Training launcher](/notebooks/rl-agent/run_slime.sh) · [Setup and commands](/notebooks/rl-agent/README.md)

The adapter extracts the store and verifier from the notebook, then generates deterministic task splits. The launcher covers supervised warm start, multi-turn GRPO, evaluation, and a second GRPO round on either new tasks or a mixture of old and new tasks. The optional thinking variant adds a bounded reasoning span before each command and records how many tokens it uses. We tested task generation and rollouts on CPU with a simulated sampler. The full Megatron and SGLang run still needs hardware validation.

## The notebook

<!-- notebook -->
