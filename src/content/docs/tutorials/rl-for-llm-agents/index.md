---
title: "Training Tool-Using Agent With RL"
description: "A customer asks to exchange a shirt for a medium. The agent finds the order, checks the store policy, and asks the inventory tool for stock. The tool times out."
sidebar:
  order: 0
  label: "Overview"
lastUpdated: 2026-11-18
---

A customer asks to exchange a shirt for a medium. The agent finds the order, checks the store policy, and asks the inventory tool for stock. The tool times out. One attempt tells the customer that the shirt is out of stock. Another checks again and completes the exchange. The replies can sound equally plausible, and the order history tells them apart.

## Why RL Post-Training?

A pretrained language model can write a customer service reply. A prompt supplies the store policy and the available tools. [Supervised fine-tuning](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#supervised-fine-tuning) teaches the model to imitate successful examples, including the sequence of tool calls. The model may still encounter a combination of tool results it has never seen in a demonstration.

**Reinforcement learning** gives the agent practice on its own attempts. Each attempt runs against a fresh copy of the store. A verifier checks whether the agent handled the request correctly, including the resulting order and inventory state. The model update uses those scores to change the probability of its sampled actions. This makes it possible to improve a decision even when several different trajectories would satisfy the request.

The verifier may only report failure after the last turn. The decision that caused it could have happened much earlier, when the agent interpreted a timeout as an inventory result. Connecting the eventual outcome to the sampled actions is the **credit assignment** problem. A one-decision exercise makes the update small enough to calculate by hand. The multi-turn lab shows what the trainer can learn when a score covers an entire exchange.

Prompting, a change to the tool, or additional demonstrations can repair some failures. A write API also needs to enforce rules such as exchange eligibility every time it is called. RL becomes worth testing when the remaining decisions still fail often, the simulator can reproduce those failures, and a verifier can reliably grade the alternatives. The project compares reinforcement learning with rejection fine-tuning, which trains on the agent’s own verified successes.

A [reasoning model](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#reasoning-models-and-test-time-compute) can write intermediate tokens before it acts. In this project, those tokens might spell out why a timeout leaves stock unknown before the agent retries. The [reasoning lab](/tutorials/rl-for-llm-agents/multi-turn-rollouts/#reasoning-before-tool-calls) shows how to train on both the thought and the command using the same verified result.

## The Order Exchange Project

The same order anchors the notebook and the chapters. Order O-17 holds a small shirt. The customer wants a medium and exactly one is in stock. The first stock lookup times out. A bounded retry reveals the stock count, and the agent can submit exactly one exchange. Later episodes change the starting inventory, eligibility rules, and faults. The notebook shows why these changes alter rewards and what the trained policy can infer from the tool observations.

<aside class="cm-model-card" id="order-exchange-project">
<p class="cm-model-title"><span class="cm-model-label">Running example</span>Order exchange agent</p>

The project trains a policy to decide when to retry an inventory lookup, submit an exchange, or escalate a reported defect.

| Part | Project implementation |
| --- | --- |
| [Environment](/tutorials/rl-for-llm-agents/agent-and-environment/) | A resettable Python simulator with typed tools and injected faults |
| [Reward](/tutorials/rl-for-llm-agents/tasks-and-verification/) | A verifier for the final order, inventory, and resolution |
| [Training](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/) | Supervised warm start, rejection fine-tuning, and GRPO on a small CPU policy |
| [Reasoning](/tutorials/rl-for-llm-agents/multi-turn-rollouts/#reasoning-before-tool-calls) | A CPU exercise in reasoning before retry, with an optional free-form reasoning run in the GPU pilot |
| [GPU pilot](/tutorials/rl-for-llm-agents/scaling-the-training-loop/) | A 1.5B dense base model with slime, Megatron, and SGLang. One 48 GB L40S is the first target |
| [Release exercise](/tutorials/rl-for-llm-agents/continual-learning-and-release/) | A second training round for final-sale cases and evaluation on both policies |

The [project notebook](/tutorials/rl-for-llm-agents/notebook/) follows O-17 from a scripted replay to a learned retry decision. Its measured CPU comparison starts with a supervised warm start and tests two ways to improve it on the same held-out fixtures. The optional GPU pilot files show how those fixtures and the verifier carry over to a language-model run.

</aside>

## Chapters

| Chapter | Project milestone |
| --- | --- |
| [Agent and Environment](/tutorials/rl-for-llm-agents/agent-and-environment/) | Build a store that can execute actions, isolate attempts, and replay an episode. |
| [Tasks and Verification](/tutorials/rl-for-llm-agents/tasks-and-verification/) | Generate tasks and verify the final state before using scores for training. |
| [Model Baselines](/tutorials/rl-for-llm-agents/base-model-and-baselines/) | Compare prompt-only behavior, supervised fine-tuning, and training on verified successes. |
| [Policy Gradients](/tutorials/rl-for-llm-agents/policy-gradients/) | Work through one update from a sampled action to a changed action probability. |
| [GRPO](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/) | Sample several attempts at the same task and compute an update from their rewards. |
| [Multi-Turn Rollouts](/tutorials/rl-for-llm-agents/multi-turn-rollouts/) | Preserve the agent’s reasoning, actions, and tool observations across the full episode. |
| [Evaluation and Ablations](/tutorials/rl-for-llm-agents/evaluation-and-ablations/) | Measure reliability over repeated attempts and isolate the contribution of one training choice. |
| [Training Systems](/tutorials/rl-for-llm-agents/scaling-the-training-loop/) | Profile rollout generation and training on one GPU, then study how separate GPU pools overlap their work. |
| [Continual Learning](/tutorials/rl-for-llm-agents/continual-learning-and-release/) | Train a candidate for a new policy and check what it retains before promotion. |

## Related tutorials

[LLM Fundamentals](/tutorials/llm-fundamentals/) explains token probabilities, decoding, and post-training. [LLM Deep Dive](/tutorials/llm-deep-dive/) develops the systems ideas behind checkpointing, efficient adaptation, and distributed training.
