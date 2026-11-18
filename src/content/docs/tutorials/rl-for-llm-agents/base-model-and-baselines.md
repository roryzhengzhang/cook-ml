---
title: "Model Baselines"
description: "Compare prompt-only behavior, supervised fine-tuning, and training on verified successes."
sidebar:
  order: 3
  label: "Model Baselines"
lastUpdated: 2026-11-18
---

A language model can describe how a return works and still leave O-17 untouched. To complete the exchange it has to produce calls the store can execute, use the results, and stop after giving the customer an accurate status. The scripted agent gives us a completed trace. A pretrained model gives us a way to see which parts of that behavior it already produces.

A **baseline** is a reference result obtained under stated conditions. The prompt-only agent provides a baseline for model behavior before weight updates. Supervised fine-tuning adds examples of how to act across several turns. Rejection fine-tuning collects the agent's own attempts and turns the verified successes into another training set. Comparing these stages gives the later reinforcement learning experiment a way to tell whether its improvement came from the update or from seeing better examples.

![Baseline training stages](/images/rl-for-llm-agents/17c-baseline-training-stages.png)

The first sections work through scripted, prompted, and supervised versions of the same agent. The serialized trace then shows exactly which tokens the supervised update learns to produce. Fresh rollouts reveal how often the model gets things right and how often equally plausible attempts fail. Those frequencies decide what rejection fine-tuning can collect and whether GRPO will have a useful comparison to make.

## Checkpoint baselines

To measure what training changes, we need to be able to run the model as it was before the update. A **checkpoint** saves the weights at a particular stage so we can load that version again. The **base model** here is a language model whose weights already reflect pretraining. It starts with a general ability to continue text. The project supplies task instructions and a way to attempt typed tool calls, then measures what the unchanged checkpoint does with that context. [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/) explains how this stage relates to instruction-tuned models.

A **prompt-only baseline** gives the model a description of the store, its tool schemas, and the same user request the later checkpoints receive. A documented set of demonstrations can be included in the prompt as **few-shot examples**. Changing those examples can change the result without changing a single weight. The prompt version therefore belongs in the evaluation record beside the checkpoint.

For O-17, even a policy that knows to retry might output the string “check stock” as a normal message. The harness can only execute a call that satisfies the action schema. An evaluation report can distinguish invalid formatting, valid calls leading to a failed task, and actual task successes. The first two categories point to different changes in the prompt and training data.

A **scripted baseline** uses explicit rules for the current task. For an eligible order with one remaining medium, it looks up the order, checks policy, reads stock, retries the injected timeout, and submits the exchange. It is a check on what the task permits. Its behavior on rule combinations outside its design depends on the coverage of its rules.

Baseline conditions matter for the store as much as they do for the model. All policy runs should use the same simulator version, parser and safety rules, tool and token budgets, and evaluation fixtures. A baseline can also analyze parser failures by class without silently repairing the outputs of only one policy.

The proposed full training pilot starts near the one-billion-parameter scale. [slime's 1.5B model configuration](https://github.com/THUDM/slime/blob/main/scripts/models/qwen2.5-1.5B.sh) provides a concrete candidate in the Qwen2.5-1.5B base family. An initial evaluation has to establish whether that checkpoint can learn the tool format and generate a mix of passing and failing attempts on suitable tasks.

## Expert demonstrations

The store’s scripted policy already knows to retry after a stock timeout. Its trace shows the model which calls to make and what the agent sees between calls. A sequence that shows how to complete the task is an **expert demonstration**. The successful O-17 trace contains five tool calls. It records two different responses from the same stock tool, so the retry makes sense in context.

Demonstrations cover decisions where the next action depends on the previous observation. An episode that reports one medium can proceed to submission. An episode with zero mediums should report unavailability without changing the order. An episode with a timeout can retry within its budget. Demonstrations that always proceed to submission would teach an action sequence that treats these distinct observations as interchangeable.

A deterministic script can generate different valid histories by varying which independent read it performs first. That adds useful diversity only when the observed state and subsequent action agree. An expert trace that changes the order of calls by reordering its stored messages after execution can produce impossible dependencies.

The examples also teach the boundaries of a task. Some should show that an ineligible order ends with a clear refusal. A malformed order identifier may require a clarification or a lookup error. An example where an exchange succeeds can vary in the wording of its final reply while preserving the structured result and state change.

A **counterfactual pair** changes one relevant fact while holding the rest of the case fixed. Such pairs give the model reasons to pay attention to tool results. A pair of fixtures can have nearly identical histories through the first four calls. If the fourth result changes from stock 1 to stock 0, the next correct assistant action changes from submitting an exchange to finishing with an unavailable outcome. The paired traces make that dependence visible during inspection.

Demonstrations still come from the training split. [Dataset splits](/tutorials/rl-for-llm-agents/tasks-and-verification/#dataset-splits) explains how paraphrases of the same scenario can leak across partitions. A validation case with a new rule combination is more useful when it checks how the model adapts than when it checks how well the model repeats a familiar trace.

## Tool serialization

The model consumes tokens, so the messages and tool records must be turned into a token sequence. A **chat template** specifies how role boundaries, tool calls, arguments, and results appear in that sequence. The chosen template must agree with the serving parser about how a valid call begins and ends. It replaces the store's Python objects with a text format the model can read and emit.

| Role | Example content in the O-17 episode | Where it comes from |
|---|---|---|
| User | Exchange order O-17 for a medium | Customer request |
| Assistant | A call to `check_stock` for M | Policy output |
| Tool | Timeout | Store result |
| Assistant | A new stock call for M | Policy output |
| Tool | Medium stock is 1 | Store result |
| Assistant | A call to `submit_exchange` for O-17 and M | Policy output |

The rollout template and training template need the same opening cues. A **prefix** is everything the model sees before it produces the next token. If the rollout prefix stops just after a tool result, the next control tokens should put the model at the start of the assistant turn. The round-trip test can confirm that supplying those tokens while replaying the trace reproduces the stored action boundary.

A **role marker** identifies the source of text. A tool role is inserted by the harness. It is distinct from the assistant continuing its own text to sound like a tool. Keeping role boundaries explicit makes it possible to inspect what the agent produced and what was supplied by the environment after each action.

**Tokenization** maps the serialized text into token identifiers, as [Tokenization](/tutorials/llm-fundamentals/tokenization-and-embeddings/) shows. A call name can occupy multiple tokens. The illustrative target names later in this chapter stand for selected positions. The actual mask and parser test read token positions from the model's template and tokenizer.

In practice, the first check after selecting a base checkpoint is a round trip. A known chat trace is serialized and tokenized. The assistant call is decoded and parsed by the same code used at rollout time. In the pilot, a **command grammar** lists the complete set of accepted command lines. Action and observation markers distinguish each assistant command from the simulator response in the context. The expanded chat roles in the table show the same distinction for a native tool template. Restoring the same tool name and arguments tests the boundary where a correct training example could otherwise turn into an invalid action.

## Supervised fine-tuning

Training tries to increase the model's probability of producing expert actions. **Supervised fine-tuning** (SFT) fills in known assistant turns and trains the model to predict them. The model sees the earlier turns from the demonstration when making each prediction. This is **teacher forcing** from [Language Modeling](/tutorials/llm-fundamentals/language-modeling-and-next-token-prediction/). In the replay of O-17, the timeout and the successful retry result are already present at their proper positions.

The objective is the average **negative log likelihood** across selected targets. A target with low probability contributes more because the negative logarithm is larger. Suppose the model gives three selected tokens probabilities of 0.50, 0.25, and 0.80. Their individual losses are about 0.693, 1.386, and 0.223. Their sum is 2.303 and their mean is about 0.768.

$$
\mathcal{L}=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_t
$$

Here $t$ indexes target positions. The probability $p_t$ is the model's probability of the example's target token at position $t$ given all earlier tokens. The **loss mask** $m_t$ equals 1 at positions being learned and 0 elsewhere. The denominator counts selected positions. In the three-token example, a mask of 1 at all three positions divides the sum by 3.

**Loss aggregation** determines how individual token losses count toward the training objective. The denominator changes what receives weight. Averaging over selected token positions gives a long demonstration more opportunity to contribute than a short one. Averaging the mean loss of each trace weights each trace equally. The tutorial uses a token average for the worked SFT calculation and records aggregation choices in framework experiments.

Several correct actions can fit a history. SFT raises the probability of the one shown in an expert demonstration. A set of diverse successful traces can teach several valid responses and expose more contexts. It can still leave gaps where the model's own decisions lead to histories that never appear in those traces.

## Training masks

A training mask marks the parts of a trace the agent should learn to produce. If the harness gave the agent “stock is 1,” that result should help the model choose the next action. The model should have no loss at positions where the harness supplied the result. User messages and **padding** follow the same rule. Padding fills out shorter sequences when a batch needs tensors of equal length. The mask still includes assistant-generated tool arguments and final responses where the dataset intends to teach them. A **stop token** marks the end of a generation, and the mask can include it to teach the intended end boundary.

![Assistant loss mask](/images/rl-for-llm-agents/17c-assistant-loss-mask.png)

### Practice: Masked loss on a tool trace

The toy sequence assigns conditional probabilities to five target positions. Two contain text supplied by the user and the store, and three belong to assistant turns.

<details class="cm-steps not-content" data-walkthrough="masked-loss-on-a-tool-trace">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Masked supervised loss</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">An illustrative assistant training mask with five target positions.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The user and the tool positions have probabilities 0.10 and 0.05 in the example. The three assistant positions have probabilities 0.50, 0.25, and 0.80.</div><img src="/images/steps/17c-masked-loss/step-01.png" alt="Step 1 of 3, Masked supervised loss" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The intended mask counts the three assistant positions. The product of their probabilities is 0.10, so the total negative log likelihood is about 2.303 and the mean is about 0.768.</div><img src="/images/steps/17c-masked-loss/step-02.png" alt="Step 2 of 3, Masked supervised loss" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>A mask that also counts the user and tool positions multiplies by another 0.005. The resulting total is about 7.601 and the mean over all five targets is about 1.520. That loss penalizes the model for predictions the agent will never need to make.</div><img src="/images/steps/17c-masked-loss/step-03.png" alt="Step 3 of 3, Masked supervised loss" loading="lazy" decoding="async"></li>
</ol>
</details>

A token after a tool result should still depend on the preceding result when predicting the next action. The difference is in which positions have a target contribution and which positions supply context. When training uses next-token prediction, the target mask is shifted consistently with the targets. Inspecting the first assistant token after a tool result is a direct check for an off-by-one error.

An explicit mask count is valuable when migrating between tokenizers. The underlying tool call may be the same while its punctuation or arguments split differently. Test the actual encoded sequence and confirm the first generated token after each observation receives the intended loss.

In the full project, [slime's SFT example](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3-4B-base-sft.sh) runs the training step on stored conversations. The reference store experiment would use train-only mode for the supervised stages. A one-batch check can compare the framework's active token count with the notebook's hand-marked trace before committing a full run.

The thinking variant adds short rationales to the same expert traces. After the stock tool times out, the demonstration says that the store returned no stock count, then calls `check_stock` again. The supervised loss learns the rationale and the command. The labels and the tool result stay in context with mask 0. This gives the model examples of how to use evidence before it starts collecting its own [reasoning rollouts](/tutorials/rl-for-llm-agents/multi-turn-rollouts/#reasoning-before-tool-calls).

## Rejection fine-tuning

A model that already solves some tasks can provide its own additional demonstrations. In **rejection fine-tuning** (RFT), the policy runs several attempts, the verifier keeps those that pass, and another supervised update imitates the successful trajectories. **Expert iteration** repeats that cycle with the resulting policy. CS336's [Alignment and Reasoning RL assignment](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf) uses this comparison on verifiable reasoning tasks.

Consider four attempts of the timeout fixture. One retries and gets an exchange confirmed. Two report a stockout from the timeout and one runs out of turns. Rejection fine-tuning adds the one verified success. A group with four failed attempts supplies no training target. That is a direct way to see why the procedure can improve a promising policy yet struggle to start from one that almost never succeeds.

The retained traces change the effective data mix. Four easy tasks with four successful attempts each produce 16 examples. Four hard tasks with one success each produce 4 examples. The original fixture mix was half easy and half hard. The kept set is 80% easy unless the sampling or weighting strategy corrects for that skew.

![Rejection data composition](/images/rl-for-llm-agents/17c-rejection-data-composition.png)

Every retained trace should be counted once before training. If identical successful attempts dominate the kept set, reporting the number of unique task IDs and unique trajectories helps explain what the model saw. The accepted subset can also be stratified by task family to show whether filtering erased examples of hard cases.

RFT makes a strong baseline because it relies on the same task, samples, and verifier as GRPO. It uses supervised training on successful traces. GRPO can use the contrast between success and failure in the same group. To isolate the difference, start both runs at the same checkpoint and report their rollout counts, sampled token counts, and optimizer budgets.

## Rollout diversity

A stronger warm start often increases the fraction of well-formed actions. How much that helps an RL run depends on whether the policy can produce varied outcomes on the same tasks. If a rollout group always gets the same score, comparing its rewards provides no direction for raising one attempt over another. A count of **mixed-outcome groups** tells the pilot whether this occurs frequently.

Suppose a given task has a single-attempt success probability of 0.25 under a fixed policy. Four independent attempts are all successes with probability $0.25^4$. They are all failures with probability $0.75^4$. The remaining probability is $1-0.25^4-0.75^4$, about 0.680. About 68% of groups would contain both outcomes under this simplified model.

![Mixed rollout groups](/images/rl-for-llm-agents/17c-mixed-rollout-groups.png)

The curve peaks when success and failure are equally likely. For four attempts, a single-task success probability of 0.5 gives a mixed-group probability of $1-2\times 0.5^4=0.875$. Raising completion from 50% to 95% is good for the agent, but leaves only about 18.5% of groups mixed on that task. Training may need more challenging fixtures as easier ones saturate.

The success rate can vary by fixture and attempts can become correlated if rollout workers accidentally share state or reuse the same samples. The formula explains why the pilot should log mixed groups directly. A single average success rate can hide that most easy tasks are always solved and most hard tasks always fail.

A pilot can check whether invalid calls dominate failures and whether the hard cases ever pass. Additional demonstrations can help when the model struggles with the action format. Task difficulty can be adjusted on the validation split when no run discovers the successful path. The held-out report should continue to include difficult cases so the pilot's curriculum decisions are visible.

A **pilot run** exercises the training pipeline for a small budget before the main experiment. It can start with a short context and a small rollout batch. The eventual memory requirement comes from the measured high-water mark for the pinned software image and model. A GPU with more memory can hold a larger batch or context, and a modest profile can validate the core learning loop when it stays inside its limits.

## Baseline comparison

A **run manifest** ties each row back to exactly one run. It can include the source checkpoint, training-set hash, run seed, model revision, template revision, and stop configuration. That record lets a new evaluator load both agents and replay the same O-17 fixtures without guessing which behavior came from the model.

A baseline comparison starts with a pinned environment and a common evaluation harness. These checkpoints come from different training procedures, so the report records the intervention behind each result.

1. The prompt-only run establishes what the base checkpoint does on the fixed tasks with the documented prompt and sampling settings.
2. The scripted policy verifies feasibility on the cases it was built to solve. Its trace also identifies where its coverage ends.
3. The SFT run starts from the same base checkpoint and uses verified training demonstrations. The report separates valid tool calls from completed tasks.
4. The rejection fine-tuning run starts from that SFT checkpoint and reports the number of rollouts sampled, the number kept, and their breakdown by task family.
5. The pilot records group outcomes on the same validation fixtures. The next chapter can now explain a policy update using attempts with both successful and failed outcomes.

---

**Related**: [Agent and Environment](/tutorials/rl-for-llm-agents/agent-and-environment/) · [Tasks and Verification](/tutorials/rl-for-llm-agents/tasks-and-verification/) · [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/)
