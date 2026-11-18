---
title: "Evaluation and Ablations"
description: "Measure reliability over repeated attempts and isolate the contribution of one training choice."
sidebar:
  order: 7
  label: "Evaluation and Ablations"
lastUpdated: 2026-11-18
---

The agent produces the sentence “Your exchange is complete.” The order still shows size S. In another attempt it changes the order to M twice, even though the store had one shirt left. Both replies sound like successes. A store evaluator can check the order, the inventory and the transaction log, and reach a different conclusion in each case.

A training curve measures what happened on the tasks the model practiced under the scoring rule it optimized. Here the question is whether a saved checkpoint gets the intended result on fresh tasks under the same limits as the checkpoint it might replace. That takes a fixture that can be loaded again, a clear definition of success and enough repeated runs to see whether the result is dependable. The counts in this chapter are worked examples.

![One request three outcomes](/images/rl-for-llm-agents/17g-one-request-three-outcomes.png)

The state check explains why a polished answer can conceal an incomplete exchange. The evaluation then separates tasks that guide development from tasks that support a final claim. Multiple attempts on each task expose intermittent failures. Multiple training runs expose a different source of uncertainty. With those measurements in place, the final sections compare training methods and work out how to test a specific explanation for a gain.

## Outcome and side effects

An exchange succeeds when it changes the right order to the right size, consumes the right stock and records one transaction. The user also needs a reply that accurately reflects what happened. A **task outcome** is a pass or fail based on those requirements. An **invariant** is a rule that must hold throughout the attempt, such as keeping inventory nonnegative and avoiding writes to a different order.

For O-17, the order starts at size S and one size M is available. The first call to check stock times out and the next reports one available. If the agent exchanges the order once, the final size is M, available M stock is zero and exactly one exchange is recorded. The evaluator can check those facts separately. The individual checks tell a reader which part failed even when the score is a single zero.

A correctly declined exchange has a different expected state. When nothing is in stock, the unchanged order is the right outcome if the agent reports the shortage. For an ineligible order, the success condition checks the policy reason and the absence of a write. A universal rule that rewards every mutation would encourage the wrong behavior for both cases.

A **side-effect violation** is an unwanted change caused by an attempt. A wrong-order exchange and an extra decrement both count. One way to report it is the fraction of attempts with one or more violations. A second count records the number of violations by type. A failure to finish and a mutation to the wrong order can both be task failures while carrying very different consequences for release.

### Practice: Three exchange outcomes

Three attempts start with fresh copies of O-17. The evaluator checks their state changes and reply.

1. The first attempt retries the stock check, exchanges once and confirms the result. Its success is 1 and its side-effect violation flag is 0.
2. The second attempt confirms without submitting an exchange. Its success is 0 and its violation flag is 0 because the state was untouched.
3. The third attempt submits twice and leaves stock at −1 in the hypothetical faulty service. Its success is 0 and its violation flag is 1. An invariant check catches the negative stock even though the order ends at M.

One of the three attempts succeeds and one violates a safety condition. Each fraction is one-third, but each describes a separate aspect of the model's behavior. The project simulator guards against underflow. Testing an intentionally faulty simulator state demonstrates that the evaluator still catches it.

In industry, Sierra's τ-bench checks the final database state against a goal and applies rules for the task. The project follows the same idea by scoring from trusted state and keeping the agent's reply available for inspection. A communication rubric can be useful for richer responses when its accuracy is checked against human examples. [τ-bench paper](https://arxiv.org/abs/2406.12045) · [Sierra benchmark repository](https://github.com/sierra-research/tau2-bench)

## Evaluation splits

A test measures generalization only if the examples are new in the way the claim requires. A new order ID measures performance on an unfamiliar instance of a familiar task. A new combination of eligibility and a tool fault tests whether the agent can combine behaviors it saw separately. Those are different questions, so the project labels both kinds.

The **training split** supplies tasks that can change the weights. The **validation split** supports choices such as the reward weights and which checkpoint to evaluate. A **locked test split** supports the final comparison. The test configuration is fixed before looking at the results. If an engineer changes the policy to address a locked-test failure and scores it again, that result describes a development cycle. A fresh locked suite is needed for a new confirmatory claim.

A **task family** groups requests that share a workflow, such as eligible exchanges with available stock. A **fixture fingerprint** is a stable identifier built from the task's underlying state, policy rules and fault schedule. It catches examples where changing the order ID or rephrasing the user message would otherwise hide duplicate evidence. The fingerprint definition follows the claim. A holdout for language variation can reuse a workflow. A holdout for new workflow combinations needs a split by the combination itself.

Suppose a generator emits 10 phrasings for each of 60 fixtures. Randomly splitting the 600 utterances can put alternate phrasings of a fixture into training and test. Splitting the underlying 60 fixtures into groups of 36, 12 and 12 assigns all 10 variants of each fixture together. A second set of deliberately held-out combinations can ask a stronger question about new conditions.

Versioning makes the split reproducible. An **evaluation manifest** specifies the policy version, fixture IDs, simulator build, scoring rules and run configuration. Pinning the validation and test manifests lets two checkpoints start each attempt in the same world, including O-17's tool timeout. The manifest records which variants share a parent fixture so later analysis can preserve their dependence.

![Fixture-level evaluation split](/images/rl-for-llm-agents/17g-fixture-level-evaluation-split.png)

A test suite can also contain **challenge cases** chosen to stress known weaknesses. O-17's injected timeout is useful even if the production rate of that fault is unknown. The report can give the natural mixture and the challenge slice their own results, since swapping their proportions changes the aggregate. A result on a fault-enriched suite should be read with that mixture in mind.

An **oracle** supplies the expected resolution from the fixture before model evaluation. On an eligible order with one item in stock, it can say that the size should change. When the policy rules conflict, the fixture needs a resolved rule or an explicit escalation outcome before its result should contribute to a checkpoint comparison.

## Repeated attempts

One successful attempt shows that the model can handle O-17 under that run's conditions. The next person sees one attempt too. Running the same task several times from the same snapshot shows whether the action is reliable when the model samples a different response.

The **single-attempt success rate** is the average number of passes per attempt. The **all-attempts success rate** asks whether every attempt in a set of $k$ runs succeeds. The **at-least-one success rate** asks whether any attempt in the set succeeds. In benchmark notation these are pass@1, pass$^k$ and pass@$k$ respectively. The retry measure assumes a workflow can recognize success and safely retry, including knowing whether any previous operation may have committed. Reading inventory again carries a different risk from resubmitting a write.

### Practice: Retry scores for one task

O-17 runs four times from the same starting fixture. Three attempts succeed. The fourth reports that the shirt is unavailable after the timeout. To work out both retry measures at two attempts, the example draws a pair uniformly from the four logged attempts.

<details class="cm-steps not-content" data-walkthrough="retry-scores-for-one-task">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Repeat one task</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">Four attempts on one task contain three successful exchanges.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>Four attempts contain three passes, so the single-attempt estimate is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3</mn><mi mathvariant="normal">/</mi><mn>4</mn><mo>=</mo><mn>0.75</mn></mrow><annotation encoding="application/x-tex">3/4=0.75</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mord">3/4</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.75</span></span></span></span>.</span></div><img src="/images/steps/17g-retry-scores-for-one-task/step-01.png" alt="Step 1 of 3, Repeat one task" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The four attempts make six unordered pairs. Three pair two successful attempts, so the all-attempts estimate at two is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3</mn><mi mathvariant="normal">/</mi><mn>6</mn><mo>=</mo><mn>0.50</mn></mrow><annotation encoding="application/x-tex">3/6=0.50</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mord">3/6</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.50</span></span></span></span>.</span></div><img src="/images/steps/17g-retry-scores-for-one-task/step-02.png" alt="Step 2 of 3, Repeat one task" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>Every pair includes at least one success because there is only one failed attempt. The at-least-one estimate at two is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>6</mn><mi mathvariant="normal">/</mi><mn>6</mn><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">6/6=1</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mord">6/6</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">1</span></span></span></span>.</span></div><img src="/images/steps/17g-retry-scores-for-one-task/step-03.png" alt="Step 3 of 3, Repeat one task" loading="lazy" decoding="async"></li>
</ol>
</details>

![Two-attempt reliability](/images/rl-for-llm-agents/17g-two-attempt-reliability.png)

With $n$ sampled attempts and $c$ successes for a task, the **combination count** $\binom{n}{k}$ is the number of distinct groups of size $k$. Here $k$ is no larger than $n$. Counting groups gives

$$
\text{pass}^{k}=\frac{\binom{c}{k}}{\binom{n}{k}},qquad
\text{pass@}k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

The first numerator counts groups containing only passes. The second numerator counts groups containing only failures, and subtracting that fraction gives the chance of at least one pass. A combination count is zero when the pool contains fewer items than the group size. Computing these scores within each task and then averaging gives tasks equal weight when the benchmark specifies that weighting.

The **trial protocol** specifies what is held fixed when the same task runs again. Keeping O-17's first-call timeout fixed isolates the agent's response to that fault. Drawing a new fault schedule each time measures reliability over both the model and the environment. The report states which question it asks.

The estimates summarize repeated draws under a defined reset and sampling protocol. Server failures shared by attempts, changing tool conditions or conversation history carried over from a prior attempt can change what the repeated trials mean. A client that tries twice against the same live store needs its own evaluation, including the side effects of both tries. [HumanEval's pass@$k$](https://arxiv.org/abs/2107.03374) · [τ-bench's pass$^k$](https://arxiv.org/abs/2406.12045)

## Sampling and training variation

Re-running O-17 four times with a saved checkpoint tests variation in its answers. Repeating the entire training experiment four times tests whether the improvement depends on a lucky training run. Early rollout samples can change which actions receive positive advantages, so training can diverge into different behaviors even from the same starting checkpoint.

An **evaluation seed** controls behavior sampling or an explicitly randomized fixture. A **training seed** controls choices such as task order and the model's rollouts during learning. The comparison reuses the same evaluation fixture list and sampling configuration for each checkpoint. It pairs the training runs by seed schedule where practical and reports the scores from every run alongside their mean.

Suppose the same test yields paired gains of 1, 7, −2 and 6 percentage points for four GRPO checkpoints over four rejection fine-tuning checkpoints. The average is $(1+7-2+6)/4=3$ points. The range spans −2 to 7 points. Reporting only the 7-point run would overstate the evidence. The four evaluation scores also share a task suite, so they should not be treated as four entirely independent draws of future customers.

A **paired comparison** matches outcomes by task before computing their difference. If one checkpoint succeeds on 82 of 100 fixtures and the other on 76, their six-point difference can arise from six newly fixed cases with no regressions, or from many gains and many losses. Counts of wins, losses and ties reveal that difference. When matched conditions allow, retaining the joint results also gives a tighter uncertainty calculation if easy and hard cases tend to affect both models in the same direction.

<details class="cm-collapse" id="task-level-uncertainty">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">Task-level uncertainty</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

A **paired bootstrap** estimates task sampling uncertainty by drawing a fresh list of task IDs with replacement and recomputing the difference between models. All attempts and both models for a selected task move together. If the evaluation has several paraphrases of one fixture, resample the underlying fixture instead so that a family of near duplicates does not turn into many independent observations. The resulting interval describes uncertainty for the benchmark sampling design. Variability across training runs is reported separately, with the individual runs visible.

</details>

![Four training seed differences](/images/rl-for-llm-agents/17g-four-training-seed-differences.png)

CS336's Spring 2026 alignment assignment uses four random seeds for its standard GRPO experiment and asks for individual variation alongside aggregate plots. The Cook ML report can use that design for the main comparison while the reader follows one reproducible run. [Stanford CS336 alignment assignment](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf)

## Matched budgets

More tool calls can increase success while slowing the customer down. More generated tokens can give a model longer to work through the store policy. A fair comparison fixes the tool and token ceilings, sampling settings, termination rule, fixture versions and weight checkpoint selection rule before running the models.

Budget accounting includes attempts that fail or hit a limit. If a model makes 400 tool calls across 100 attempts and completes 80 tasks, it uses $400/80=5$ calls per successful task. Another model makes 360 calls and completes 60 tasks, so it uses 6 calls per success. The second model uses fewer calls in total. The first spends fewer calls per completion. If a run succeeds zero times, cost per completion is undefined and the report retains both the zero success count and the total spend.

A **latency percentile** gives the time within which a chosen share of attempts ends. If 94 of 100 requests finish within two seconds and the next one finishes at eight seconds, the nearest-rank 95th percentile is eight seconds. This **tail latency** can reflect sandbox startup, tool retries or a long model response. Logging phase timings lets a reader identify the source. The report records target hardware and separates model latency from end-to-end latency.

Training budgets measure a different cost. Rejection fine-tuning and GRPO can receive the same starting checkpoint and number of generated tokens while using different amounts of optimizer time. The report counts sampling and training compute separately and states which budget was held fixed. A comparison at equal GPU-hours can answer a second question about which method makes better use of a compute allocation.

A [reasoning policy](/tutorials/rl-for-llm-agents/multi-turn-rollouts/#reasoning-before-tool-calls) spends some of its budget before it chooses a command. Compare it with the direct policy on the same tasks, including timeouts with stock and timeouts without stock. Count verified completions, invalid actions, budget exhaustion, generated tokens per completion, and end-to-end latency. Comparing two separately trained checkpoints measures the entire pipeline. To isolate the effect of writing a thought at answer time, use one checkpoint trained in both formats and toggle only the reasoning step.

## Baselines and ablations

A 75% exchange success rate means more once we know what the same tasks looked like before training. A **baseline** is the model or procedure we use as that point of comparison. The scripted agent checks that the expected outcomes are achievable. The prompted base model measures what is available before weight updates. The supervised fine-tuned checkpoint shows what imitation of expert demonstrations buys. Rejection fine-tuning adds learning from the model's own verified successes. GRPO adds an advantage signal computed within each rollout group.

An **ablation** changes one part of a system to test a particular explanation. When GRPO beats rejection fine-tuning, it may be tempting to credit the use of failed attempts. Those methods also differ in how training weights the successful examples and how quickly the policy changes. A controlled next experiment can keep the same GRPO implementation and change only the reward or the group size.

### Practice: Attempts or distinct tasks

A training batch can collect 128 attempts in two ways. Both runs start with the same supervised checkpoint, use the same task sampler and share the same tool and token ceilings.

1. The first arm samples 16 distinct task seeds and runs 8 attempts for each, using $16\times8=128$ attempts.
2. The second arm samples 32 distinct task seeds and runs 4 attempts for each, using $32\times4=128$ attempts.
3. The comparison reports the fraction of groups with both success and failure, the actual generated token counts and held-out success by family. Those quantities show whether extra attempts found more useful differences or whether the agent benefited more from seeing a broader set of situations.

A larger group makes it likelier that a moderately difficult task produces a mix of scores. A larger number of distinct tasks exposes training to more variants. If the two arms generate sharply different token totals, the same number of attempts can hide a substantial compute difference. A rerun with a generated-token budget makes that tradeoff explicit.

An ablation should also preserve enough context to diagnose a surprising result. If the smaller-group arm loses only on final-sale cases, the group-size change may have altered how many examples of that rule received a useful advantage. Slices with few cases make that conclusion tentative. Storing the group-level rewards lets a reader test this explanation against the runs.

## Checkpoint comparison report

A release report connects the winning and losing cases back to the traces. Suppose the candidate resolves more stock timeouts but starts submitting extra writes on an already exchanged order. The aggregate score can rise while a release gate fails. Counts by fault type explain what improved and what needs attention before another run.

The project report identifies each checkpoint, the paired test manifests, the budgets and the verifier version. It shows episode success, repeated-trial reliability, side-effect violations, cost per completion and variation across seeds. A small set of disagreement traces preserves the exact tool calls and ending state, which lets the next chapter locate the time spent in model generation, the simulator and training. [Training Systems](/tutorials/rl-for-llm-agents/scaling-the-training-loop/)

A **paired error report** can list a stable fixture ID and the first point where each candidate diverged, such as parsing a timeout as proof of zero stock. Totals reveal the frequency. Aligned traces point to the action, observation or scoring decision that accounts for the difference.

The model selection rule is set on validation results. The locked comparison then supports a bounded claim about the chosen checkpoint and test conditions. If the next policy changes which exchanges count as correct, the release comparison needs a new policy-aware test and a check that older skills still work. [Continual Learning](/tutorials/rl-for-llm-agents/continual-learning-and-release/)

---

**Related**: [Tasks and Verification](/tutorials/rl-for-llm-agents/tasks-and-verification/) · [Multi-Turn Rollouts](/tutorials/rl-for-llm-agents/multi-turn-rollouts/)
