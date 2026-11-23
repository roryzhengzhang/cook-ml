---
title: "Tasks and Verification"
description: "Generate tasks and verify the final state before using scores for training."
sidebar:
  order: 2
  label: "Tasks and Verification"
lastUpdated: 2026-11-18
---

The store can grade a sentence such as “Your exchange is complete” by opening the order record. In one attempt, O-17 now holds a medium shirt and stock has fallen to 0. In another, the same sentence follows a timeout and the order still holds a small. A word match accepts both responses. The store now has enough evidence to tell them apart.

A **verifier** is a program that checks whether an attempt fulfilled a task. It needs an exact starting case and a description of the acceptable outcomes. For a task with more than one successful path, that description can name properties of the resulting state. It can also require an accurate customer-facing result when the store should take no action.

![Exchange outcome verification](/images/rl-for-llm-agents/17b-exchange-outcome-verification.png)

The task fixture pins the case the agent will face. Outcome checks then turn the state change into a score. Once a verifier can reject plausible mistakes, the task family and split design decide what that score can tell us about learning. The chapter closes by admitting a new task using tests for each of these pieces.

## Task fixtures

A fixture should be sufficient to start the task again and judge its result. For O-17 this means a request to move from S to M, a paid order that permits the exchange, one medium in stock, and a timeout injected on the first stock call. The verifier uses an **expected outcome** to name the state and customer-facing result an attempt can earn credit for.

Some of the fixture is public to the agent. It gets the customer's request, the tool descriptions, and the limits of the interaction. Other fields are only for the environment and evaluator. The one available unit becomes visible when the tool returns it. The expected outcome stays with the verifier throughout the episode. Leaking that outcome into the prompt would give away the distinction this task is meant to test.

A **precondition** is a fact that must hold before a valid action. The exchange needs a matching paid order, permission under the policy, and an available target size. A **postcondition** is a fact that must hold afterwards. A successful submission here leaves the order at M and makes exactly one corresponding record. Describing both clarifies which facts the tool enforces and which effects the grader checks.

The same request can start with a stock count of 0. The target state then keeps O-17 at S with zero exchange records. The customer-facing result should report that the requested size is unavailable. An ineligible order gives a separate expected outcome. It also preserves the original order but calls for the correct eligibility explanation. One set of tools can therefore support counterfactual cases where a change to one fact reverses the right decision.

A test can describe acceptable outcomes without fixing the exact action order. A scripted policy may check eligibility before it reads inventory. Another policy may fetch both in either order and still obey all write preconditions. A trace rule belongs in the fixture when the order of calls itself matters, for example if the store must confirm eligibility before disclosing private order details.

A **fixture version** identifies the precise rules, data, and fault behavior that started a run. If the simulator changes the eligibility window or introduces a retry limit, the version changes with it. Results from the older fixture still describe the behavior of an agent tested under the earlier policy.

A **fault schedule** specifies which operations fail and when. Including the planned fault in the fixture lets all attempts in a group face the same initial problem. It also lets the evaluator distinguish an injected event from a real worker failure that needs to be recorded as infrastructure trouble.

## Verifiers

The verifier reads the initial state, final state, and relevant events from the attempt. The **state delta** is the change between those snapshots. A good O-17 delta changes the order size to M, decreases stock M by 1, and records one exchange for the requested order. A claim of success requires a matching change in the environment.

| Case | Final O-17 size | Medium stock | Exchange records | Expected result |
|---|---|---:|---:|---|
| Eligible with one M | M | 0 | 1 | Exchange confirmed |
| Eligible with zero M | S | 0 | 0 | Medium unavailable |
| Ineligible with one M | S | 1 | 0 | Ineligibility explained |

A **side effect** is another change made while completing the request. An attempt might exchange O-17 correctly and also edit an unrelated order. Checking expected changes alone would miss the extra write. An evaluator can compare a protected set of records or inspect an immutable transaction log for forbidden changes. In this store, the expected effects include any legitimate changes that the simulator makes during the exchange.

The write is recorded as an event tied to one order and one target size. It is possible for the order to be size M because it began that way or because another operation changed it. The event supplies the missing evidence that this attempt performed the exchange. Matching event fields also catches a transaction linked to a different order.

A **structured outcome** records the meaning of the agent's final response in known fields. It can include an outcome category and the order identifier. A deterministic verifier can then check that the category matches the store state. Readability and tone can be scored separately when they matter to an application.

A structured outcome should distinguish an accurate refusal from an infrastructure error. When the fixture demands an eligible and in-stock exchange, a timeout followed by a stockout claim is a task failure. When the fixture has zero stock, reporting unavailability with no write is itself a success. This keeps the reward from pushing the agent to change an order every time.

In practice, terminal state checks are useful whenever success is observable in software. Sierra's [τ-bench](https://github.com/sierra-research/tau2-bench) runs agents through customer-service tasks with policies and tools. The store project uses that same idea at a scale where every field can be inspected by hand. A grader for tone or ambiguous policy explanations would require human labels or a judge calibrated against them.

## Reward design

After an attempt, the verifier can say whether the order and inventory ended up where they should. The trainer needs that judgment as a number. The **reward function** sets the score from the evidence recorded in the episode. The first experiment gives a **binary reward** of 1 when the verifier passes and 0 otherwise. This defines a narrow objective that can be checked directly. It also keeps failure codes available in the trace, so a reward of 0 can be diagnosed.

In four illustrative attempts, one agent retries, completes the exchange, and gets 1. Another calls the timeout a stockout. A third attempts a disallowed write. The last keeps asking for inventory until its budget expires. Those three get 0, so the group's mean reward is one success divided by four attempts, or 0.25. The numbers are properties of the constructed traces. They do not estimate a model's success rate.

The score supplies **sparse feedback** because it arrives only at the end. A model that stalls after several good read calls receives the same zero as a model that made up the order identifier at the start. That makes the signal easy to audit and leaves the question of which earlier action mattered to the training algorithm.

A **failure reason** identifies the check that failed. A reward of 0 paired with an invalid tool schema suggests a different fix from a reward of 0 paired with an incorrect exchange. Trace counts by failure reason also reveal shifts that a mean reward might hide. A reward can stay binary as verifier diagnostics become richer.

A **shaping reward** gives intermediate credit or a cost for something expected to help the overall goal. A small tool-call cost can discourage waste, but its magnitude changes behavior. With success worth 1 and each tool call costing 0.05, the five-call recovery earns 0.75. A three-call failure earns −0.15. At a cost of 0.60 per call, the successful route earns −2.00 and the failure earns −1.80. The new reward prefers the failed run because it is shorter.

![Reward and tool cost](/images/rl-for-llm-agents/17b-reward-and-tool-cost.png)

The two routes tie when the extra cost of two calls equals the one point for success. That happens at a per-call cost of 0.50. With any larger cost, the reward ranks these two particular routes in the wrong order. This little boundary check is useful before training with any mixture of task reward and efficiency penalties.

**Reward hacking** happens when a policy earns a high score through behavior that misses the real objective. Paying the model for each check might lead it to repeat the same stock lookup. Giving credit for a short response might encourage it to abandon orders after the first transient error. Report full task success on a separate evaluation even when the optimization reward includes shaping terms.

## Verifier tests

A verifier is executable code and needs its own tests before model runs can depend on its output. A **positive test** runs a known valid trace and expects a passing score. A **negative test** gives the verifier a plausible error and checks that it fails for the right reason. Both are needed because a broken grader can award 1 to every model or award 0 to every policy.

### Practice: One valid exchange and two false completions

The examples all use the request for a medium from O-17. They preserve the difference between the words the customer sees and the record the verifier can inspect.

<details class="cm-steps not-content" data-walkthrough="one-valid-exchange-and-two-false-completions">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Exchange verification</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">The verifier checks order size, stock, and the exchange record together.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The valid trace checks policy, gets a timeout, retries stock, and submits one exchange. The verifier sees S become M, stock go from 1 to 0, and the exchange count rise from 0 to 1. The confirmed outcome agrees with the store.</div><img src="/images/steps/17b-exchange-verifier/step-01.png" alt="Step 1 of 3, Exchange verification" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>A false completion uses the same confirmation after the timeout and performs no submission. The order stays S, stock stays 1, and the exchange count stays 0. The state check rejects the trace.</div><img src="/images/steps/17b-exchange-verifier/step-02.png" alt="Step 2 of 3, Exchange verification" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>Another false completion reaches M but records two exchange events in a deliberately corrupted fixture. A size-only check would accept it. The count and affected-record checks reject it.</div><img src="/images/steps/17b-exchange-verifier/step-03.png" alt="Step 3 of 3, Exchange verification" loading="lazy" decoding="async"></li>
</ol>
</details>

The same suite can exercise check failures one at a time. Setting only the final size to M should still fail if the event count is zero. Setting the event count to 1 with stock left at 1 should still fail. These cases point to the missing evidence by holding the rest of the state fixed.

A **mutation test** deliberately weakens the code or edits a trace to see whether the tests catch it. Deleting the exchange-count check should make the duplicate write case fail its test. Changing an ineligible outcome to a successful one should do the same. If no test reacts to a mutation, the suite has a blind spot in that area.

The store guards writes during execution. The verifier checks the resulting behavior afterwards. The runtime guard protects the invariant that stock stays nonnegative. The negative tests still exercise that invariant against intentionally corrupted data, which is a way to check the grader without asking the real store to perform the forbidden write.

A verifier can also behave incorrectly on edge cases in its own input. An absent final result, a malformed event, or an unknown order should produce a recorded failure reason. An exception from the verifier is a measurement problem. The training loop can track it separately from failures that the verifier scored successfully.

## Task families

Training cases come from an explicit set of **task families**, which are groups of cases that exercise the same underlying decision. A direct eligible exchange, an exchange that requires a retry, and an exchange prohibited by policy can each have variants with different order identifiers, sizes, or wording. The family keeps track of the decision structure that those surface changes leave intact.

A **task generator** produces fixtures by combining a request template, store records, and a fault schedule. The notebook fixes the target size at M and varies stock counts, timeouts, and eligibility. An extension with two eligible sizes to request, two possible starting stock counts, and either zero or one timeout would give eight combinations before varying eligibility. A rule such as excluding exchanges to the same size can reduce that count. Logging the generator seed and version establishes exactly which combinations were included.

![Task combinations](/images/rl-for-llm-agents/17b-task-combinations.png)

The generator needs a **feasibility check** to confirm that the task can be completed under its constraints. A retry case whose minimum valid path takes five calls cannot be evaluated as an ordinary success task with a four-call tool budget. Running the scripted policy or a known valid trace on the fixture tests both reachability and the integrity of the reward calculation.

A balanced training batch is easy to describe once families have labels. With 20 tasks in a batch, reserving 5 for retries guarantees retry tasks make up 25% of the batch. Success on that deliberately balanced mix measures learning on it. A separate report weighted for a store's real task mix would answer a different operational question.

A **task mix** is the share of attempts drawn from each family. Training on 90 easy eligible exchanges and 10 retry cases can conceal a poor retry policy. Even zero successes on retries would allow aggregate performance of 90% when the ordinary cases all pass. Reporting counts and success by family preserves the lesson the retry cases were designed to teach.

Difficulty can also shift as the agent changes. On the first checkpoint, a valid write may be unusual and justify more direct examples. On a later checkpoint that solves nearly every direct exchange, retry and refusal cases may provide more variation. The validation distribution stays fixed while experiments change the training mix.

The September 2026 [Hugging Face RL Environments release](https://huggingface.co/blog/rl-environments) includes NVIDIA NeMo Gym's Workplace Assistant, a multi-step environment with business tasks, databases, and tools. A production task pool can be much larger than this tutorial's store. It still benefits from versioned fixtures and a way to check whether each exercise is feasible.

## Dataset splits

The training split is where the policy learns from demonstrations and generated attempts. The **validation split** is used to choose settings such as the task mix or checkpoint. The **test split** stays out of those choices until a fixed evaluation. Using one test set to choose among models gradually turns it into validation data.

The [CPU lab’s split](/tutorials/rl-for-llm-agents/notebook/#store-task-splits) trains on a timeout with stock available and on zero stock with no timeout. One held-out fixture combines a timeout with zero stock. The agent has to retry first and then report unavailability. The split check compares the conditions in the fixtures, so giving a training fixture a new order ID cannot make it held out.

A **variant** changes wording or incidental details without creating a new decision structure. Suppose one retry case has four paraphrases. One paraphrase is in the test split. If the other three were independently assigned to test with probability 0.2 and train otherwise, all three stay out of training with probability $0.2^3=0.008$. At least one leaks into training with probability $1-0.008=0.992$. That is a 99.2% chance of a near-duplicate in train for the chosen test paraphrase.

![Task family split](/images/rl-for-llm-agents/17b-task-family-split.png)

A **grouped split** assigns all variants of one case family to the same partition. This measures transfer across unfamiliar cases. A **compositional holdout** goes further by withholding a combination of facts the agent can encounter separately during training, such as a stockout after a timeout. The two tests answer different questions about how well the policy handles the fixture generator.

Real trace-derived tasks can share more than wording. Two support requests might point to the same order or come from a shared incident. Those relationships should seed the grouping before anonymization removes the clues. Record the split assignment and audit for overlapping order templates or reward labels after transformation.

Evaluation also varies across repeated attempts. Five retries of one held-out task estimate behavior for that task. They are five trajectories from one fixture. Five distinct held-out fixtures test broader coverage. Reporting both counts preserves that distinction when comparing agents.

An exact hash can catch identical task records. A task-family identifier catches generated duplicates whose request wording or order ID changes. A frozen evaluation manifest pins which tasks and versions were used, so a later score can be attributed to the checkpoint or to a documented change in the benchmark.

Different split designs can coexist in the same report. A conventional grouped validation set measures progress without repeatedly tuning on the final test. A versioned challenge set can isolate one new rule or fault combination. Results retain their split names so an improvement on one does not erase a regression on another.

## Task admission

It helps to run dataset checks before generating model actions. A failed exact-hash or case-family audit can invalidate the eventual comparison no matter how many GPU hours go into it. A failed scripted trace can reveal an impossible task before the learner spends its chance budget trying to solve it.

A task admission check keeps a poorly specified exercise from consuming rollout and GPU time. It starts with a scenario that matters to the application, demonstrates that it can be solved, and records the ways a plausible attempt can fail.

1. The fixture starts from a versioned store and defines the visible request, fault schedule, and stopping budget.
2. A known valid trace meets the preconditions and reaches the expected outcome. The verifier checks the allowed changes and correct final response.
3. Negative traces cover false confirmations, wrong orders, duplicate events, and inappropriate refusals. The recorded failure reason identifies what needs to be fixed.
4. The split manifest groups related variants and assigns the task to one partition. Validation tasks guide training decisions, and the locked test set measures the resulting policy.
5. The report records the task family, verifier version, and relevant score. These become the fixed reference for comparing the base model and its post-trained checkpoints in [Base Model and Baselines](/tutorials/rl-for-llm-agents/base-model-and-baselines/).

---

**Related**: [Agent and Environment](/tutorials/rl-for-llm-agents/agent-and-environment/) · [Model Baselines](/tutorials/rl-for-llm-agents/base-model-and-baselines/) · [Evaluation and Safety](/tutorials/llm-fundamentals/evaluation-reliability-and-safety/)
