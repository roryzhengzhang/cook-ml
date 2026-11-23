---
title: "Multi-Turn Rollouts"
description: "Preserve the agent’s reasoning, actions, and tool observations across the full episode."
sidebar:
  order: 6
  label: "Multi-Turn Rollouts"
lastUpdated: 2026-11-18
---

The inventory service returns a timeout on order O-17. The agent has enough information to know the exchange is allowed and too little to know whether it can be filled. It can call the service again. The second request returns stock 1, and a valid submission moves the order from size S to M. The grader has to remember the order state at the beginning and check it again at the end.

A replay of this exchange tells an engineer which call succeeded and which one failed. The trainer requires another kind of record. It needs the exact tokens the model sampled and the observations each sampled action had available.

![Stock retry](/images/rl-for-llm-agents/17f-stock-retry.png)

A multi-turn rollout ties three things together. The agent keeps making decisions after the first tool call. The store keeps its state across those calls. The trainer eventually has to recover the exact model actions from a transcript that includes both choices and observations. The chapter traces those pieces through one exchange and its failed counterpart, then uses their differences to explain which feedback can guide the next update.

## The agent harness

A model can propose a tool call by generating text. An **agent harness** is the program that turns the proposal into an action and hands the result back as context. It parses the generated call, checks the fields against the tool schema, invokes the corresponding function, and gives the policy another turn. The same model can behave differently under a harness that formats tool errors differently or allows fewer calls before stopping.

The store handles four actions. A call to `lookup_order` finds the purchased shirt. `check_policy` reports whether it can be exchanged. `check_stock` looks for the requested size, and `submit_exchange` attempts the change. The environment owns the permission and inventory checks at write time. The policy proposes what to do, even when a persuasive sentence in its context might suggest skipping those checks.

The language-model pilot uses a small wire format. A model emits one named action per line. The harness supplies the order and size from the task fixture. That design keeps the first training run focused on deciding when to look up, retry, submit or finish. Using model-sampled tool arguments would add a second problem for an expanded lab.

A **turn** here means one model generation with the context available at that moment. One turn can end in a tool call, after which the service returns an observation. It can also end with a final structured decision. The harness tracks the remaining turn budget so an invalid tool call, an unresolved error, or a long response cannot keep an episode running indefinitely.

Keeping the loop small helps isolate failures. If the policy emits a stock call with the wrong size field, schema validation can return a structured error with enough detail for a repair. A successful retry after that error is a different skill from recovery after an inventory timeout. They deserve separate counters even when both lead to reward 1.

The harness also decides how to treat tool output. A string returned by the stock service is input from that service. Its structured fields and error types carry the facts about the store. The action validator and verifier can consult trusted state directly. That gives the training implementation a route to diagnose a transcript whose final claim and actual outcome disagree.

The order of observations defines what information was available for a decision. A final transcript with stock 1 can conceal the fact that the first answer gave no count at all. The harness records each tool result in order so the trainer can reuse the prefix that existed before the retry.

### Practice: One inventory recovery

The following playback locates the point where the agent has to choose between treating the error as a fact and trying the read again.

<details class="cm-steps not-content" data-walkthrough="one-inventory-recovery">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Inventory recovery</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">A failed read can be retried while the order and inventory stay intact.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The order lookup reports size S and the policy lookup confirms eligibility for an exchange.</div><img src="/images/steps/17f-inventory-recovery/step-01.png" alt="Step 1 of 4, Inventory recovery" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The first stock lookup returns a timeout. The order is still size S and the attempt has learned nothing about the quantity of medium shirts.</div><img src="/images/steps/17f-inventory-recovery/step-02.png" alt="Step 2 of 4, Inventory recovery" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>The retry returns stock 1. The remaining budget permits the submission.</div><img src="/images/steps/17f-inventory-recovery/step-03.png" alt="Step 3 of 4, Inventory recovery" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 4</span>A valid submission records one exchange, leaves the order at size M, and reduces the remaining medium stock from 1 to 0.</div><img src="/images/steps/17f-inventory-recovery/step-04.png" alt="Step 4 of 4, Inventory recovery" loading="lazy" decoding="async"></li>
</ol>
</details>

Retrying here targets a read. A write can take effect before the caller hears back. A real write interface can require an **idempotency key**, an identifier the tool uses to recognize repeat submissions of the same operation. The store verifier still checks that the order changed exactly once. The trace needs to distinguish a retry after an uncertain write from a second intended write.

## Isolation and reproducibility

A GRPO group should compare attempts facing the same task. If four attempts share the store, the first successful exchange might take the only medium from inventory. An identical decision in each later attempt then receives a different score because another rollout already changed the world. Each attempt therefore gets a fresh store from the same fixture, and any scripted fault for that fixture is scoped to its own attempt.

A fixture version and a task identifier are enough to recreate a deterministic store. A per-attempt identifier keeps the logs and state separate. If the environment has additional randomness, the runner also records the seed or the emitted events. Preserving the conditions of the task still allows the policy to draw different actions. Two trajectories can face exactly the same first-call timeout and choose different ways forward.

Task isolation can be tested with a pair of attempts. Two independent copies of O-17 should each begin with stock 1. A valid exchange in copy A should take its count to zero, while copy B should still report 1 after its own single injected timeout. Group rewards are meaningful only after this property holds under concurrent execution as well as serial execution.

A **replay** reconstructs an episode from saved actions and observations. The store can replay the actions against a fresh fixture to check that the same state transition and reward follow. It can also replay a captured rollout into the trainer to test masks and recorded probabilities. The latter requires the token data from the original run. A text transcript can be sufficient for finding a bad tool argument and still leave a training bug invisible.

Recording an environment revision and a verifier revision makes later replays easier to interpret. A new policy rule can change an expected outcome while the exact same action trace still replays correctly. Evaluating an older checkpoint on the new rule measures migration. Reproducing the older run requires the older verifier and initial fixture.

Separating the store instance from the model server also makes the learning loop cheaper to debug. The same fixture can be played by a scripted agent on the CPU. Small replay cases then establish the tool outcomes before the GPU pilot varies how the actions are sampled.

## Trace and token alignment

A retry looks sensible after a timeout and wasteful after a stock count has already arrived. A **trajectory record** connects each decision to the observation the model had when it made that decision. For each turn it stores the model input IDs, the sampled output IDs, a loss mask, and the behavior log probabilities when the sampler supplies them. It also carries the matching tool call and result, the model version, and the reason the episode ended. Store safety checks and the eventual score attach to the episode as a whole.

In the [CPU rollout trace](/tutorials/rl-for-llm-agents/notebook/#multiturn-trace), the same stock tool can return a timeout on one turn and a count on the next. The policy samples whether to retry after the first observation and whether to submit after confirmed stock. Each record keeps the context, the chosen action and its behavior log probability so the update can score each choice under the conditions it saw.

![Trainable spans](/images/rl-for-llm-agents/17f-trainable-spans.png)

The observation that medium stock equals 1 should help the model choose to submit. It is a conditioning input for the next decision. Its position should have no direct action loss. Using the same loss on everything that appeared in the transcript would ask the agent to predict the stock service's output as though it had chosen that output.

A trace also keeps the identities of attempts, turns, and tool calls. The first failed inventory lookup and a successful retry have the same tool name. Distinct call identifiers let the trace viewer line each result up with its request. Training needs this mapping when a tool finishes after another one starts or when two answers share the same visible phrasing.

### Practice: A masked exchange prefix

This fragment assigns small token counts for arithmetic. A real chat template can use many more tokens to render each call and observation.

<details class="cm-steps not-content" data-walkthrough="a-masked-exchange-prefix">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Token-level loss mask</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">A tool result is visible in context and has mask value zero.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>An assistant emits a two-token call to check stock. Both positions have mask value 1 and a behavior log probability.</div><img src="/images/steps/17f-masked-exchange-prefix/step-01.png" alt="Step 1 of 3, Token-level loss mask" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The tool adds three tokens for the timeout. All three positions have mask value 0.</div><img src="/images/steps/17f-masked-exchange-prefix/step-02.png" alt="Step 2 of 3, Token-level loss mask" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>The assistant samples a two-token retry. The two positions have mask value 1. The fragment's mask is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow><annotation encoding="application/x-tex">[1,1,0,0,0,1,1]</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">[</span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">0</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">0</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">0</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">1</span><span class="mclose">]</span></span></span></span>,</span> so four sampled token positions contribute to the action loss.</div><img src="/images/steps/17f-masked-exchange-prefix/step-03.png" alt="Step 3 of 3, Token-level loss mask" loading="lazy" decoding="async"></li>
</ol>
</details>

A rendered transcript can omit delimiters, insert spacing, or combine pieces that were encoded separately. The trainer needs the actual token IDs and prefix used at sampling time for each chosen action. In an append-only episode one token stream can represent the exchange. If a framework rewrites context between turns, one record per distinct sampled prefix can preserve alignment. The shared rollout identifier keeps the pieces of one attempt together for reward and loss reduction. [slime customization guide](https://thudm.github.io/slime/get_started/customization.html#returning-multiple-training-samples-for-one-prompt)

A prefix stored at decision time can be compared with the reassembled transcript in tests. Exact equality is a strong check for an append-only harness. A context compactor changes what the policy sees, so the trace can preserve each relevant prefix with its own mask. Counting each generated action once avoids giving earlier turns extra credit simply because their tokens appear in several later contexts.

A separate observability trace can be readable without carrying all of the data needed for an update. The project can display tool name, latency, outcome, policy version and task identifier in [the evaluation workflow](/tutorials/rl-for-llm-agents/continual-learning-and-release/). The saved training trace then preserves the exact input IDs, masks and behavior log probabilities. Both records can share identifiers so an unexpected policy update can be traced back to the episode.

One source of bad token records is a retry that sends the same model request twice. The harness can assign one durable turn identifier and record the attempt that actually supplied the action. The harness can then attach a completed tool result to that exact turn. This preserves the cause of a state change when retries appear in the log.

### Reasoning before tool calls

The stock tool returns a timeout on O-17. A policy that goes straight to an action can call `check_stock` again. A reasoning policy first generates a short thought such as “The lookup timed out. I still need a stock count.” It then generates the command with those extra tokens in its context. This is [test-time compute](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/#reasoning-models-and-test-time-compute) applied to a decision inside a tool-using episode. The extra text gives the model space to work through the evidence.

The pilot gives a turn two output spans. The first begins after a `Thought:` label and has up to 32 generated tokens. The next begins after an `Action:` label and has up to 24. The harness only parses the action span, so mentioning a command in a thought cannot change the order. The token trace stores the original sampled IDs and log probabilities for both spans. The supplied labels and tool observations have mask 0, just as they do in the worked example above.

The verifier checks the order and inventory after the episode. If a successful retry gets a positive group advantage, that advantage applies to the sampled thought and command tokens. A persuasive rationale earns nothing by itself. DeepSeek-R1 training also illustrates how a reward on the outcome can influence the intermediate text that came before it. In an agent, the outcome also depends on what its tools did.

The [CPU notebook](/tutorials/rl-for-llm-agents/notebook/#reasoning-before-a-retry) reduces the thought to two choices. One treats the timeout as unknown stock. The other assumes there is no stock. The choice changes the retry probability, so the notebook can show how a final exchange reward changes both decisions. The GPU pilot replaces those fixed choices with generated text. Its comparison uses the same simulator and verifier, with [matched limits](/tutorials/rl-for-llm-agents/evaluation-and-ablations/#matched-budgets) and a count of the extra tokens used.

## Terminal rewards and credit assignment

The store verifier can tell whether the state is correct when an episode ends. It can give a binary **terminal reward**, a score assigned after the last step. Using that score makes it straightforward to test the objective. The verifier compares the final order, exchange records, inventory, and structured decision to the expected consequences of the original request.

The four GRPO attempts can include two valid recoveries, one premature decline and one timeout loop. The successful attempts get positive advantages when grouped with the failures. The sampled tokens from a successful episode receive a common weight, including the important retry and routine choices made later. **Credit assignment** is the problem of using that result to decide how to update the earlier choices. The outcome says which sequence did better. It leaves the importance of individual choices uncertain.

A more targeted exercise can compare terminal reward with a shaping term for resolving an ambiguous result and proceeding to a valid decision. The term is attractive because its evidence can be checked earlier. It can also change preferences. If every successful lookup earned 0.1, ten repeated lookups would earn as much as an exchange worth 1. A bounded bonus tied to a verified milestone prevents this exact accumulation, and separate evaluation still has to check whether the signal encouraged the intended behavior.

![Outcome and credit](/images/rl-for-llm-agents/17f-outcome-and-credit.png)

An experiment can keep the same tasks, model, group sizes, and compute budget while changing the signal. The report needs held-out success, unnecessary calls, and incorrect mutations along with the optimized reward. Both versions can use the same grader for release evaluation, which keeps a shaped reward increase from being mistaken for a task improvement.

A terminal reward can leave many attempts with the same outcome early in training. If all four fail and produce similar invalid calls, a shaped stock-check score may still reward irrelevant progress. A warm start can make the desirable responses reachable in the first place. Then a comparison can test whether the added feedback improves recovery once the model knows how to call the tools.

A longer episode can contain two independently understandable mistakes. The agent might retry stock correctly and later choose the wrong order identifier for submission. Outcome-only training gives both actions in that trajectory the same negative advantage. A per-step signal can give the successful retry a different weight. It requires valid evidence at the retry boundary and more bookkeeping when the signal is propagated.

Choosing a per-step signal also requires deciding which timescale it describes. A stock read might succeed quickly, and the inventory can still change before a later write. The write service checks its current state before mutating the order. An intermediate stock milestone can teach the action sequence while the final verifier catches an invalid submission.

<details class="cm-collapse">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><h3 class="cm-collapse-title" id="credit-assignment-methods">Credit assignment methods</h3></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

Suppose O-17 succeeds after the agent writes “stock is still unknown,” retries the lookup, and submits the exchange. The same thought and retry can appear in a failed attempt that reports completion without submitting the exchange. The retry can also reveal that stock is zero and lead to a correct refusal. A final score mixes decisions, what the tool reveals, and what the agent does with that information. When successes are rare or many turns intervene, the same early decision can appear in both high- and low-scoring trajectories. It takes many samples for its average effect to emerge.

**Episode-level advantages.** In [the GRPO update used here](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/#group-advantages), a group with rewards $[1,0,1,0]$ gives the successful attempts advantages of about $+1$ and the failures about $-1$ with population standard deviation. Those signs apply to the sampled thought and retry in each attempt, even when the retry itself was sensible in both. Grouping attempts on the same fixture controls differences in starting difficulty. The advantage still has the resolution of a whole episode. A longer thought also adds more trainable token positions, which is why the [loss reduction](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/#masks-and-loss-reduction) matters.

**A critic and generalized advantage estimation.** A critic predicts the remaining reward from each observed prefix. Suppose it estimates a 0.30 chance of success just before the retry and 0.80 after stock is confirmed. With a discount of 1 and zero immediate reward, the one-step advantage estimate is $0+0.80-0.30=+0.50$. If the next action finishes without submitting and ends in failure, its estimate is $0+0-0.80=-0.80$. On this two-step suffix, generalized advantage estimation with $\lambda=0.5$ gives the retry $0.50+0.5(-0.80)=+0.10$. With $\lambda=1$ it gets $-0.30$, the final reward minus the original prediction. Smaller values lean more on the critic. This can reduce variance, but a critic that rarely sees timeouts may give systematically poor estimates. The jump after the retry also includes what the tool revealed. [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

**Counterfactual rollouts.** Save the store state and model prefix immediately after the timeout. Force a retry in one branch and a decline in another, then let the same frozen policy finish each branch. Pair the attempts by simulator and model sampling seeds. If 14 of 20 retry continuations pass and 0 of 20 decline continuations pass, the estimated difference is 0.70 on this fixture under that continuation policy. Branching holds fixed the information available at the decision. It costs extra rollouts and the result depends on how well the continuation policy can use the new observation. The same experiment can vary a thought span before generating the action, though an unnatural replacement can change the prompt in ways that are hard to interpret. The simulator's [reset and replay](/tutorials/rl-for-llm-agents/agent-and-environment/#reset-and-replay) checks are what make these comparisons possible.

**Process supervision and reward shaping.** A step checker can verify that a timeout left the stock count unknown or that a write uses the order ID returned by lookup. Such labels can train a process reward model or help select traces for supervised training. A label that says “this step is valid” can still give limited guidance about whether it was necessary. Adding step bonuses directly to the reward also changes what the agent is paid to do. One way to preserve the original return ranking is **potential-based shaping**. Assign progress scores to states and add the discounted change in that score to each transition. In an undiscounted episode, a score of 0 before a valid stock check, 0.2 afterward, and 0 at every terminal state gives $+0.2$ on entry and $-0.2$ by the end. The additions cancel, so a failed exchange still has total return 0. With discount $\gamma$, the added term is $\gamma\Phi(s_{t+1})-\Phi(s_t)$, where $\Phi$ is the progress score and $s_t$ is the state before the action. Computing one GRPO advantage from the unchanged total would give the same update as before. To use the earlier feedback, the estimator needs the reward at each turn. [Process supervision](https://arxiv.org/abs/2305.20050) · [Potential-based shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

**Learned reward redistribution.** When useful milestones are hard to label, a method such as RUDDER learns to predict the final return from action histories. It uses changes in those predictions to move reward toward earlier moments that help predict the outcome, with the redistribution designed to preserve the total return. On O-17, the predictor might shift credit toward the first valid retry. It can also pick up a phrase that happens to correlate with success. A replay that changes the phrase and resamples the later actions can test that explanation. [RUDDER](https://arxiv.org/abs/1806.07857)

For the store project, the first comparison can branch at the timeout and at the later submission. That tells us which decision to target before trying a new training signal. The direct GRPO run provides the baseline. A shaped reward, a critic, or a process checker can then be compared on the same held-out tasks and tool budgets. The final verifier still measures whether the exchange was correct, and the [ablation](/tutorials/rl-for-llm-agents/evaluation-and-ablations/#baselines-and-ablations) can count extra reads and invalid writes to catch behavior the new signal accidentally encouraged.

</details>

## Episode endings and failed infrastructure

A timeout can be part of the task or a fault in the system running the experiment. O-17 contains a planned first-call timeout, so recovering from it is one of the agent's decisions. Running out of turns after making repeated invalid calls is an agent outcome too. A GPU worker disappearing before the stock service responds tells the experiment much less about the policy.

A **termination** happens when the task reaches one of its defined endings. A correct exchange and a valid ineligible response are both endings. A **truncation** happens when an external limit cuts an episode short, such as a turn ceiling or time limit. For this project, a policy that reaches the declared turn ceiling without completing the task receives the configured failure outcome. An unrelated worker failure is recorded separately and retried or excluded under a stated policy.

![Episode endings](/images/rl-for-llm-agents/17f-episode-endings.png)

This distinction matters even when every outcome becomes a scalar. Automatically scoring every failed worker as zero can reward short trajectories when only long ones are interrupted by infrastructure. Quietly retrying until a particular task succeeds can bias results in the opposite direction. A fixed retry policy, an explicit terminal reason, and counts by reason make those data choices visible.

The plan can also distinguish a cut-off generation from a structured refusal. A refusal on eligible stock earns failure because it ends the task without the required exchange. If the output token limit cuts off a tool call midway, the harness records why parsing failed. Keeping both in the evaluation makes it possible to see whether a shorter token cap created the apparent regression.

A rollout needs a record even when all of its model turns are unusable. The trainer can count and exclude runs missing token data without hiding rate spikes from operations. A different category covers complete model generations that make invalid actions. Those outcomes can still be evidence about the model.

## The slime rollout boundary

The handwritten loop shows what each part of training needs from a rollout. The GPU run passes the same exchange tasks to slime. SGLang serves the sampling policy, the custom generator runs the store's tool loop, and the verifier scores the final state. Megatron receives action tokens and their training weights after generation finishes.

Slime's `--custom-generate-function-path` hook can own the sequence of model calls and tools for one rollout. Its `--custom-rm-path` hook can feed a programmatic verifier's score to the training pipeline. The trace adapter carries the tokens, loss mask, and sampled log probabilities into the training sample and records the episode's group and rollout identity. That identity matters when the same episode needs several trainable segments. Slime groups sibling segments by `rollout_id` so they represent one outcome. A generated prefix shared by two segments should contribute loss once. [slime agentic RL guide](https://thudm.github.io/slime/get_started/agent.html) · [slime customization guide](https://thudm.github.io/slime/get_started/customization.html)

In a small synchronous run, the sampler uses one policy version for a complete exchange attempt. That gives the trainer a clean baseline for checking behavior probabilities. The scaling chapter can later overlap updates and generation and examine the policy-version metadata already recorded by the same trace. More concurrency is useful once it is possible to explain which weights chose each action.

The hardware plan starts with one colocated GPU. SGLang and Megatron use the card in turns, while several isolated store sessions can be active during generation. The first sizing check measures memory and time for collection and update as separate phases. Each phase has different sources of memory pressure, including cached token state during generation and optimizer state or activations during training.

Before opening more workers, a run can record whether GPU time is spent sampling tokens or waiting for tools. As an illustration, a ten-second collection phase containing six seconds of waiting cannot be reduced below that wait by faster token generation alone. Logging two numbers gives the systems lab a reason to change scheduling rather than the model.

A useful integration check runs only the rollout first. The trace should show a successful recovery plus at least one failed path with the correct reward. A train-only replay can then confirm that the learner scores the exact recorded action tokens and that tool output positions contribute no direct loss. This localizes failures before an end-to-end run has to reveal them through a learning curve.

## From traces to experiments

A trace can show the first wrong decision long before an average reward can. The same evidence supports very different fixes. Repeated malformed calls suggest inspecting the token template and warm-start examples. If a valid call changes the wrong state, check what the tool does and what the verifier accepts. Correctly parsed calls that give up after a transient error provide a useful task for measuring the effect of RL.

One report row can compare two checkpoints on the exact O-17 fixture with an identical turn budget. It records the policy decision after the timeout, the eventual order state, and the count of write attempts. The fixture can stay in a fixed diagnostic set so another run can check the failure without exposing evaluation tasks to the training sampler.

Once a failure family is repeatable, a small test can distinguish a fix from a policy change. A harness fix should let the same captured action parse correctly. A training change should alter how often the model samples the desired action in the same context. This separation lets production changes be evaluated without losing the original diagnosis.

The project uses fixed, known examples to check the harness and mask, then keeps independent task families for evaluation. Evaluation reads the stored outcome, counts valid and invalid writes, and checks how often an agent recovers when the store has temporary faults. Comparing otherwise similar runs on the same held-out tasks leaves room to investigate why a change improved the score. A trace from a failed exchange can then become a new regression test without turning that exact held-out case into training data.

---

**Related**: [Agent and Environment](/tutorials/rl-for-llm-agents/agent-and-environment/) · [Tasks and Verification](/tutorials/rl-for-llm-agents/tasks-and-verification/) · [GRPO](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/)
