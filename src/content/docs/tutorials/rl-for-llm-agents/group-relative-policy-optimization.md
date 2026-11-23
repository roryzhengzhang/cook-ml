---
title: "GRPO"
description: "Sample several attempts at the same task and compute an update from their rewards."
sidebar:
  order: 5
  label: "GRPO"
lastUpdated: 2026-11-18
---

Four attempts at the same exchange earn $1$, $0$, $1$, and $0$. The simulator knows which attempts worked. To train the model, the four scores need to say how strongly to reinforce each attempt, which tokens to update, and how much to trust data sampled before the latest update.

All four attempts begin on an eligible order with one medium shirt in stock. They see the same injected timeout if they reach the stock check. Comparing these attempts avoids confusing a fortunate starting inventory with a good retry decision.

![Group rewards](/images/rl-for-llm-agents/17e-group-rewards.png)

Group Relative Policy Optimization uses attempts at the same task to set a baseline. The exchange example makes that calculation small enough to do by hand. It then follows the score to an individual sampled token, checks how clipping deals with a policy that has moved, and gives the frozen reference policy its own role. The chapter closes by turning the arithmetic into tests for the same loss used by the GPU trainer.

## Groups and task difficulty

A reward of 1 says an attempt passed the verifier. It provides more information about a task a policy solves sometimes than about a task the policy solves every time. On a hard exchange, an agent may have to recover from a transient timeout. An easy exchange may have stock available on the first call. Comparing scores across those tasks mixes the policy's choices with the effort demanded by the fixture.

A **rollout group** is a set of attempts at the same starting task. Each attempt uses its own environment initialized from the same snapshot. The task has the same order, inventory and fault schedule across attempts. The policy can choose different actions because it samples its output. At order O-17, the first stock call times out in every attempt that reaches it. A policy may retry, decline, or run out of turns. Valid recovery keeps the meaning of a reward of 1 consistent across attempts.

**Group Relative Policy Optimization** (GRPO) uses the group's observed rewards to estimate a baseline. The update can use this estimate without training a separate critic. That trades the cost of learning a value function for the cost of sampling several attempts per task. The original method introduced it for mathematical reasoning. Here the same comparison grades a stateful exchange. [DeepSeekMath paper](https://arxiv.org/html/2402.03300#S4.SS1)

A group can still contain several successful strategies. One agent retries inventory immediately. Another checks the policy again and then retries. If they satisfy the same verifier, their rewards agree. This is useful when several action paths are valid. It also means the verifier decides which distinctions the trainer can learn from in that group.

Grouping is part of the experiment design. An identifier for the frozen task snapshot keeps the data loader from treating two similar prompts with different states as one group. A separate rollout identifier distinguishes each attempt. Later, when the same attempt has several model calls, the training set can still reconstruct the original reward comparison.

The same method can use a reward for speed or cost, provided the number still reflects what the designer wants optimized. For the first exchange experiment, binary rewards keep the meaning simple. A positive score first requires the right store outcome. Duration and tool count are available as evaluation measurements.

## Group advantages

A group baseline starts with the average reward. If two out of four attempts receive 1, the average is 0.5. Subtracting it leaves successful attempts at $+0.5$ and failed ones at $-0.5$. Scaling those differences by the spread of the group gives a common unit across groups.

$$
\widehat A_i=\frac{R_i-\bar R}{s_R+\varepsilon}
$$

Here $i$ indexes one rollout, $R_i$ is its outcome reward, $\bar R$ is the group's mean reward, and $s_R$ is its standard deviation. The small positive $\varepsilon$ makes division safe. The estimate $\widehat A_i$ measures how far the rollout's outcome lies above or below the group average. The threshold $\varepsilon$ in this equation serves a numerical purpose. The clipping interval later in the chapter has its own parameter.

### Practice: Four exchange attempts

For this walkthrough, the denominator uses population standard deviation. The four squared deviations are each $0.25$, and their average is also $0.25$.

<details class="cm-steps not-content" data-walkthrough="four-exchange-attempts">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Group-normalized rewards</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">The population spread is 0.5 for rewards of 1, 0, 1 and 0.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The four outcome rewards are <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">]</mo></mrow><annotation encoding="application/x-tex">[1,0,1,0]</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">[</span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">0</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">0</span><span class="mclose">]</span></span></span></span>.</span> Adding them gives 2. Dividing by 4 gives mean <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5</mn></mrow><annotation encoding="application/x-tex">0.5</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.5</span></span></span></span>.</span></div><img src="/images/steps/17e-four-exchange-attempts/step-01.png" alt="Step 1 of 3, Group-normalized rewards" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The four deviations are <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false">[</mo><mo>+</mo><mn>0.5</mn><mo separator="true">,</mo><mo>−</mo><mn>0.5</mn><mo separator="true">,</mo><mo>+</mo><mn>0.5</mn><mo separator="true">,</mo><mo>−</mo><mn>0.5</mn><mo stretchy="false">]</mo></mrow><annotation encoding="application/x-tex">[+0.5,-0.5,+0.5,-0.5]</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">[</span><span class="mord">+</span><span class="mord">0.5</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">−</span><span class="mord">0.5</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">+</span><span class="mord">0.5</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">−</span><span class="mord">0.5</span><span class="mclose">]</span></span></span></span>.</span> Their squares have mean <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.25</mn></mrow><annotation encoding="application/x-tex">0.25</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.25</span></span></span></span>,</span> so the population standard deviation is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msqrt><mn>0.25</mn></msqrt><mo>=</mo><mn>0.5</mn></mrow><annotation encoding="application/x-tex">\sqrt{0.25}=0.5</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1.04em;vertical-align:-0.1328em;"></span><span class="mord sqrt"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height:0.9072em;"><span class="svg-align" style="top:-3em;"><span class="pstrut" style="height:3em;"></span><span class="mord" style="padding-left:0.833em;"><span class="mord">0.25</span></span></span><span style="top:-2.8672em;"><span class="pstrut" style="height:3em;"></span><span class="hide-tail" style="min-width:0.853em;height:1.08em;"><svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l0 -0
c5.3,-9.3,12,-14,20,-14
H400000v40H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M834 80h400000v40h-400000z"/></svg></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height:0.1328em;"><span></span></span></span></span></span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.5</span></span></span></span>.</span></div><img src="/images/steps/17e-four-exchange-attempts/step-02.png" alt="Step 2 of 3, Group-normalized rewards" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>The last step divides each deviation by <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5</mn></mrow><annotation encoding="application/x-tex">0.5</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.5</span></span></span></span> plus a small numerical constant. The group advantages are approximately <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false">[</mo><mo>+</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mo>+</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">]</mo></mrow><annotation encoding="application/x-tex">[+1,-1,+1,-1]</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">[</span><span class="mord">+</span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">−</span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">+</span><span class="mord">1</span><span class="mpunct">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord">−</span><span class="mord">1</span><span class="mclose">]</span></span></span></span>.</span></div><img src="/images/steps/17e-four-exchange-attempts/step-03.png" alt="Step 3 of 3, Group-normalized rewards" loading="lazy" decoding="async"></li>
</ol>
</details>

An implementation using **sample standard deviation** divides the sum of squared deviations by 3 for this group. That gives $\sqrt{1/3}\approx0.5774$ and advantages of about $\pm0.8660$ before the numerical constant. The slime experiment guide uses this sample convention in its default normalized example. Either convention can be used in a teaching calculation, but a parity test needs both implementations to agree on it. [slime GRPO walkthrough](https://thudm.github.io/slime/get_started/experiment-guide.html#choose-the-task-and-model)

![Centered rewards](/images/rl-for-llm-agents/17e-centered-rewards.png)

The spread changes the meaning of a one-point reward difference. A strongly mixed group and a group with one unusual outcome can produce different advantage magnitudes for their successful attempts. That is why the experiment records the reward vector and the reduction convention together. The final checkpoint's scores can then be traced back to a specific objective.

A reward difference can also stand out more when the group is almost uniform. In $[1,0,0,0]$ the success is $0.75$ above the mean, and each failure is $0.25$ below it. The population standard deviation is $\sqrt{3}/4$. The success receives about $+1.73$ while each failure receives about $-0.58$. This effect follows from comparing against the other attempts.

<details class="cm-collapse" id="a-sampled-baseline">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">A sampled baseline</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

The fixed baseline in the policy gradient chapter uses only information available before sampling the action. The group mean includes the current rollout's own reward. Subtracting that mean in the simple independent, identically distributed case scales the expected unnormalized update by $(G-1)/G$, where $G$ is the number of attempts. A leave-one-out baseline averages the other rewards for each rollout. Scaling by a group-dependent standard deviation can also change how tasks contribute. The existence of a mean-zero group alone is insufficient to establish the same expectation as the fixed-baseline estimator.

</details>

## Uniform rewards

A group of four successes has mean 1 and a group of four failures has mean 0. In both cases every difference from the group mean is zero. The resulting reward advantage is zero even though the two groups represent very different behaviors. A separate penalty or auxiliary loss can still contribute an update.

An available-stock fixture can become easy enough that all eight attempts in a CPU training group complete the exchange. Once that happens, the group contributes zero reward advantage. A harder training fixture that requires a retry can still produce a mix of scores and give the update something to compare.

A success rate of 90% shows how rapidly signal can disappear near either end. Suppose each attempt is sampled independently and the conditional success probability is $p=0.9$ on the same fixture. A group of size $G=4$ is all-success with probability $0.9^4=0.6561$ and all-failure with probability $0.1^4=0.0001$. The chance of a group with varied rewards is $1-0.6561-0.0001=0.3438$. Around two thirds of groups from that fixture would have zero outcome-driven advantage.

![Informative groups](/images/rl-for-llm-agents/17e-informative-groups.png)

Larger groups can provide more comparisons at one difficulty. They cost more sampled trajectories per fixture. For a fixed budget of 32 attempts, eight groups of four compare more distinct task instances than four groups of eight. A useful experiment compares those settings on the same distribution and records the proportion of varied groups alongside held-out task success.

Some implementations use **dynamic sampling**. For binary rewards, one version keeps only groups with both success and failure and samples further groups to fill the batch. This is a way to avoid spending optimizer steps on uniform groups. It changes the training mixture and can consume extra rollout compute. Tasks a model always fails may need a better starting policy, staged difficulty, or more informative feedback. Discarding their groups alone supplies no successful behavior to reinforce. [DAPO paper](https://arxiv.org/abs/2503.14476) · [slime customization guide](https://thudm.github.io/slime/get_started/customization.html)

With a larger group and the same distribution, each task costs more rollout tokens. Long tool sequences can also make nominally equal groups expensive in different ways. A useful comparison records attempted rollouts, completed rollouts and generated tokens before deciding whether the increase in signal was worth the time.

The store can report whether a group was uniformly successful, uniformly unsuccessful, or mixed. Combining the first two into one counter hides the corrective action. Uniform success can be a sign the fixtures have become too easy. Uniform failure can expose invalid tool calls, missing skills, a flawed fixture, or an overly strict grader. Looking at a few traces separates those cases better than changing the group size immediately.

The math assumes each rollout samples independently conditional on the task fixture. Reusing a cached answer as though it were a new draw can make the group look artificially uniform. Saving a random seed or identifying the generation request helps check whether four identical results came from chance or shared generation behavior.

## Token ratios and clipping

A response-level advantage can weight all sampled tokens in that response. If a rollout retries the stock lookup and completes the exchange, its sampled tool-call tokens receive a positive weight. The advantage says which way to push the sampled choices. The ratio checks how much the probability of each choice has already changed. The model scores each token given the prefix it actually saw, and clipping applies to that token’s ratio.

$$
\rho_{i,t}=\exp\big(\ell_{i,t}^{\text{new}}-\ell_{i,t}^{\text{beh}}\big)
$$

Here $i$ is the rollout, $t$ is the token position, $\ell_{i,t}^{\text{new}}$ is the current model's log probability of the sampled token, and $\ell_{i,t}^{\text{beh}}$ is its log probability under the sampling or **behavior policy**. The **learner policy** is the one currently being updated. Exponentiating the difference of natural log probabilities gives their ratio $\rho_{i,t}$.

If the token had probability $0.40$ when sampled and $0.50$ now, the log probability difference is $\ln(0.5)-\ln(0.4)=\ln(1.25)$. The ratio is $1.25$. With a positive advantage of $1$ and a clipping interval from $0.8$ to $1.2$, the clipped PPO-style contribution is $1.2$. If the ratio were $0.75$ with the same positive advantage, the contribution would be $0.75$. The actor remains accountable for that drop in probability.

Sampling configuration is part of the behavior policy. Temperature changes token probabilities. A restricted top-k or top-p sampler can even assign zero to previously possible tokens. The notebook can start with an unrestricted sampling distribution and check how its stored log probabilities line up with the learner before changing precision, sampling rules or kernels. That check catches alignment problems while both sides still use the same weights. [PPO clipping](https://arxiv.org/abs/1707.06347) · [slime experiment guide](https://thudm.github.io/slime/get_started/experiment-guide.html)

An implementation that assumes equal starting weights can check the log-probability difference before every optimizer step in a small replay. If the two systems disagree, the cause could be a different token boundary, the wrong model version, or a different sampling distribution. An off-policy correction is easier to reason about after the exercise controls those three sources of disagreement.

The clipped ratio attaches to the probability of a particular sampled token. Gradient norm clipping is a separate operation applied to the completed update. Interchanging the two ideas would hide whether a sample was clipped because its probability moved or because the whole gradient was large. The training report can log both so a run can be diagnosed after the fact.

For a ratio test, one saved log probability belongs to each selected position. The positions are sometimes stored in a shorter response array and mapped into a full token sequence. A parity case can start with a simple uninterrupted response before adding tool observations and padding.

## The reference policy

The policy that produced a batch tells the learner how to weight old samples. After several batches, we may also want to know how far the model has moved from the checkpoint that first learned to call the store tools. A **reference policy** is a fixed copy used for that comparison. The warm-start checkpoint is a reasonable reference when a run is meant to preserve the behavior learned in supervised fine-tuning. The behavior policy generally changes when new batches are collected.

A **Kullback–Leibler divergence** (KL) measures how different two probability distributions are. A penalty based on it can discourage the learner from drifting too far from the reference. The original GRPO objective includes a reference-based penalty. An experiment can set its coefficient to zero so a first run isolates the reward-driven update. The coefficient becomes a choice to test when there is a measured behavior worth preserving across updates. [DeepSeekMath objective](https://arxiv.org/html/2402.03300#S4.SS1.SSS1)

Consider one token from order O-17. Its behavior probability might be $0.40$, its learner probability $0.50$, and its reference probability $0.45$. The learner-to-behavior ratio $0.50/0.40=1.25$ determines the clipping calculation. The learner and reference probabilities enter the KL calculation. A fresh rollout can set the behavior probability to $0.50$ while the reference remains at $0.45$. In that case the importance ratio returns to $1$, yet a reference penalty can still act.

![Policy versions](/images/rl-for-llm-agents/17e-policy-versions.png)

A reference penalty allows a deliberately small update when the task rewards also favor a wider move. That trade can be tested by measuring both new exchange success and retention of familiar tool formats. A small penalty has no automatic interpretation in customer outcomes. The evaluation suite determines whether the lost improvement or retained behavior mattered.

The reference penalty can use the same selected token positions as the reward loss. Otherwise a long environment observation would change the size of the regularizer even though the policy never emitted those tokens. The batch needs one consistent definition of the positions it trains.

## Masks and loss reduction

A rollout contains tokens the model chose and tokens supplied by the environment. A **loss mask** is an indicator that selects positions with a direct objective term. On a printed exchange trace, the stock count of 1 is an input from the inventory service. It should influence the probability of the next sampled action. Its token should have a zero in the action loss mask.

The selected contributions also need averaging. A mean over trajectories makes each trajectory count equally after taking its token average. If a four-token response and an eight-token response both have a per-token contribution of $0.3$, the sequence means both read $0.3$. Averaging all tokens together gives each token the same weight, so the eight-token response supplies twice as many terms as the four-token response. A fixed denominator is another choice. It changes what happens when lengths shift through training. A comparison of reductions needs the same sampled traces, masks and objective terms. [Understanding R1-Zero-like training](https://arxiv.org/abs/2503.20783)

For example, two terms of $0.3$ from a short episode and six terms of $0.1$ from a long episode give episode means of $0.3$ and $0.1$. Their mean is $0.2$. Averaging all eight tokens gives $(0.6+0.6)/8=0.15$. Identical traces produce different scalar losses because the longer episode has greater weight in the token mean.

This choice is observable in a batch that contains both quick declines and longer successful exchanges. The baseline GRPO comparison can hold the reducer fixed while varying task and group composition. A later ablation can change only the reducer and check whether average length and success move together.

Response padding is another source of denominator errors. A batch with four selected tokens and four padding positions has four contributing terms. Dividing by the padded length of eight would halve their mean and make the update depend on how sequences happened to be batched.

A batch implementation should also handle empty or rejected attempts explicitly. If a timeout is external and the trace cannot be trusted, a run can retry or flag the entire group. If the policy closes the conversation before taking an action, its sampled ending can still be scored as part of an unsuccessful attempt. Those are different observations for the learner. Both should appear in operational counters.

## A reference update

The notebook's small policy can test the sign of the update, exact averaging and clipping against fixed probabilities. This part is deliberately independent of serving and long-running jobs. The procedure uses the same ordering as the main training loop.

<figure class="cm-algo not-content"><figcaption class="cm-algo-head"><span class="cm-algo-tag">Algorithm 1</span><span class="cm-algo-name">One GRPO update</span></figcaption><div class="cm-algo-io"><div class="cm-algo-io-row"><span class="cm-algo-io-label">Input</span><span class="cm-algo-io-body"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">One task group with sampled tokens, rewards, masks, and behavior log probabilities</span></span></span></span></span></span></div><div class="cm-algo-io-row"><span class="cm-algo-io-label">Output</span><span class="cm-algo-io-body"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">Updated learner weights</span></span></span></span></span></span></div></div><ol class="cm-algo-lines"><li style="--indent:0"><span class="cm-algo-ln">1</span><span class="cm-algo-code"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">Center the group rewards and divide by the chosen spread</span></span></span></span></span></span></li><li style="--indent:0"><span class="cm-algo-ln">2</span><span class="cm-algo-code"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">Score the sampled tokens with the current learner</span></span></span></span></span></span></li><li style="--indent:0"><span class="cm-algo-ln">3</span><span class="cm-algo-code"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">Compute probability ratios and clipped token contributions</span></span></span></span></span></span></li><li style="--indent:0"><span class="cm-algo-ln">4</span><span class="cm-algo-code"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">Apply the mask and reduce the selected terms</span></span></span></span></span></span></li><li style="--indent:0"><span class="cm-algo-ln">5</span><span class="cm-algo-code"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.8889em;vertical-align:-0.1944em;"></span><span class="mord text"><span class="mord">Take a gradient step and log diagnostics</span></span></span></span></span></span></li></ol></figure>

A replayed batch can compare a successful rollout with a decline on the same task. The two advantages should have opposite signs, and changing an observation token's reported log probability should leave that position's direct loss unchanged. An all-equal group provides a second test with reference regularization disabled. The reward objective should be zero. That establishes loss behavior before a changing success rate makes a bug hard to isolate.

The shift between inputs and targets is also worth testing. A token is predicted from its prefix, so its score belongs to the prediction just before that token in the sequence. Moving the mask by one position can score a tool token and skip a sampled action without changing the number of active positions. A tiny fixture can distinguish the two by assigning them different log probabilities.

Reproducibility needs the model checkpoint, the task generator version, group size, sampling parameters, behavior-policy version and exact evaluation fixtures. A reported advantage distribution is interpretable after those inputs are recorded. The next chapter shows where this information appears in a multi-turn trace.

A GPU comparison against rejection fine-tuning can control for the opportunity to generate good examples. Rejection fine-tuning reuses only successes as supervised targets. An outcome-based GRPO update also draws a contrast from the failed attempts in each mixed group. Comparing the two with the same rollout budget tests whether this contrast is paying for itself.

The GPU recipe applies the same logic to model-sampled trajectories. **SGLang** samples the sequences and **Megatron** computes the training update in slime. With colocated placement, both phases use the same GPU in turn. A small multi-turn run can then log the fraction of uniform groups, the ratio distribution, episode length and held-out success. Rising training reward supports a claim about the sampled tasks. A claim about reliability also needs held-out scenarios, repeated runs and checks for unwanted state changes. [slime training workflow](https://thudm.github.io/slime/get_started/experiment-guide.html)

---

**Related**: [Policy Gradients](/tutorials/rl-for-llm-agents/policy-gradients/) · [Multi-Turn Rollouts](/tutorials/rl-for-llm-agents/multi-turn-rollouts/) · [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/)
