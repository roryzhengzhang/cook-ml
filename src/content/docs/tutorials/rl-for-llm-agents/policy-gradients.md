---
title: "Policy Gradients"
description: "Work through one update from a sampled action to a changed action probability."
sidebar:
  order: 4
  label: "Policy Gradients"
lastUpdated: 2026-11-18
---

The policy starts by choosing the correct exchange only 40% of the time. Its next attempt succeeds. One training update raises that chance to about 47%. Nothing about order O-17 has changed. The update changed how likely the agent is to make the same decision when it sees those facts again.

The trainer sees the result of one sampled choice at a time. A good update must make use of that small piece of evidence even when several decisions and a tool error came before the outcome.

![Exchange objective](/images/rl-for-llm-agents/17d-exchange-objective.png)

This chapter reduces the store to one decision so the improvement can be checked with a calculator. Expected reward sets the objective. REINFORCE estimates the direction of improvement from the decisions the policy actually makes. The full store adds a wait between a good decision and its payoff, so the same update needs a return and a way to measure how surprising it was. The chapter ends with Proximal Policy Optimization, where an update can revise its own training data's probabilities enough to make the original samples a poor guide.

## Policy objective

An exchange policy has a number attached to each action it could take. Before the first exercise, the environment reveals all the facts for O-17 in one observation. The order is eligible and one medium shirt is available. The policy chooses either to exchange the shirt or to decline. A valid exchange earns a reward of 1. Declining earns 0. The agent starts at 40% exchange and 60% decline.

The **expected reward** is the average reward the policy would get over many attempts from this same starting point. The experiment can calculate it exactly because it knows the outcome of both choices. With one attempt it will observe either 1 or 0. With many attempts the mean should approach 0.40. These are two views of one objective, one from the model and simulator together and one from a sample of their behavior.

A **logit** is an unbounded score that can be converted to a probability. The toy policy keeps the decline logit at zero and learns one score for exchange. The **sigmoid** takes this score and maps it between zero and one.

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}\qquad J(z)=p\times 1+(1-p)\times 0=p
$$

Here $z$ is the exchange logit, $e$ is the base of the natural logarithm, $p$ is the exchange probability, and $J(z)$ is the expected reward. Increasing the logit increases the expected reward for this observation. A real policy has one set of weights for many observations. It also needs cases where decline is correct so it can learn when each decision is appropriate.

An ineligible order exposes the difference. Its correct decision is decline, so the policy earns 0 from exchange and 1 from decline. If the same score gave exchange probability $0.4$, the expected reward would be $0.6$. The gradient would favor a lower score on that observation. A policy with enough information and capacity can learn different probabilities for the two orders.

The toy problem is a **contextual bandit**. It supplies context, takes one action, and immediately produces a reward. Using one decision first makes each probability and reward visible. Removing that restriction later changes which reward belongs to a decision, while the single-action calculation remains a test case.

## REINFORCE

Suppose the policy chose to exchange O-17 and earned a reward of 1. The update should make that action more likely in the same situation. The trainer needs to know how each weight affects the probability of the sampled action. The **score function** is the gradient of the sampled action's log probability. A gradient gives the direction and rate at which a quantity changes with each weight. Multiplying the score function by the observed reward gives the simplest policy gradient estimate. This Monte Carlo method, which uses random samples to estimate an average, is called **REINFORCE**. [Original REINFORCE paper](https://doi.org/10.1007/BF00992696)

$$
g=r\nabla_\theta\log\pi_\theta(a\mid o)
$$

Here $a$ is the sampled action, $o$ is the observation, $\pi_\theta(a\mid o)$ is the probability the policy assigns to that action, $\theta$ is the set of trainable weights, and $r$ is the observed reward. The operator $\nabla_\theta$ differentiates with respect to those weights. The estimate $g$ points in the direction used to increase expected reward. For the binary policy, the derivative is $1-p$ after sampling exchange and $-p$ after sampling decline.

Across repeated trials the sampled estimate matches the exact derivative. At $p=0.40$ the expected estimate is $0.4\times(1\times 0.6)+0.6\times(0\times(-0.4))=0.24$. This equals the derivative $p(1-p)$ obtained directly from the sigmoid. The simulator can compute the exact derivative here. The sampled version still works when an agent has far too many possible actions to enumerate. Its estimate varies from batch to batch, especially when most attempts earn zero.

### Practice: One rewarded exchange

One update can be checked in probability space after the optimizer changes the logit.

<details class="cm-steps not-content" data-walkthrough="one-rewarded-exchange">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> One policy gradient update</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">One sampled action changes the odds on the same observation.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The exchange probability starts at <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.40</mn></mrow><annotation encoding="application/x-tex">0.40</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.40</span></span></span></span>.</span> Its logit is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ln</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>0.4</mn><mi mathvariant="normal">/</mi><mn>0.6</mn><mo stretchy="false">)</mo><mo>≈</mo><mo>−</mo><mn>0.4055</mn></mrow><annotation encoding="application/x-tex">\ln(0.4/0.6)\approx -0.4055</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mop">ln</span><span class="mopen">(</span><span class="mord">0.4/0.6</span><span class="mclose">)</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">≈</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.7278em;vertical-align:-0.0833em;"></span><span class="mord">−</span><span class="mord">0.4055</span></span></span></span>,</span> where <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ln</mi><mo>⁡</mo></mrow><annotation encoding="application/x-tex">\ln</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6944em;"></span><span class="mop">ln</span></span></span></span> is the natural logarithm.</div><img src="/images/steps/17d-one-rewarded-exchange/step-01.png" alt="Step 1 of 3, One policy gradient update" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The sampled exchange earns <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1</mn></mrow><annotation encoding="application/x-tex">1</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">1</span></span></span></span>.</span> Its gradient estimate is <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>1</mn><mo>×</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mn>0.4</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.6</mn></mrow><annotation encoding="application/x-tex">1\times(1-0.4)=0.6</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.7278em;vertical-align:-0.0833em;"></span><span class="mord">1</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">(</span><span class="mord">1</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mord">0.4</span><span class="mclose">)</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.6</span></span></span></span>.</span></div><img src="/images/steps/17d-one-rewarded-exchange/step-02.png" alt="Step 2 of 3, One policy gradient update" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>Gradient ascent adds <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5</mn><mo>×</mo><mn>0.6</mn><mo>=</mo><mn>0.30</mn></mrow><annotation encoding="application/x-tex">0.5\times 0.6=0.30</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.7278em;vertical-align:-0.0833em;"></span><span class="mord">0.5</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.6</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.30</span></span></span></span> to the logit at learning rate <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.5</mn></mrow><annotation encoding="application/x-tex">0.5</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.5</span></span></span></span>.</span> The new logit is about <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>−</mo><mn>0.1055</mn></mrow><annotation encoding="application/x-tex">-0.1055</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.7278em;vertical-align:-0.0833em;"></span><span class="mord">−</span><span class="mord">0.1055</span></span></span></span>,</span> so its sigmoid is about <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4737</mn></mrow><annotation encoding="application/x-tex">0.4737</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.4737</span></span></span></span>.</span></div><img src="/images/steps/17d-one-rewarded-exchange/step-03.png" alt="Step 3 of 3, One policy gradient update" loading="lazy" decoding="async"></li>
</ol>
</details>

![One update](/images/rl-for-llm-agents/17d-one-update.png)

The new chance is 7.37 percentage points higher. This example follows one favorable sample. An all-decline batch would leave the logit unchanged under the same uncentered reward update.

## Returns and delayed rewards

One store decision can earn a reward several steps after it happens. In the full exchange, retrying a stock lookup has an immediate reward of zero. Finding a medium on that retry can make a later successful exchange possible. A **return** adds up rewards from the current decision onward. The earlier choice then has a signal that includes the eventual outcome.

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

Here $t$ is the current step, $T$ is the number of decisions, $k$ indexes rewards from step $t$, and $r_k$ is the reward observed after decision $k$. The **discount factor** $\gamma$ lies between zero and one. The return $G_t$ includes immediate reward at full strength and later reward with a smaller weight when $\gamma$ is less than one.

Suppose the last three decisions are a lookup that times out, a retry that returns stock 1, and a successful submission. The rewards are $[0,0,1]$. At discount 1 all three returns are 1. At discount 0.9 they are $[0.81,0.9,1]$. The same finish has less effect on decisions further from it. Either choice rewards the whole sequence to some degree. The return alone cannot tell whether the lookup, the retry, or a detail of the eventual request deserves the strongest credit.

Reward to go drops rewards that arrived before a sampled decision. Past rewards are already fixed when the agent chooses its next action. Including them would add noise. The store tutorial uses an undiscounted terminal outcome for its first GRPO lab so that valid longer paths can still earn the same outcome reward. Turn limits and operational costs can be measured separately before deciding whether to include either in the objective.

![Delayed return](/images/rl-for-llm-agents/17d-delayed-return.png)

The same idea applies when an action contains several tokens. If the chance of the first token is $0.5$ and the chance of the second given the first is $0.4$, the chance of the pair is $0.5×0.4=0.2$. Taking a logarithm turns multiplication into addition. The gradient of the action's log probability can therefore draw on each of the sampled tokens. Chapter [GRPO](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/) uses this connection when it works with token probabilities and clipping.

A reward of zero produces a zero update under the basic rule. Early in training an agent that almost never succeeds may use dozens of rollouts to get one informative outcome. A baseline can improve the information in mixed outcomes. More demonstrations or an easier starting task can help when the batch almost never contains a success.

## Baselines and advantages

An exchange can be correct and still provide little information about what to do next. If a policy already succeeds on nearly every eligible order, a success is close to what it usually earns. A **baseline** is an estimate of that usual return, chosen using only information available before the sampled action. Subtracting it from the return produces an **advantage**, the amount by which the observed return beat or fell short of the baseline.

$$
A_t=G_t-b(o_t)
$$

Here $A_t$ is the advantage for action $t$, $G_t$ is its return, $o_t$ is the observation available before the action, and $b(o_t)$ is the baseline at that observation. The sign changes what a sample teaches. At $p=0.4$ on O-17, a baseline of $0.4$ gives the successful exchange an advantage of $0.6$ and the incorrect decline an advantage of $-0.4$. Both samples now give a direction. The exchange gradient is $0.6\times0.6=0.36$. The decline gradient is $-0.4\times(-0.4)=0.16$. Both push toward exchange.

The mean gradient remains $0.4\times 0.36+0.6\times 0.16=0.24$. Raising one action's probability takes mass from the other action. Averaged across draws, a constant multiplied by the score function has zero effect because the probabilities always sum to one. An action-dependent baseline can break that cancellation. So can neglecting to stop gradients through a learned baseline when calculating the policy weight.

### Practice: A baseline across two outcomes

The same binary policy makes it possible to compute the variation of the two gradient estimates exactly.

<details class="cm-steps not-content" data-walkthrough="a-baseline-across-two-outcomes">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Two outcomes and a baseline</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">A fixed baseline keeps the mean update and changes the variation.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The original update gives estimates <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6</mn></mrow><annotation encoding="application/x-tex">0.6</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.6</span></span></span></span> and <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0</mn></mrow><annotation encoding="application/x-tex">0</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0</span></span></span></span> at probabilities <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4</mn></mrow><annotation encoding="application/x-tex">0.4</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.4</span></span></span></span> and <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.6</mn></mrow><annotation encoding="application/x-tex">0.6</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.6</span></span></span></span>.</span> Their squared deviations from the mean <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.24</mn></mrow><annotation encoding="application/x-tex">0.24</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.24</span></span></span></span> average to <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0864</mn></mrow><annotation encoding="application/x-tex">0.0864</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.0864</span></span></span></span>.</span></div><img src="/images/steps/17d-baseline-two-outcomes/step-01.png" alt="Step 1 of 2, Two outcomes and a baseline" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>With baseline <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.4</mn></mrow><annotation encoding="application/x-tex">0.4</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.4</span></span></span></span>,</span> the two estimates are <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.36</mn></mrow><annotation encoding="application/x-tex">0.36</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.36</span></span></span></span> and <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.16</mn></mrow><annotation encoding="application/x-tex">0.16</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.16</span></span></span></span>.</span> Their squared deviations average to <span class="cm-math-nowrap"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.0096</mn></mrow><annotation encoding="application/x-tex">0.0096</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">0.0096</span></span></span></span>.</span> The variance is nine times smaller in this example.</div><img src="/images/steps/17d-baseline-two-outcomes/step-02.png" alt="Step 2 of 2, Two outcomes and a baseline" loading="lazy" decoding="async"></li>
</ol>
</details>

![Baseline variance](/images/rl-for-llm-agents/17d-baseline-variance.png)

The lower variance here makes individual samples more informative about the direction. The calculation depends on this policy and baseline. Poor baseline estimates can help less, and the number of samples matters when judging a measured difference.

A training implementation usually minimizes a loss. The actor loss uses the negative of the advantage times the log probability, treating the advantage as fixed for that step. The negative sign converts a request to maximize the weighted log probability into a request to minimize a loss. A numerical test that checks the probability before and after the step catches a reversed sign.

## Actor and critic

The useful baseline varies across tasks. Before a known in-stock and eligible exchange, the expected return might be high. Before an ambiguous policy check with a short tool budget, it may be low. A **value function** estimates the expected future return from a state or observation under the policy. A learned approximation to that value is called a **critic**. The **actor** is the policy that chooses the action.

A critic learns from return targets or from the immediate reward plus its own estimate of what comes next. The latter uses **bootstrapping**, where a current estimate helps form a target at an earlier step. If stock is found after a retry, the critic may predict $0.75$ at that point. A one-step target with zero immediate reward and discount $0.9$ is then $0.675$. The target arrives before an entire episode finishes. Its accuracy depends on the next estimate.

$$
\delta_t=r_t+\gamma V(o_{t+1})-V(o_t)
$$

Here $\delta_t$ is the one-step **temporal difference error**, $r_t$ is the immediate reward, $\gamma$ is the discount, and $V(o_t)$ and $V(o_{t+1})$ are the critic's present and next predictions. A true terminal state has zero remaining value in this example. If the present prediction was $0.30$ before the retry, the step gives a positive error of $0.375$. The result suggests that the action led to a more promising observation than the critic expected.

<details class="cm-collapse" id="generalized-advantage-estimation">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">Generalized advantage estimation</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

A one-step error reacts quickly but can inherit a critic's mistakes. A full return waits for the outcome and often varies more. **Generalized advantage estimation** adds discounted one-step errors across later decisions and uses a second weight, usually written $\lambda$, to control how much those later errors count. A small $\lambda$ leans on the critic's near-term prediction. A larger $\lambda$ lets later observations and rewards affect earlier advantages. [Generalized advantage estimation](https://arxiv.org/abs/1506.02438)

</details>

A critic can also be poorly calibrated on observations the warm-start policy rarely sees. Its value estimate before the timeout may be trained on many ordinary exchanges and few transient failures. Logging the value and realized return for a held-out set of those states helps diagnose whether poor advantages come from the critic or from the policy's behavior after the observation.

In an LLM pipeline the critic can require its own head, optimizer state, forward pass, and training checks. That cost matters when the rollout policy already consumes much of a single GPU's memory. The next chapter builds a baseline whose direct inputs are rewards from multiple attempts at the same task.

## PPO and probability ratios

A rollout remains in the batch after the first training step changes the policy. The action was sampled when exchange had probability $0.4$. The policy being trained might now give it probability $0.5$. Its current probability divided by the probability that produced the sample is **importance ratio** $0.5/0.4=1.25$. Ratios weight actions that became more likely more heavily, and actions that became less likely less heavily, when the new policy is scored using old samples.

**Proximal Policy Optimization** (PPO) uses a clipped objective to limit the incentive for some large changes on a batch. For a positive advantage it stops rewarding an increase after the ratio passes an upper threshold. For a negative advantage the corresponding plateau is beneath the lower threshold. An unfavorable move can still make the objective worse.

$$
L_t=\min\big(\rho_t A_t,\ \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big)
$$

Here $L_t$ is the contribution the actor maximizes, $A_t$ is the estimated advantage, $\rho_t$ is the new probability divided by the sampling probability, and $\epsilon$ controls the clipping interval around 1. The minimum chooses the smaller contribution. For a rewarded exchange with advantage $0.6$ and $\epsilon=0.2$, a ratio of $1.25$ gives an unclipped value of $0.75$ and a clipped value of $0.72$. The objective selects $0.72$. For a bad decline with advantage $-0.4$ and a ratio of $0.70$, the minimum selects $-0.32$ instead of $-0.28$. That is the point where the loss stops favoring still lower probability on this sample.

![Ppo clipping](/images/rl-for-llm-agents/17d-ppo-clipping.png)

The objective discourages large favorable changes on sampled actions. Changes to shared weights can still move other actions or observations substantially. Practical runs track the fraction of ratios that land on clipped branches and the distance the policy moves between updates. The [PPO paper](https://arxiv.org/abs/1707.06347) shows how this surrogate objective supports several passes over collected data.

The scoring policy and rollout policy can still share weights at the beginning of a training pass. A ratio of 1 means the two policies currently assign the same probability to the chosen action. The objective still has a gradient because changing the learner can change its numerator. Keeping the captured denominator fixed is part of the check.

Fresh rollouts draw from the current policy, then the trainer spends an update on them. This is **on-policy** sampling at collection time. Reusing that batch after a weight change trades fewer generations for samples that came from an older distribution. Importance ratios address the change at the sampled decisions. The quality of the estimate still depends on how much those policies overlap.

The lower bound in the clipping range matters most for negative advantages. In the example that uses $A_t=-0.4$, lowering the ratio from $0.8$ to $0.7$ leaves both clipped contributions at $-0.32$. At ratio $1.3$ the contribution falls to $-0.52$. Clipping still lets the learner see a penalty for making that bad decline more likely.

## Policy gradient checks

The store's one-action fixture gives a way to test policy updates before committing a GPU to the language model. A case where exchange earned reward should raise its probability when its advantage is positive. A known bad decline with negative advantage should lower the decline probability. A zero advantage should eliminate the reward-driven contribution for that action. These checks fit on a laptop and can be repeated after changing the optimizer or loss.

A useful before-and-after check holds the set of observations fixed and inspects a mix of eligible and ineligible orders. Raising exchange probability on both states can improve one trace and hurt the agent overall. The simulator's ground truth makes that failure visible even when sampled training reward happens to rise for a small batch.

The same checks become harder to see once an action is a sequence of tokens. A positive sequence advantage can raise the likelihood of both meaningful tokens and incidental phrasing. The model needs many rollouts for the useful action patterns to recur. Each update still acts on the same ingredient as the tiny policy, the log probability of something it sampled. The next chapter groups several outcomes before that update to estimate which attempts were relatively strong.

---

**Related**: [Model Baselines](/tutorials/rl-for-llm-agents/base-model-and-baselines/) · [GRPO](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/) · [Post-Training](/tutorials/llm-fundamentals/post-training-from-base-model-to-assistant/)
