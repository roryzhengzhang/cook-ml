---
title: "Continual Learning"
description: "Train a candidate for a new policy and check what it retains before promotion."
sidebar:
  order: 9
  label: "Continual Learning"
lastUpdated: 2026-11-18
---

Under the second store policy, a final-sale shirt cannot be exchanged. If it arrived defective, the project policy sends it to an escalation path for possible credit review. The first store policy never asked the agent to make that distinction. A model that learned to complete a straightforward size exchange might keep trying to do so after the rule changes.

The release decision has two parts. The new workflow needs to work under the current policy. The old behaviors that still apply, such as recovering from a temporary stock timeout, need to survive another round of training. The previous checkpoint with updated instructions is the simplest control.

![Policy v2 decisions](/images/rl-for-llm-agents/17i-policy-v2-decisions.png)

An observed failure becomes useful once it can be reproduced from a clean state. The chapter follows that process from the trace into a fixture and then compares changing the prompt, fine-tuning on the new task and training on a mix of new and old tasks. The final sections turn the comparison into a release decision with pinned artifacts and rollback. The scores used here are examples of the decision procedure.

## Policy change

The rules a user sees and the weights that choose the next action can change independently. **Continual learning** updates a model across a sequence of new data or tasks while keeping the skills that are still useful. The store can respond to a rule change in several ways. Updating policy text or hardening a tool may solve some failures with less work.

In policy V2, the simulator refuses final-sale exchanges. It exposes a separate resolution for a defective item. The project verifier expects the structured result `escalate` when a defect is reported and `ineligible` otherwise. Both leave the order and inventory untouched. A separate review process handles any credit eligibility. Its expected behavior comes from the active policy version and the defect flag, so the evaluator can check both the result and the lack of an exchange write.

An ordinary request like O-17 still starts at size S with one M in stock. Its injected inventory timeout remains useful as a retention case. A final-sale request which used to be exchangeable now has a different expected outcome. It belongs in the policy V2 suite. Treating a correctly refused V2 case as a regression against V1 would penalize the intended change.

The active policy needs a **version identifier** carried from the request through the episode. A change during the rollout can make the verifier judge an action under rules the agent never saw. A recorded policy identifier and frozen fixture allow the same old attempt to be evaluated again under the original rules.

The first comparison runs the existing checkpoint with the new policy in context. This is the **prompt-only control**. It isolates whether the same weights can already follow the rule once they see it. Any later training run uses the same V2 instructions, the same tool rules and the same evaluation. A tool guard enforces the rule at execution time in every arm. The learned policy can still matter by choosing the correct resolution promptly and avoiding invalid calls.

This isolates a **policy shift**. The same request and initial state can require a different outcome when store policy changes. A **traffic shift** changes how often requests occur, such as a seasonal increase in sold-out exchanges, even if the right answer on any particular fixture stays the same. The two shifts call for different measurements. For a policy shift, the evaluation needs updated expected outcomes. For a traffic shift, the same cases can be reweighted or supplemented with examples from the newly common slice.

## Evidence from traces

An exchange that ends incorrectly needs to be understood before it becomes a training example. A **trace** records an attempt from request to outcome. A **span** describes one timed part of it, such as a model action, an inventory lookup or the verifier's check. A trace for O-17 can show the first stock call timing out, the retry returning one available and the transaction committing once.

The policy V2 failure looks different. The trace might contain a valid order lookup and a final-sale result followed by an attempted exchange that the tool rejects. The store state is unchanged, but the agent fails to propose the correct resolution. A reviewer can record that failure separately from a tool defect or a missing policy field. Fine-tuning on all failed traces would mix several causes that need different fixes.

For an evaluator, a trace is evidence of one observed attempt. A replay can be a new attempt started from the same fixture or a rescore of the recorded tool outcomes. The latter tests the verifier. The former tests how the candidate agent behaves. On O-17, rescoring a good retry can verify that the transaction count is one, while rerunning a different checkpoint is how a team learns whether it retries at all.

To make the trace replayable, the project records the task fixture or a reference to it, policy version, checkpoint identifier, harness revision, tool responses and final state. Customer identifiers and other sensitive fields are omitted or sanitized before traces leave the environment. Generated synthetic cases hold the fixture and expected result directly. A trace that lacks the necessary state can be useful for diagnosis while still requiring reconstruction before it can become an evaluation item.

![Trace to regression case](/images/rl-for-llm-agents/17i-trace-to-regression-case.png)

The team can keep a **failure label** that marks the first incorrect decision. This might be failure to read the policy, failure to recognize the returned status or an invalid proposal after recognizing it. Grouping by these labels reveals whether the same remedy applies across several traces. The trace carries concrete evidence, while the label summarizes the diagnosis for review.

An **annotation** is an assessment attached by a reviewer or scorer. A useful annotation names the observed action and the expected action, then identifies the task version that makes it correct. Revisions to an annotation are recorded when the policy changes. This avoids turning a subjective complaint or an old decision into the newly active reward.

In the worked integration, **Langfuse** provides the trace viewer and dataset experiments. The project attaches the verifier's score and outcome label, adds a reviewed trace to a versioned evaluation dataset and compares the new and old checkpoints on the same cases. The simulator still owns the rules and produces the scores. [Langfuse datasets](https://langfuse.com/docs/evaluation/experiments/datasets) · [Langfuse SDK experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)

## Fresh tasks and retention tasks

A failed trace can join a regression suite once the failure can be reproduced. The **validation set** supports selecting the replay mix. The final suite supports the comparison once the mix has been selected. If a model scored poorly on a locked V2 suite and the team tuned weights or thresholds to improve that score, the suite now serves follow-up development. A later confirmation requires new cases.

The related training tasks need their own set of fixtures. For a missed credit escalation, the training generator can produce varied defect conditions and store policies. The exact failure and its close variants stay together in the regression set, so a gain on the held-out case can show transfer beyond the training examples.

**Catastrophic forgetting** describes a substantial loss of previously learned behavior after further training. Here it can appear if many successive batches teach final-sale denial and the model begins denying ordinary eligible exchanges too. It can also emerge more subtly. The model might still exchange on the simple case but stop retrying after timeouts.

A **replay mixture** includes representative earlier tasks alongside fresh ones during retraining. For online GRPO, replaying tasks means resetting those earlier fixtures and collecting fresh rollouts from the current policy. Old trajectories may be useful for supervised fine-tuning or for an explicitly off-policy method. Treating them as fresh GRPO data would attach current-policy assumptions to actions generated under older weights.

### Practice: Fresh and replayed tasks

A worked round has room to sample 400 task instances before multiplying by the number of attempts per task. The two training recipes share a total budget.

1. The fresh-only run assigns all 400 slots to tasks where V2 changes the resolution.
2. The mixed run places 300 fresh tasks and 100 earlier skills in those same slots. The earlier quota includes eligible exchanges, sold-out declines and recovery from temporary tool failures.
3. Each recipe collects 4 attempts per slot. Both generate $400\times4=1{,}600$ attempts. The mix spends 400 of them rehearsing prior skills, so the comparison tests the tradeoff at an equal attempt budget.

With the current policy enabled, a replay fixture's outcome is regenerated if the rule changed. A task whose old instructions forbid an exchange may still be useful, but its expected resolution must match V2. That distinction keeps the learner from receiving contradictory rewards for the same scenario under the same text.

The mix also needs variety inside the older quota. If all 100 older draws are paraphrases of O-17, it mainly rehearses one fixture. Sampling earlier categories gives the agent a chance to retain different behaviors. That includes correctly declining a sold-out order and recovering from a transient failure before a valid exchange.

In the CPU notebook, the three policy decisions use separate parameters, and the example mixed run allocates half its slots to older tasks. This lets the notebook compare branches on a laptop. A dense language model shares weights across behaviors, which is where recovery and interference need to be measured. The 300-to-100 mixture above illustrates the arithmetic for a configurable training budget.

## Retraining comparisons

Three candidates help identify what produced a gain. The first is the previous checkpoint with V2 in the prompt. The second continues training on fresh V2 cases. The third continues from the same checkpoint on the mixed task set. The last two use comparable generated-token and update budgets. Each candidate is evaluated on the same new-case and retention fixtures with the same actions, limits and sampling settings.

The [CPU release exercise](/tutorials/rl-for-llm-agents/notebook/#continual-evaluation) copies the original GRPO policy into two candidates. The new-only candidate repeats the two V2 defect fixtures. The mixed candidate trains on those same two fixtures plus two original V1 tasks. Each update sees four fixture slots and eight attempts per slot. Both candidates then face the same V1 and V2 holdouts as the original policy.

Suppose the control succeeds on 11 of 20 new cases and 19 of 20 retention cases. Fresh-only training resolves 18 new cases and keeps 15 earlier ones. Mixed training resolves 17 new cases and keeps 19 earlier ones. The example shows a tradeoff a single average can conceal. Fresh-only fixes one more new case. Mixed restores four earlier cases. These are illustrative counts. A real report would include the paired outcomes and the seed variation from the previous chapter.

![Fresh skill and retention comparison](/images/rl-for-llm-agents/17i-fresh-skill-and-retention-comparison.png)

Using a stricter scoring rule can expose a misleading gain. For example, an agent can avoid invalid attempts by escalating every request it sees. That can look safer on a violation-only metric. Counting unnecessary escalations as failures keeps the metric from rewarding an agent that escalates every case.

The new-workflow results can be split by defect status. Imagine a candidate that escalates all ten defective cases and three of ten cases without a defect. Its broad score is 17 of 20. Seeing ten of ten on the defect slice and seven of ten elsewhere identifies unnecessary escalation as the remaining issue.

The same disagreement table can distinguish an improvement caused by the new policy text from one caused by changing the weights. On a case solved by all candidates, the new weights gained no measured benefit. A case solved only after retraining gives a starting point for inspecting what changed in the trace.

The comparison also uses a failure policy for cases training cannot fix reliably. If the simulator's policy guard incorrectly allows an exchange on a final-sale order, the environment needs a patch. If the checkpoint gets the right answer after receiving an updated policy, the training budget may be better spent on a new skill. Diagnosing the boundary keeps the evaluation honest.

A **retention set** holds tasks whose intended behavior still applies. A test for order O-17 belongs there because the new rule changes final-sale exchanges and O-17 is an ordinary order. A **new-policy set** holds both changed paths and boundary cases, including final sale with and without a defect. Reporting each separately distinguishes a missed escalation from a learned preference to escalate every difficult request.

## Promotion gates

A gain on final-sale cases can hide a regression on ordinary exchanges in the overall average. A **release gate** is a condition the candidate must meet before it can replace the current agent. A candidate may need to improve on new policy cases, preserve an acceptable level of earlier skill and avoid specific harmful writes. The limits are stated before the evaluation. This keeps a large improvement in one area from quietly changing how much decline is tolerated in another.

### Practice: Promotion gates for policy V2

The worked decision uses 20 new cases, 20 retention cases and a zero-violation requirement on the tested fixtures. The minimum is 16 completed new cases and 18 completed retention cases.

<details class="cm-steps not-content" data-walkthrough="promotion-gates-for-policy-v2">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> Release gates for a new policy</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">The candidate needs 16 new-case successes and 18 retention successes.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The prompt-only control clears retention with 19 of 20, but completes 11 new cases and fails the new-case gate.</div><img src="/images/steps/17i-promotion-gates-for-policy-v2/step-01.png" alt="Step 1 of 3, Release gates for a new policy" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The fresh-only model clears the new-case gate with 18 of 20, but keeps only 15 earlier cases and fails retention.</div><img src="/images/steps/17i-promotion-gates-for-policy-v2/step-02.png" alt="Step 2 of 3, Release gates for a new policy" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>The mixed model completes 17 new and 19 retention cases. It passes these counts. If its tested attempts also produce zero prohibited writes, it passes all three example gates.</div><img src="/images/steps/17i-promotion-gates-for-policy-v2/step-03.png" alt="Step 3 of 3, Release gates for a new policy" loading="lazy" decoding="async"></li>
</ol>
</details>

Passing 40 evaluated cases gives limited evidence about rare failures. If 40 independent attempts each have a 1 percent violation chance, the probability of seeing zero is $0.99^{40}\approx0.67$. The independence assumption makes this illustration easy to read. Related fixtures can yield more step-to-step dependence. Larger targeted suites and continued monitoring are needed to measure rare events with useful precision.

Support and failure counts belong alongside percentages. With only 20 retention fixtures, one changed case moves the measured rate by five percentage points. A demand for a one-point improvement has no sensible meaning at that resolution without more cases or another estimator. This is also why the sample size and repeated-trial policy are recorded before judging success.

A useful gate has a defined owner and a response. A forbidden write can block a release immediately. A small drop with wide uncertainty can trigger a larger rerun. Warning signals such as token use or latency can be tracked in the same report even when they do not decide promotion. The team can examine the paired traces before deciding which action to take. [Evaluation and Ablations](/tutorials/rl-for-llm-agents/evaluation-and-ablations/)

## Candidate bundles and rollback

A deployable agent comprises more than weights. The harness, system prompt, policy document, tool schemas and tool execution rules determine what the weights can do. A **release bundle** identifies the compatible versions of these parts along with the evaluation results and the verifier revision used to produce them.

A **shadow evaluation** gives the candidate copies of eligible requests inside a simulator or another isolated replica, then compares its decisions with the running version. It suppresses real side effects. That makes it possible to inspect an attempted duplicate exchange without changing a customer's order. A **canary** gives a small share of actual traffic to a promoted version while monitoring the signals that can trigger a stop.

Shadow and canary traffic need comparable task metadata. Otherwise a candidate receiving mostly easy exchange requests and a control receiving complex cases can produce an apparent gain. Comparing against matched fixtures first and monitoring live outcomes within policy slices makes that traffic mix visible.

Rollback restores a known compatible bundle. Reverting the model weights alone may leave it running a prompt, schema or simulator version that was never evaluated with those weights. A mandatory store policy is a separate constraint. If it must stay active, the fallback is a previous bundle certified under that policy or a rule-based path that can decline or escalate safely. A rollback cannot undo an exchange already applied, so transaction repair has its own process.

![Pinned release and fallback](/images/rl-for-llm-agents/17i-pinned-release-and-fallback.png)

When code, prompts and weights deploy through different systems, a manifest provides the common key. It identifies the build and a checksum for each artifact. If a new prompt is published after the candidate run, the older prompt revision can still be loaded from the manifest for a fair audit.

The release report keeps the predicted rollback trigger close to the numbers it monitors. If a single forbidden mutation causes the candidate to be withdrawn, the monitoring system needs a fast way to observe that mutation. A lagged annotation queue can still help categorize softer issues such as confusing explanations.

A **model registry** records checkpoint versions and links them to training and evaluation provenance. MLflow Model Registry is an example with versioned model records and aliases. The project can keep the local manifest and checkpoints together and use Langfuse to review evaluation traces. Either arrangement needs a link from the served revision back to its exact artifacts. [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)

## Release feedback loop

A continuing release also needs an audit trail for decisions to leave the model alone. If prompt-only clears every gate, the report can point to the evaluated prompt revision and show that further training bought no necessary improvement under the tested conditions. If the mixed run clears gates that prompt-only misses, the saved trajectories give concrete examples of the behavior it learned.

The next training round starts with a versioned report of failures and a fresh evaluation slice. A recurring mistake may enter the replay suite. A new task can enter the training generator after the expected outcome, guard rules and evaluation split are ready. The old suite stays useful for measuring retention after the policy version has been taken into account.

A **feedback loop** returns evaluated deployment failures to the team that can address them. Metrics and traces suggest a change. Reviewed data, controlled training runs and explicit evaluation then decide whether a new checkpoint is promoted. An unreviewed failure stream can overrepresent customers who provide feedback frequently or repeated instances of the same outage.

The team retains both aggregate scores and exemplars. A spike in invalid exchange calls is easier to interpret when reviewed traces reveal whether the agent tried to reuse an old policy, chose the wrong tool after a timeout or was handed contradictory instructions. Those causes warrant different fixes.

An end-to-end run begins by collecting a trace for the failure, reconstructing its fixed fixture and deciding whether the issue belongs in the prompt, tool code or training data. It evaluates the unchanged weights with current instructions. If training is warranted, it compares the fresh-only and mixed runs, checks promotion gates and attaches the complete bundle to the decision. The same process can be repeated for the next store change after the first candidate has been evaluated. [Training Systems](/tutorials/rl-for-llm-agents/scaling-the-training-loop/)

---

**Related**: [Evaluation and Ablations](/tutorials/rl-for-llm-agents/evaluation-and-ablations/) · [Training Systems](/tutorials/rl-for-llm-agents/scaling-the-training-loop/)
