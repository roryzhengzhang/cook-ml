---
title: "Training Systems"
description: "Profile rollout generation and training on one GPU, then study how separate GPU pools overlap their work."
sidebar:
  order: 8
  label: "Training Systems"
lastUpdated: 2026-11-18
---

The four attempts in an exchange batch can finish at different times. Three are ready for training after eight seconds. The fourth is still waiting for the inventory service to return. The learner has a complete batch only when the fourth arrives. Speeding up that wait and speeding up the GPU update are different jobs.

A reinforcement learning job uses the model to collect experience and then changes the model using that experience. The two phases put different demands on the GPU. They can alternate on one device or overlap on separate devices. Either layout needs to keep track of which weights produced the actions that the trainer reads.

![Colocated and separate gpu phases](/images/rl-for-llm-agents/17h-colocated-and-separate-gpu-phases.png)

The chapter starts by timing what a single batch actually does. A memory budget explains why the first project run alternates training and rollout on one GPU. Splitting those phases across devices makes overlap possible and raises a new problem when the learner changes while old attempts are still coming in. The last sections translate those decisions into slime settings and a reproducible pilot. All timing and memory arithmetic here is illustrative.

## Phase timings

A rollout spends time sampling action tokens, running tools and checking the result. When O-17's stock check times out, the agent waits for that response before deciding whether to try again. The GPU may serve other rollouts during the wait. The elapsed time for the group still depends on when the last rollout finishes.

The [timing exercise](/tutorials/rl-for-llm-agents/notebook/#colocation-budget) gives four simulated attempts durations of 5, 6, 8 and 12 seconds. The 12-second attempt sets the group’s wall time. On the single-GPU pilot, the same measurement would reveal whether the agent is waiting on a long exchange, the trainer or the transfer of updated weights.

A **wall-clock time** is elapsed time measured from start to finish. **Throughput** is work completed per unit of that time, such as verified episodes per second. A **phase breakdown** accounts for time in generation, tool calls, reward checks, batching, the optimizer update and weight sync. Recording it from clean boundaries avoids mistaking time when the GPU is running an inference kernel for time when a sample is waiting in an input queue.

### Practice: A rollout batch on one GPU

One illustrative batch contains four O-17 attempts. Their end-to-end rollout durations are 5, 6, 8 and 12 seconds. The learner needs 6 seconds to update the model and the handoff takes 2 seconds for this timed example.

<details class="cm-steps not-content" data-walkthrough="a-rollout-batch-on-one-gpu">
<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> A colocated training round</span><span class="cm-collapse-hint" aria-hidden="true"></span><span class="cm-steps-sub">The last attempt returns after 12 seconds, then the model updates.</span></summary>
<ol class="cm-steps-list">
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 1</span>The rollout phase lasts 12 seconds because it starts all four attempts together and waits for the last one.</div><img src="/images/steps/17h-rollout-batch-on-one-gpu/step-01.png" alt="Step 1 of 3, A colocated training round" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 2</span>The optimizer phase and weight handoff bring the batch to <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>12</mn><mo>+</mo><mn>6</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>20</mn></mrow><annotation encoding="application/x-tex">12+6+2=20</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.7278em;vertical-align:-0.0833em;"></span><span class="mord">12</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.7278em;vertical-align:-0.0833em;"></span><span class="mord">6</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">2</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6444em;"></span><span class="mord">20</span></span></span></span> seconds.</div><img src="/images/steps/17h-rollout-batch-on-one-gpu/step-02.png" alt="Step 2 of 3, A colocated training round" loading="lazy" decoding="async"></li>
<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step 3</span>Four verified attempts in 20 seconds give 0.2 attempts per second. The final four seconds of rollout reduce batch throughput even when the other three attempts are already ready.</div><img src="/images/steps/17h-rollout-batch-on-one-gpu/step-03.png" alt="Step 3 of 3, A colocated training round" loading="lazy" decoding="async"></li>
</ol>
</details>

A **straggler** is an unusually late item that delays work depending on the full batch. Within a group for GRPO, the trainer needs the group rewards before it can compare attempts. A timeout can appear as one such late item. Timing complete groups and their individual attempts shows whether the bottleneck sits in tool service, inference, or a few particular tasks.

The clock around a GPU API call can undercount computation if the host queues work and returns immediately. Phase timing should use synchronized boundaries or GPU events for device work. The profiler also distinguishes intervals that overlap. Summing them measures total activity. Measuring from batch start to batch finish measures the delay users actually observe.

![Rollout straggler](/images/rl-for-llm-agents/17h-rollout-straggler.png)

A **cold start** includes one-time work such as loading weights, importing kernels or creating the simulation worker. A **steady-state batch** runs after that work has already happened. The first batch might take 35 seconds and the next three around 20 each. Both numbers are relevant to a short tutorial run. The longer sequence also gives a steadier throughput estimate for an overnight training job.

**Backpressure** limits how much work producers can create before downstream consumers accept more. If grading slows while SGLang keeps generating, finished outputs can pile up and hold memory. The profiler can attribute the delay to the verifier queue. A bound on pending work makes that contention observable and keeps a temporary tool outage from expanding the queue indefinitely.

The unit of output matters as much as the timing. Generation tokens per second can rise while verified groups per second falls if the model produces longer responses. The project records both and tags the total by task length and ending reason. A higher device throughput then has a visible connection to how much training data actually arrived.

## Memory accounting

A checkpoint stores model weights. A live training process also holds gradients, optimizer state and temporary results needed for the backward pass. SGLang keeps weights and a **key-value cache** that stores attention's keys and values for the tokens already processed. The cache grows with the number and length of active sequences. The two phases can therefore reach different memory peaks even with the same model.

For a representative 1.5-billion-parameter model, **BF16** uses two bytes per parameter. Just one copy of the weights uses approximately $1.5\times10^9\times2=3\times10^9$ bytes, or 3 decimal GB. An illustrative accounting adds 3 GB for BF16 gradients, 6 GB for a full-precision master weight copy and 12 GB for two full-precision Adam moment buffers. The first tracks an average of recent gradients and the second tracks an average of their squares. That brings tensors associated with parameters to about 24 GB before temporary activations, CUDA allocations, communication buffers or anything held by the inference engine. Actual layouts can store or shard these tensors differently.

![Parameter memory breakdown](/images/rl-for-llm-agents/17h-parameter-memory-breakdown.png)

Those are decimal units. An operating-system tool may report gibibytes, which divide the byte count by $2^{30}$, while decimal GB divides it by $10^9$. The same 24-billion-byte allocation is about 22.4 GiB. Capacity checks compare quantities in the same units.

The arithmetic explains why a 3 GB checkpoint can be unsuitable for a 24 GB GPU in a naive full-parameter training layout. It also identifies practical knobs. Smaller **microbatches** lower the number of training sequences held simultaneously. **Gradient accumulation** combines gradients from several of those microbatches before an optimizer step. **Activation recomputation** discards selected intermediates and computes them again during the backward pass. These changes trade elapsed time for a lower memory peak.

**Allocated memory** describes bytes currently held by live tensors under the framework's allocator. **Reserved memory** includes blocks the allocator keeps for reuse. Device-wide accounting includes memory that never passed through that allocator, such as a separate inference worker. The measured peak needs the relevant device and process scope, since the readings answer different questions.

When the bottleneck is generation, leaving fewer rollouts active at once shrinks the key-value cache. Shorter histories shrink it too, but trimming a conversation can change the agent's actions. The evaluation then needs to reflect the new history policy. The model size, actual precision and worst episode length belong in the memory measurements alongside the figure from the short pilot. [Serving Systems](/tutorials/llm-deep-dive/inference-systems-at-scale/)

## Colocation

One GPU can generate a batch and then train on it. The ability to alternate those jobs on the same hardware is called **colocation**. In the store project, the SGLang phase collects the tool trajectories. Megatron then uses their scores to update the policy. The inference side loads the new weights before the next batch. **Offload** moves selected state out of GPU memory while the other phase needs that space.

An illustrative run might reach 30 GB during training and 18 GB during rollout. Their sum is 48 GB. If the phases alternate and free their temporary state, the observed peak can be close to the larger phase. The actual peak still includes anything kept resident and any temporary copies during weight transfer. A before-and-after memory sample can miss that spike, so the pilot records peak allocated and peak reserved memory by phase and observes device memory during handoff.

The initial target is one NVIDIA L40S with 48 GB of memory and a 1.5B dense base model. A smaller setup on a 24 GB RTX 4090 is another candidate. The project records the exact model, image, driver and hardware after the full SFT-to-evaluation run completes and resumes on each proposed card. The L40S and RTX profiles are targets pending that test. [NVIDIA L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/) · [NVIDIA RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) · [slime hardware guidance](https://thudm.github.io/slime/get_started/quick_start.html)

The weight handoff needs its own check. A short prompt can be generated immediately before and after an update with the same model settings and compared against the corresponding checkpoint versions. If the collector continues to report the pre-update version, a flat learning curve could come from workers drawing all of their examples from stale weights.

A readable first run puts correctness checks before long experiments. In generation-only mode, the expected tool order and reward can be inspected on a small batch. In training-only replay, a batch previously captured from the engine checks that the trainer sees the expected probabilities and mask. A short combined run then tests weight handoff and checkpoint resume. [slime experiment guide](https://thudm.github.io/slime/get_started/experiment-guide.html)

## Separate rollout workers

A second GPU can run inference while the first trains. This layout is called **disaggregation** when the two jobs use separate device pools. It helps most when the next group of rollouts can be produced while the learner works on a prior group. A separate device still sits idle if the experiment waits for a new policy before it will start collecting.

Using the earlier example, the serial sum of 12 seconds of rollout work and 6 seconds of training work is 18 seconds with handoff excluded. An ideal overlap would approach the longer of the two stages, or 12 seconds per steady-state batch. For four attempts, that is about 0.33 per second instead of 0.22. Synchronization, startup, uneven tasks and queue bounds can reduce that gain. This calculation is an upper bound for that simplified example.

**Asynchronous collection** allows workers to keep collecting while the learner updates on other groups. A **queue** holds completed or pending work until the learner can consume it. A bounded queue limits how far collection can run ahead. Measuring queue depth and completed-group time shows whether the next step needs more rollout capacity or a faster learner. Allocating more rollout resources to a learner-bound run can increase idle workers with little effect on training.

### Practice: Ideal overlap

The earlier 12-second rollout and 6-second update fit into a simple steady-state schedule. This example excludes weight sync, which the real benchmark measures separately.

1. The first rollout uses seconds 0 through 12. Training for that batch uses seconds 12 through 18.
2. The rollout worker begins the second batch at second 12 and finishes it at second 24. The first update fits into six of those twelve seconds.
3. Once the pipeline is filled, each rollout batch arrives every twelve seconds. The first result still had to pass through both stages.

The example illustrates why batching and concurrency optimize rate while an individual customer episode can retain its original latency. Work at the store still finishes when its own tool calls and model actions finish.

<details class="cm-collapse" id="model-and-sequence-parallelism">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">Model and sequence parallelism</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

A model that exceeds one GPU can divide weights and computation across devices. **Tensor parallelism** distributes an operation's tensors across workers. **Pipeline parallelism** places different layers on different devices. **Context parallelism** distributes sequence positions for long inputs. The right layout depends on the model and communication cost. Splitting a small model solely to pursue a larger GPU count can spend time in communication while leaving both cards underused. [Parallelism](/tutorials/llm-deep-dive/distributed-training-and-parallelism/)

</details>

## Policy lag

Overlapping rollout and training changes which policy produced a sample. A worker can start O-17 under checkpoint v10. By the time its retry returns and the group finishes, the learner may be about to update version v12. The model that generated the tokens is the behavior policy. Their stored log probabilities describe that policy's actual choices.

**Policy lag** is the gap between the behavior version and the learner version that consumes a trajectory. A sample generated by v10 and used at v12 has age two updates. A **staleness limit** caps that age. An age-one limit would exclude that sample or trigger recollection. The queue policy also needs to account for the whole group if advantages compare attempts from a shared task.

The log-probability ratio from [GRPO](/tutorials/rl-for-llm-agents/group-relative-policy-optimization/) gives a way to correct for differences in how a sampled action is weighted. Suppose a token had probability 0.20 under the collecting policy and 0.30 under the learner. The ratio is 1.5. Much larger ratios can make a small number of samples dominate the gradient. Ratio clipping or other corrections bound some of that effect, while tighter age bounds reduce exposure to stale samples. The tradeoff is lost work and possible selection for faster episodes if older, longer groups are dropped more often.

![Policy version age](/images/rl-for-llm-agents/17h-policy-version-age.png)

An age limit counts versions. Action probabilities can move by different amounts across the same number of updates. A large learning-rate step can create a large ratio across a single version. A series of tiny updates can produce a small ratio across several versions. Logging the age histogram next to the ratio and clipping statistics distinguishes time spent waiting from actual drift in the policy.

The project uses a bounded number of retries and keeps an ending reason for every attempt. This makes a throughput gain visible even if the new schedule sends a greater share of hard episodes to the timeout path.

The first async experiment keeps a single behavior version for each complete rollout and logs age at consumption. In more advanced systems, a rollout can continue after a weight update. The bookkeeping then needs the provenance and behavior probabilities for the affected segments or tokens, plus a clear cache policy. Asynchronous speed matters only together with the rate of usable groups and held-out behavior. [slime scheduling guide](https://thudm.github.io/slime/get_started/experiment-guide.html#choose-gpu-ownership-then-scheduling) · [NeMo RL async GRPO guide](https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html)

## Training frameworks

Moving rollout generation onto a second GPU also means moving completed attempts back to the learner and sending new weights in the other direction. A training framework coordinates that work. It launches the learner and inference workers, places them on devices, moves completed samples, performs model updates and keeps serving weights current. The store simulator and verifier still define what counts as success.

In the project, **slime** coordinates Megatron for training and SGLang for generation. The custom generation hook drives the exchange episode. A custom reward hook can score its outcome, depending on where the verifier is called. The project profiles its synchronous colocated path first. The async lab then holds the task distribution, starting weights and evaluation fixed while changing the scheduling. The GLM-5 technical report describes slime as its training infrastructure and discusses customizable rollouts and failure recovery at larger scale. [slime customization guide](https://thudm.github.io/slime/get_started/customization.html) · [GLM-5 technical report](https://arxiv.org/html/2602.15763v2#S3.SS6)

Library choice follows the boundary the project needs to customize. The options below use different interfaces and defaults, so a port should confirm matching masks, loss normalization and reward grouping before comparing curves.

| Framework | What its interface makes easy to study |
|---|---|
| Hugging Face TRL | GRPO with custom reward functions and tool or environment integration. [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) |
| verl | Multi-turn rollouts with backend options including PyTorch sharding and Megatron. [verl project](https://github.com/verl-project/verl) |
| Prime RL | Asynchronous rollout and training with native integration for its verifiers environments. [Prime RL project](https://github.com/PrimeIntellect-ai/prime-rl) |
| NVIDIA NeMo RL | Separate collection, weight refit and bounded trajectory age in an asynchronous pipeline. [NeMo RL async GRPO guide](https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html) |

## Pilot and reproducibility

The hardware pilot starts with a known fixture and ends with a recoverable training run. Passing the first optimizer update proves the phases can connect. Resuming from a saved checkpoint after interrupting the next batch checks whether the training history, sample ordering and policy version can be reconstructed.

A **run manifest** ties together the initialization checkpoint, code revision, container image, driver, hardware, maximum context and generation limits, seeds and environment build. The first full report captures peak device memory, wall time by phase, total GPU-hours, usable groups per second and held-out success. The profiler records how many episodes end through a timeout or truncation, since throughput can rise simply because more attempts stop early.

**Recovery** includes knowing what was committed. Suppose a run checkpoints after update 20 and crashes with two rollouts in flight. On restart it may recollect those attempts from the recorded task seeds, or restore a saved queue if its implementation supports that. Either approach needs a documented rule for records collected after update 20 so a group's samples are consumed at most once per intended update.

A **compute budget** translates planned work into an explicit hardware allocation. Two devices held for 45 minutes cost 1.5 GPU-hours even if one spent part of that time waiting on tools. Reporting that quantity alongside wall time distinguishes saving reader time from saving total hardware time.

If four GPUs are available, two independent learner-and-rollout pairs can increase experiment throughput across seeds. They answer a different question from two devices that overlap the phases within a single run. CS336's systems assignment profiles kernels and end-to-end training, and its scaling assignment works within a fixed compute budget. The project preserves those habits by measuring the single-device recipe before spending additional hardware on a particular bottleneck. [CS336 systems assignment](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf) · [CS336 scaling assignment](https://github.com/stanford-cs336/assignment3-scaling/blob/main/cs336_assignment3_scaling.pdf)

The pilot compares hardware and configurations at fixed validation quality as well as fixed work. If a lower memory cap saves one allocation but doubles the time to reach the same held-out success, the report should show the total GPU-hours for that run. A faster first batch cannot tell whether that tradeoff is worthwhile.

A fast run still needs the evaluation from [Evaluation and Ablations](/tutorials/rl-for-llm-agents/evaluation-and-ablations/). Once it produces a candidate whose quality can be reproduced, the model can be trained again on an updated workflow and measured against both versions of the store's tasks. [Continual Learning](/tutorials/rl-for-llm-agents/continual-learning-and-release/)

---

**Related**: [Multi-Turn Rollouts](/tutorials/rl-for-llm-agents/multi-turn-rollouts/) · [Parallelism](/tutorials/llm-deep-dive/distributed-training-and-parallelism/)
