---
title: "Agent and Environment"
description: "Build a store that can execute actions, isolate attempts, and replay an episode."
sidebar:
  order: 1
  label: "Agent and Environment"
lastUpdated: 2026-11-18
---

The first inventory lookup for order O-17 times out. The next lookup returns one medium shirt. A customer receives the requested size only if the system treats the timeout as a reason to retry and completes the exchange after that check. A successful run has a physical result in miniature. The order goes from S to M, the stock count goes from 1 to 0, and the store records one exchange.

An **agent** is a program that reads a request and chooses a series of actions to carry it out. In this project, a language model makes the choices and a small Python program runs the surrounding loop. An **environment** holds the store records and implements how each action affects them. Keeping the store in a simulator makes it possible to inspect exactly what changed after a run, and to give another attempt the same starting conditions.

![Order exchange episode](/images/rl-for-llm-agents/17a-order-exchange-episode.png)

The chapter begins with the order and inventory records because they determine what the agent can accomplish. It then follows the messages across the agent loop and records the complete attempt. The trace makes reset and replay meaningful. Sandbox isolation carries that same idea to settings where multiple agents run concurrently or execute code.

## State and observations

The first observation says less than the simulator's order table. That gap determines which tools the policy needs.

The store knows how many medium shirts are available before the agent learns the answer. Its **state** is the collection of facts the simulator needs to determine the result of the next action. The initial state for O-17 includes a paid order for a size S shirt, an exchange eligibility rule, inventory of 1 in size M, and zero recorded exchanges. The simulator also tracks which stock calls have been made because the fixture injects a timeout on the first one.

The customer's message gives the agent an order identifier and a requested size. The **observation** is information the environment reveals to the agent. The first lookup can tell it the order is paid and the current shirt size is S. It learns whether M is available only after a successful stock call. Reward calculation can read the full simulator state later because it has a different job from the policy that chooses actions.

![State and observations](/images/rl-for-llm-agents/17a-state-and-observations.png)

A **policy** chooses an action using the information the agent has. A scripted policy follows explicit rules. A language model policy assigns probabilities to continuations and may sample a tool call or a final response. With a limited observation, the model can use **history**, the sequence of previous actions and results, to distinguish an initial stock request from a retry.

The environment has separate reasons for retaining each fact. Order size determines what will change. Inventory determines whether it can change. The fault counter determines what a read will reveal. A cache in the agent can hold an earlier observation, but the simulator still uses current inventory when it processes the next call.

**Partial observability** means the information at the time of a decision can fit several different world states. The same timeout message could come from an inventory table with one unit or zero. The agent can live with that ambiguity, ask the store again, or end the request with an accurate explanation. The actual next action depends on the budget and the policy for the store, both facts the agent can use.

<details class="cm-collapse" id="the-markov-property">
<summary><span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span><span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span><span class="cm-collapse-title">The Markov property</span></span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>

A Markov decision process describes a world where the current state and chosen action determine the distribution of the next state. The simulator's call counter belongs in that state. With a counter of 0, checking stock returns a timeout and advances the counter to 1. With a counter of 1 on the same fixture, checking stock reads the inventory. Omitting the counter would make two seemingly identical states produce different observations. The agent sees only part of this state, so its view is partially observable even with a fully specified simulator.

</details>

There can be more than one uncertainty. The store may know which unit is on hand, while the agent is still unsure which size the customer wants. That is a different task fixture from O-17, where the request is explicit. A clarification turn belongs in the first fixture because part of its goal is learning what to act on.

## Transitions

An **action** is a choice the environment can receive and process. Looking up an order, checking a policy, and checking stock are read actions. The exchange submission is a write action. A **transition** is the change in state caused by an action. Reading stock changes the simulated call counter. Submitting a valid exchange also changes order and inventory records.

| Tool | Decision the result supports | Store state affected |
|---|---|---|
| `lookup_order` | Which order and size are on file | Order and inventory are read |
| `check_policy` | Whether the exchange is allowed | Policy is read |
| `check_stock` | Whether the requested size is available | Call counter advances |
| `submit_exchange` | Whether the store accepted the change | Order, inventory, and exchange record can change |

A **tool schema** gives a function's expected argument names and types. Every submission needs a matching order and a target size. A general stock request also names the requested size. A general store interface can reject a missing size before it touches a record. The pilot uses one order and one target size per session, so the harness already knows the arguments. Its action format uses a bare tool name. The underlying write function still checks the order and target size. Store rules still apply to well-formed calls. A correctly formatted submission for an ineligible order must be rejected by the write operation.

Tool errors also differ in what should happen next. A bad size is a validation error the agent can correct by supplying a valid value. A timeout on a read leaves the state of the order known. A lost response from a write creates a new question about whether the write already took effect. In a production service, checking the order state or passing an idempotency key, a stable identifier for retries of the same operation, can prevent a duplicate exchange.

The submission checks its preconditions against current state. A prior stock check could return 1, and a sale could claim the unit before the exchange commits. The write therefore checks inventory again at the point where it updates the order. In this simulation the eligible write advances all its related records together. In a store backed by a database, a transaction serves the same purpose.

With a successful transition, stock changes by $0-1=-1$ and the number of exchange records changes by $1-0=1$. With a read timeout, both differences are 0. Those two arithmetic checks are more informative than a success phrase in the model's reply because the underlying values come from the store.

![Exchange state changes](/images/rl-for-llm-agents/17a-exchange-state-changes.png)

The explicit state effect is what makes a success test possible. A response that claims the order changed could follow an error or no write at all. The verifier can read the order, inventory delta, and record count together. [The verifier section](/tutorials/rl-for-llm-agents/tasks-and-verification/#verifiers) develops those checks.

## The agent loop

The **agent loop** connects the model to the store. It gives the policy the current context, checks the returned action against the tool schema, calls the tool, and appends its result to the context. The policy now sees enough to make another decision. The loop can stop after an assistant response or when its execution budget is exhausted.

In the O-17 run, the first policy choice looks up the order and the next checks eligibility. The agent asks for M and sees a timeout. The next model call can choose to retry based on the error message and the earlier conversation. After the successful stock response the agent submits the exchange. The store acknowledges the update before a final customer reply.

A **harness** is the code around the model that manages these turns, the available tools, and the stop rules. It gives each incoming turn a role and validates the response before execution. The model may propose a call, but the harness maps that proposal to one real operation in the environment. A reply such as “I completed the exchange” does not run the submission tool.

![Agent loop messages](/images/rl-for-llm-agents/17a-agent-loop-messages.png)

The simulator returns a typed timeout so the model can distinguish it from an available count of 0. That separation also helps evaluation. A trace can say whether execution hit a transient fault, the agent misread a valid result, or the state changed unexpectedly. Text fields returned by tools are data even when the tool call and its status came from a trusted part of the harness.

The harness can preserve the call identifier alongside the observation to connect each response to its action. That matters if the agent sends more than one call in a turn. Sequence numbers also keep a repeated stock response from looking like a fresh update. The notebook's shortest useful harness can make one call per turn, and the same trace vocabulary works when calls run in parallel.

A **step budget** limits how many decisions can be made in one attempt. It bounds the cost of repeated retries and supplies a clear stop for a policy that loops. The O-17 path needs five tool calls when the first stock call times out. A four-call budget would deny this recovery even for a policy that chose the right calls, so the budget is part of the task conditions.

The step limit also separates actions from model invocations. The timeout case has five tool calls. A main loop that asks the model whether to act or reply makes one additional invocation to produce the final answer. A token budget and a tool-call budget therefore bound different sources of cost.

## Episodes and traces

One reset followed by one bounded attempt is an **episode**. Its **trajectory** is the ordered sequence of actions, observations, and any rewards produced along the way. A **rollout** collects a trajectory by running a particular policy in the environment. The same task can yield several trajectories because the policy may make different choices on each attempt.

The record can distinguish raw model candidates from calls that were accepted. A malformed action may still be valuable training evidence because the policy produced it. An accepted tool call supplies a verified event about what ran. Keeping both prevents a replay from silently upgrading a proposal into a state change.

A **trace** is the record engineers use to inspect what happened. For a tool call, it can store the turn number, action, validated arguments, tool result, and whether the call changed state. An execution trace also makes the attempt reproducible when it links back to the environment version and task fixture. Later training traces carry additional information, such as the model tokens sampled for each decision.

A **policy version** identifies the weights and configuration that chose those actions. Keeping it on a trace becomes essential once two checkpoints are generating data at the same time. The store can still reproduce a scripted action replay when the model has changed because the recorded actions are the inputs to that replay.

For O-17, all normal events after reset can be described without looking at a model's internal state. The call counter changes from 0 to 1 at the failed lookup. Inventory stays at 1 through both reads. The write brings inventory to 0 and adds one exchange event. A replay can check those transitions individually before it checks the overall outcome.

**Termination** means the episode reached a defined end state such as a final response or a handled refusal. **Truncation** means an outside limit cut it short, such as the allowed number of turns. A task can terminate unsuccessfully when the agent gives up after a timeout. Truncation is a separate reason to stop, and keeping both reasons makes it possible to count timeouts without mistaking them for decisions to stop.

The same outcome code can accompany different stop reasons. A responsible handoff after exhausting a configured retry rule may be a supported final outcome in one fixture. A truncated episode may leave the user with no usable response. For this reason a trace preserves the stop event before the verifier maps the complete attempt to success or failure.

An **outcome reward** is a score attached to the result of the complete attempt. A terminal score can identify the O-17 episode as successful. The trace is needed to find which earlier decisions enabled it. This is why [Policy Gradients](/tutorials/rl-for-llm-agents/policy-gradients/) eventually has to connect an outcome back to the actions that helped produce it.

## Reset and replay

To repeat O-17, we need to recover the same request, stock count, and planned tool failure. A **task fixture** records those starting conditions together so an attempt can begin from them again. A **reset** instantiates fresh mutable state from that fixture. All attempts in a group can start with one available medium and their own zero-valued stock call counter.

Reset must copy mutable records as well as scalar fields. A fresh order record pointing to the same inventory dictionary as an old attempt still shares stock. The same problem applies to an exchange log or an error injector. Keeping the original fixture immutable allows the same input to be reused in training, testing, and evaluation.

A **seed** specifies a starting point for a pseudorandom sequence. If the simulator draws a fault schedule, storing the seed and simulator version helps reproduce those faults. Exact replay can instead log the realized sequence of observations. When the environment runs against external services, seed alone cannot reproduce changes made outside the simulator.

An **action replay** sends a recorded sequence of actions to fresh state. In O-17, a replay of the scripted exchange must see a timeout on the third call, an available medium on the fourth, and the same order and inventory at the end. This exercise checks the environment before comparing model policies. A replay of model generation has more moving parts, including the sampled tokens and the model checkpoint.

A replay has a stronger and a weaker form. The stronger test compares each observation and state delta in sequence. The weaker test only compares the final store state. Two scripts can both reach M while one wastes a call or triggers an avoidable error, so checking individual steps helps when debugging the environment itself.

A useful isolation test interleaves two copies of the same task. Both should observe the first-call timeout. If one copy completes first, the other should still have one unit in stock in its private store. Running the test in both possible orders catches state accidentally stored in a module-level variable.

The seeded O-17 run is also an inexpensive **smoke test**. That is a short check of the main path before a larger run. An exact failure here suggests an issue with the environment or harness. A mismatch on the same fixture only after sampling from a model calls for inspecting the generation and tool parsing stages as well.

## Sandbox isolation

The same exchange tool that changes a simulated order could change a customer’s order if it pointed at the live store. A **sandbox** limits where the agent and its tools can act. In the store project an agent can call only typed operations on a disposable copy of the store. Each rollout gets its own state. Both conditions matter. A caller with access to a shared database could affect real records even if its local process were new.

![Sandbox state](/images/rl-for-llm-agents/17a-sandbox-state.png)

An isolated state instance is enough for the notebook's trusted Python functions. Packaging the simulator as a Docker image also pins the files and dependencies that form the environment. [OpenEnv](https://huggingface.co/docs/openenv/index) gives the trainer a standard interface for resetting and stepping that image. It keeps the store implementation usable by another trainer that speaks the same environment protocol.

An agent allowed to execute code needs an execution boundary around that code. A managed sandbox such as [Modal Sandboxes](https://modal.com/docs/guide/sandbox-networking) or [E2B](https://docs.e2b.dev/network/internet-access) gives each session its own workspace and exposes controls for outbound connections. An extension can compare the ordinary store tool loop with a sandboxed session and measure startup, reset, and cleanup time before choosing a deployment layout.

The scope of permissions matters as much as the runtime choice. A sandbox can receive a job identifier and read-only reference material. The simulator can issue per-rollout write access to its disposable order table. When a test needs access to a real external service, an explicit outbound allowlist sets which connections are even possible.

The trainer and reward grader need their own trusted inputs. Full credentials belong outside the rollout workspace, and the reward can read the resulting store state through a dedicated verifier. If a rollout could replace its own reward record, raising the reported reward could become easier than completing an exchange.

Parallel rollout workers also have a lifecycle. Each session starts with a clean snapshot, retains its own state across turns, and releases its resources when it stops. A timeout at the infrastructure layer belongs in execution diagnostics. A timeout injected by the task stays part of the episode the policy is learning to handle.

## Environment interface

The training code uses two operations to interact with the store. A single reset produces the starting observation and a fresh store. A step consumes a validated action and returns the next observation and stop information. A verifier can then examine the finished attempt. Each operation has an output that can be checked before training starts.

In the [O-17 replay](/tutorials/rl-for-llm-agents/notebook/#store-episode), creating a new store from the fixture restores size S, one medium and zero stock calls. The first call to `check_stock` returns a timeout. The second returns the count because the call counter has advanced, while the shirt and inventory stay as they were.

The environment is ready for the next chapter when the same input yields a consistent replay and concurrent attempts stay independent. These checks establish what the training code can rely on from the store.

1. The fixture defines O-17, its size S shirt, a requested M, and a stock count of 1. The environment starts each attempt by building a fresh store from those facts.
2. The tool schema accepts well-formed calls, and store rules govern the associated transitions. The first stock check times out and the second reports one available medium.
3. The scripted policy completes one exchange within the decision budget. The final state records size M, stock 0, and one exchange event.
4. A second run replays the same calls and reaches the same outcome. A pair of interleaved runs completes independently from separate state.
5. The trace preserves the request, tool calls, observed results, and stopping reason. It gives the next chapter concrete evidence for deciding what should earn a reward.

---

**Related**: [Tasks and Verification](/tutorials/rl-for-llm-agents/tasks-and-verification/) · [LLM Applications](/tutorials/llm-fundamentals/using-llms-in-applications/)
