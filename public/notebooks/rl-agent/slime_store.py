"""Store tasks and a slime adapter for the optional language-model pilot.

Make the fixture and verifier available to workers by extracting the notebook's
marked cells with `python slime_store.py prepare path/to/rl-agent.ipynb path/to/data`.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

ACTIONS = ("lookup_order", "check_policy", "check_stock", "submit_exchange")
RESOLUTIONS = ("exchanged", "out_of_stock", "ineligible", "escalate", "unresolved")
TURN_LIMIT = 10
ACTION_TOKENS = 24
THINKING_TOKENS = 32


def prompt_for(task, *, thinking=False):
    instruction = (
        "Before each action, write one short line about the latest evidence. "
        if thinking else ""
    )
    first_prefix = "Thought: " if thinking else "Action: "
    return (
        f"The customer asks to exchange order {task.order_id} from size S to size M. "
        "Look up the order, check eligibility, and check stock if eligible. "
        "Try the stock check again if it times out, up to twice. "
        "Escalate if the policy tool says to. Submit exactly one eligible exchange "
        "when stock is available. Each action must occupy a single line. "
        f"{instruction}"
        f"Use one of {', '.join(ACTIONS)} or 'finish' followed by one of "
        f"{', '.join(RESOLUTIONS)}.\n{first_prefix}"
    )


def apply_action(store, action):
    command, _, argument = action.strip().partition(" ")
    if command == "finish":
        return None, argument if argument in RESOLUTIONS else "unresolved"
    if argument:
        return {"status": "invalid_action"}, None
    if command == "submit_exchange":
        return store.submit_exchange(store.task.order_id, "M"), None
    readers = {
        "lookup_order": store.lookup_order,
        "check_policy": store.check_policy,
        "check_stock": store.check_stock,
    }
    if command in readers:
        return readers[command](), None
    return {"status": "invalid_action"}, None


def observation_text(observation, *, thinking=False):
    next_prefix = "Thought: " if thinking else "Action: "
    return f"\nObservation: {json.dumps(observation, sort_keys=True)}\n{next_prefix}"


def thinking_example(action, observation):
    if action == "check_stock" and observation.get("status") == "timeout":
        return "The last lookup timed out. It did not return a stock count."
    return {
        "lookup_order": "I need to confirm which order the customer wants to exchange.",
        "check_policy": "The order exists. I need to check whether it can be exchanged.",
        "check_stock": "The policy allows the exchange. I need a stock count before submitting.",
        "submit_exchange": "The stock count confirms a medium is available.",
        "finish exchanged": "The store confirmed that the exchange succeeded.",
        "finish out_of_stock": "The inventory tool confirmed that no medium is available.",
        "finish ineligible": "The policy tool says the order is ineligible.",
        "finish escalate": "The policy tool says the reported defect needs escalation.",
    }[action]


def action_prefix(thought):
    return "Action: " if thought.endswith("\n") else "\nAction: "


def demonstration(task, store_type):
    store = store_type(task)
    actions = ["lookup_order", "check_policy"]
    policy = store.check_policy()
    if policy["escalate"]:
        return actions + ["finish escalate"]
    if not policy["eligible"]:
        return actions + ["finish ineligible"]
    actions += ["check_stock"] * (task.stock_timeouts + 1)
    if task.stock:
        return actions + ["submit_exchange", "finish exchanged"]
    return actions + ["finish out_of_stock"]


async def generate(args, sample, sampling_params, evaluation=False):
    from store_core import Store, Task
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post

    if args.partial_rollout:
        raise ValueError("This pilot uses fresh, complete episodes")
    tokenizer = GenerateState(args).tokenizer
    store = Store(Task(**sample.metadata["task"]))
    thinking = sample.metadata.get("thinking", False)
    sample.tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    transcript = []
    resolution = "unresolved"
    reason = "turn_budget"
    thinking_tokens = action_tokens = thinking_truncations = 0
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    params = {**sampling_params, "stop": ["\n"], "no_stop_trim": True}

    def remaining():
        return params["max_new_tokens"] - sample.response_length

    def append_context(text):
        sample.append_response_tokens(
            args, tokens=tokenizer.encode(text, add_special_tokens=False),
            trainable=False, text=text, update_terminal_info=False,
        )

    async def sample_line(token_limit):
        output = await post(url, {
            "input_ids": sample.tokens,
            "sampling_params": {**params, "max_new_tokens": token_limit},
            "return_logprob": True,
        })
        meta = output["meta_info"]
        finish = meta["finish_reason"]["type"]
        if finish == "abort":
            return "", finish, 0
        token_logps = meta["output_token_logprobs"]
        if not token_logps:
            raise RuntimeError("Sampler returned no token IDs or log probabilities")
        sample.append_response_tokens(
            args, tokens=[entry[1] for entry in token_logps],
            log_probs=[entry[0] for entry in token_logps],
            meta_info=meta, text=output["text"], update_terminal_info=False,
        )
        return output["text"], finish, len(token_logps)

    action_prefix_budget = max(
        len(tokenizer.encode(action_prefix(text), add_special_tokens=False))
        for text in ("", "\n")
    ) if thinking else 0
    for _ in range(TURN_LIMIT):
        if thinking:
            # Reserve room for the harness label and the complete action.
            thought_budget = min(THINKING_TOKENS, remaining() - action_prefix_budget - ACTION_TOKENS)
            if thought_budget <= 0:
                reason = "response_budget"
                break
            thought, finish, count = await sample_line(thought_budget)
            if finish == "abort":
                sample.status = sample.Status.ABORTED
                return sample
            thinking_tokens += count
            thinking_truncations += int(finish == "length")
            append_context(action_prefix(thought))

        if remaining() <= 0:
            reason = "response_budget"
            break
        line, finish, count = await sample_line(min(ACTION_TOKENS, remaining()))
        if finish == "abort":
            sample.status = sample.Status.ABORTED
            return sample
        action_tokens += count
        if finish == "length":
            reason = "action_budget"
            break
        action = line.strip()
        observation, terminal = apply_action(store, action)
        transcript.append({"action": action, "observation": observation})
        if terminal is not None:
            resolution, reason = terminal, "finished"
            break
        context = observation_text(observation, thinking=thinking)
        if thinking and len(tokenizer.encode(context, add_special_tokens=False)) > remaining():
            reason = "response_budget"
            break
        append_context(context)

    sample.metadata["episode"] = {
        "resolution": resolution, "reason": reason,
        "state": asdict(store.state), "tool_trace": transcript,
        "thinking_tokens": thinking_tokens, "action_tokens": action_tokens,
        "thinking_truncations": thinking_truncations,
    }
    sample.status = sample.Status.COMPLETED if reason == "finished" else sample.Status.TRUNCATED
    return sample


async def reward(args, sample):
    from store_core import Snapshot, Task, score

    episode = sample.metadata["episode"]
    return score(
        Task(**sample.metadata["task"]), episode["resolution"], Snapshot(**episode["state"])
    )


def generate_sft(args, rollout_id, data_buffer, evaluation=False):
    from store_core import Store, Task
    from slime.utils.processing_utils import load_tokenizer

    if evaluation:
        raise ValueError("Use the agent rollout hook for evaluation")
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=False)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    for group in groups:
        (sample,) = group
        task = Task(**sample.metadata["task"])
        store = Store(task)
        thinking = sample.metadata.get("thinking", False)
        tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
        mask = []
        observation = {}

        def add_text(text, *, trainable):
            ids = tokenizer.encode(text, add_special_tokens=False)
            tokens.extend(ids)
            mask.extend([int(trainable)] * len(ids))

        for action in demonstration(task, Store):
            if thinking:
                thought = thinking_example(action, observation) + "\n"
                add_text(thought, trainable=True)
                add_text(action_prefix(thought), trainable=False)
            add_text(action + "\n", trainable=True)
            observation, terminal = apply_action(store, action)
            if terminal is None:
                add_text(observation_text(observation, thinking=thinking), trainable=False)
        sample.tokens = tokens
        sample.loss_mask = mask
        sample.response_length = len(mask)
        sample.reward = 0.0
        sample.status = sample.Status.COMPLETED
    return groups


def dataset(Task, split):
    validation = split in {"validation", "v2-test"}
    if split == "v2-train" or split == "v2-test":
        stocks = (2, 4) if validation else (0, 1, 3)
        cases = [
            dict(final_sale=True, stock=stock, defect_reported=defect)
            for stock in stocks for defect in (True, False)
        ]
    else:
        stocks = (2, 4) if validation else (0, 1, 3)
        cases = [
            dict(stock=stock, eligible=eligible, stock_timeouts=faults)
            for stock in stocks for eligible in (True, False) for faults in (0, 1, 2)
        ]
        if validation:
            cases.extend([dict(stock=0, stock_timeouts=1), dict(stock=0, stock_timeouts=2)])
        else:
            cases = [case for case in cases if case["stock"] or not case["stock_timeouts"]]
    return [Task(f"{split}-{index:02d}", **case) for index, case in enumerate(cases)]


def prepare(notebook_path, destination, *, thinking=False):
    destination.mkdir(parents=True, exist_ok=True)
    cells = json.loads(notebook_path.read_text())["cells"]
    source = "\n\n".join(
        "".join(cell["source"]) for cell in cells if "store-core" in cell["metadata"].get("tags", [])
    )
    if not source:
        raise ValueError("No environment cells found in the notebook")
    core = destination / "store_core.py"
    core.write_text(source + "\n")

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("store_core", core)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    old = dataset(module.Task, "train")
    new = dataset(module.Task, "v2-train")
    splits = {
        "train": old,
        "validation": dataset(module.Task, "validation"),
        "v2-train": new,
        "v2-mixed": new + old[:len(new)],
        "v2-test": dataset(module.Task, "v2-test"),
    }
    for name, tasks in splits.items():
        with (destination / f"{name}.jsonl").open("w") as rows:
            for task in tasks:
                metadata = {"task": asdict(task)}
                if thinking:
                    metadata["thinking"] = True
                rows.write(json.dumps({"prompt": prompt_for(task, thinking=thinking), "metadata": metadata}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--thinking", action="store_true", help="Add a reasoning turn before each action")
    options = parser.parse_args()
    prepare(options.notebook, options.destination, thinking=options.thinking)
