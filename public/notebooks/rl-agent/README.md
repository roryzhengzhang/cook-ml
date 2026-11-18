# Run the slime GPU pilot

The notebook is the CPU project and the source for the store and verifier. This pilot adds a pretrained model. It uses **Qwen2.5-1.5B**, **slime**, **Megatron**, and **SGLang**. The starting hardware target is one L40S with 48 GB. The GPU run and memory fit are unverified. The Python adapter has been checked against a simulated sampler and the launcher commands have been checked in dry-run mode.

## Prepare the environment

The adapter uses the sample and generation interfaces in [slime revision 79aa41a](https://github.com/THUDM/slime/tree/79aa41a884ab0bf65222c94418f6357d3c31192b). Set up slime with a compatible Megatron, SGLang, Ray, and CUDA installation following the [upstream environment guide](https://thudm.github.io/slime/get_started/quick_start.html). Check the kernel support for the GPU used by the pilot. Start on a dedicated Linux GPU host or inside its own container.

Save `rl-agent.ipynb`, `slime_store.py`, and `run_slime.sh` in the same working directory. The launcher assumes slime is at `/root/slime` and Megatron at `/root/Megatron-LM`. Set `SLIME_ROOT` and `MEGATRON_ROOT` if they are elsewhere.

```sh
python slime_store.py prepare rl-agent.ipynb data
hf download Qwen/Qwen2.5-1.5B --local-dir "$PWD/base-hf"
ray start --head --num-gpus 1
```

Convert the Hugging Face weights from a shell with the model recipe loaded.

```bash
pilot_dir=$PWD
(
  cd "${SLIME_ROOT:-/root/slime}"
  source scripts/models/qwen2.5-1.5B.sh
  PYTHONPATH="${MEGATRON_ROOT:-/root/Megatron-LM}:${PYTHONPATH:-}" \
    python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$pilot_dir/base-hf" \
      --save "$pilot_dir/base-megatron"
)
```

## Follow the training stages

The launcher accepts the stage, task directory, tokenizer/model directory, input checkpoint, and output directory. The optional sixth argument names the task split. Adding `COOK_ML_DRY_RUN=1` prints the resulting slime command.

```sh
# Short integration run with one GRPO rollout batch.
COOK_ML_STEPS=4 bash run_slime.sh sft data base-hf base-megatron smoke-sft
COOK_ML_STEPS=1 bash run_slime.sh grpo data base-hf smoke-sft smoke-grpo

# Main candidate and comparisons.
bash run_slime.sh sft data base-hf base-megatron sft
bash run_slime.sh grpo data base-hf sft grpo
bash run_slime.sh eval data base-hf sft sft-eval validation
bash run_slime.sh eval data base-hf grpo grpo-eval validation

# Start both V2 candidates from the same frozen V1 checkpoint.
bash run_slime.sh grpo data base-hf grpo v2-new v2-train
bash run_slime.sh grpo data base-hf grpo v2-mixed v2-mixed
bash run_slime.sh eval data base-hf v2-new v2-new-eval v2-test
bash run_slime.sh eval data base-hf v2-new v2-new-retention validation
bash run_slime.sh eval data base-hf v2-mixed v2-mixed-eval v2-test
bash run_slime.sh eval data base-hf v2-mixed v2-mixed-retention validation
```

The shorter runs check that the sampler, verifier, and trainer exchange data. A successful learning run also needs mixed rewards within a meaningful fraction of groups. The recorded task metadata includes the state and tool trace, so the pilot can inspect which mistakes rule out learning. For a completed run, record the device, software image, seed, peak memory, elapsed time, and held-out scores. The notebook's CPU results illustrate the format.

GRPO normalizes each group of four attempts and clips the token probability ratio. The reference launch uses temperature 1 with top-p 1 and the sampled log probabilities returned by SGLang. The loss mask covers sampled model tokens. Tool results are appended as context. The parser recognizes a small line-based action vocabulary, and the simulator enforces the exchange rules before a write can change state.

## Train with reasoning before actions

The thinking variant starts from the same Qwen2.5 base checkpoint. Its supervised demonstrations put a short rationale grounded in the latest observation before each command. At rollout time the model generates a free-form reasoning line and then a separate command. The harness reserves up to 32 tokens for the reasoning line and 24 for the command. If the reasoning line hits its cap, the model still gets a chance to act. The store receives only the command.

The trainer uses the original sampled token IDs and log probabilities for both spans. The harness supplies the role labels and observations as masked context. The final outcome gives the same group-relative advantage to the thought and command tokens in an attempt.

```sh
python slime_store.py prepare rl-agent.ipynb data-thinking --thinking
bash run_slime.sh sft data-thinking base-hf base-megatron thinking-sft
COOK_ML_RESPONSE_TOKENS=768 bash run_slime.sh grpo data-thinking base-hf thinking-sft thinking-grpo
COOK_ML_RESPONSE_TOKENS=768 bash run_slime.sh eval data-thinking base-hf thinking-grpo thinking-eval validation
COOK_ML_RESPONSE_TOKENS=768 bash run_slime.sh eval data base-hf grpo direct-eval validation
```

These commands compare the two training pipelines. For a separate test of reasoning at answer time, evaluate a checkpoint that has learned both formats on each validation split with the same response ceiling, tool limits, sampling settings, and seeds. The response ceiling counts both sampled tokens and inserted context. The evaluation saves rollout data in its output directory. Each episode records `thinking_tokens`, `action_tokens`, and `thinking_truncations` alongside the tool trace and store state. Compare verifier success, invalid calls, budget exhaustion, and generated tokens per successful exchange. Include latency measured on the serving host. The adapter and CPU exercise can be checked locally. Free-form model training needs a GPU run to establish quality and cost.
