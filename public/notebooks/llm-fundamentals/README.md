# Train a story model from scratch

This project builds a byte-level tokenizer and an 11-million-parameter Transformer in PyTorch. Pretraining teaches the model to continue short stories. Supervised fine-tuning teaches it to write a story from a request. Both checkpoints answer the same held-out prompts so their results can be compared.

<a href="/tutorials/llm-fundamentals/#hardware-requirement" target="_blank" rel="noopener">Hardware requirements</a> · <a href="/notebooks/llm-fundamentals.ipynb" download>Download the Jupyter notebook</a>

## Run on CPU

The default profile uses a small generated dataset. It runs the training and evaluation steps offline once PyTorch is installed. The commands assume Python 3.12 and the Python script downloaded using **Download full .py** in the workspace.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python llm-fundamentals.py
```

Jupyter users can install `jupyterlab`, open the downloaded `.ipynb` file, and run all cells from the top.

## Train on TinyStories

The `story` profile streams TinyStories and TinyStories-Instruct and runs on one Apple Silicon or NVIDIA GPU. NVIDIA systems need a [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/).

```bash
python -m pip install "datasets==3.6.0" "huggingface_hub<1"
COOK_ML_PROFILE=story python llm-fundamentals.py
```

The profile selects CUDA when available and otherwise tries Apple MPS. `COOK_ML_DEVICE=cuda` or `COOK_ML_DEVICE=mps` selects the device explicitly. Checkpoints go under `runs/llm-fundamentals/<profile>/`. `COOK_ML_RESUME=1` resumes a matching run, and `COOK_ML_LABS=1` enables optional experiments that train or sample extra models.

## Where to find the code

- `foundations/` covers setup, data, the tokenizer, and the Transformer.
- `base_model/` covers pretraining, sampling, and the key-value cache.
- `post_training/` covers instruction tuning, held-out evaluation, and preference learning.
- `experiments/` includes labs on decoding, model internals, serving, retrieval, and reliability.

The topic files share the state established by earlier notebook cells. The full Python script and Jupyter notebook run those cells in order.
