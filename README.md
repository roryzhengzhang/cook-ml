<p align="center">
  <img src="./public/favicon.svg" width="64" height="64" alt="">
</p>

<h1 align="center">Cook ML</h1>

<p align="center">
  <a href="https://cook-ml.com/tutorials/">Read the tutorials</a> ·
  <a href="https://cook-ml.com/tutorials/llm-fundamentals/notebook/">Browse the notebook</a> ·
  <a href="./public/notebooks/llm-fundamentals.ipynb">Get the notebook file</a>
</p>

Cook ML makes machine learning easier to understand by connecting worked examples to code you can run. The tutorials start with how language models work, ground the ideas in a storytelling model you train from scratch, and show how larger models change the engineering decisions.

## Tutorials

| Resource | What it covers |
| --- | --- |
| [LLM Fundamentals](https://cook-ml.com/tutorials/llm-fundamentals/) | **Start here.** Nine chapters on how language models represent text, learn from data, generate responses, and are evaluated. |
| [LLM Deep Dive](https://cook-ml.com/tutorials/llm-deep-dive/) | **Go deeper.** Eight chapters on efficient attention, long context, distributed training, adaptation, and serving at scale. |
| [Project notebook](https://cook-ml.com/tutorials/llm-fundamentals/notebook/) | **Build it.** Train and evaluate a small language model for storytelling from scratch in PyTorch. Code examples in Fundamentals link to their notebook cells. |

## Get started locally

```bash
git clone https://github.com/roryzhengzhang/cook-ml.git
cd cook-ml
```

### Run the notebook

Use Python 3.12. The default profile runs the training and evaluation pipeline on CPU with a small offline dataset.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch jupyterlab
COOK_ML_OUTPUT="$PWD/runs/llm-fundamentals" jupyter lab public/notebooks/llm-fundamentals.ipynb
```

In JupyterLab, select **Run → Run All Cells**. To train on TinyStories and TinyStories-Instruct, install the streaming dependencies and launch the `story` profile on one Apple Silicon or NVIDIA GPU. NVIDIA users need a [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/). See the [hardware guide](https://cook-ml.com/tutorials/llm-fundamentals/notebook/#hardware-requirement) for memory and runtime.

```bash
python -m pip install "datasets==3.6.0" "huggingface_hub<1"
COOK_ML_PROFILE=story COOK_ML_OUTPUT="$PWD/runs/llm-fundamentals" \
  jupyter lab public/notebooks/llm-fundamentals.ipynb
```

### Run the website

Use Node.js 24. The checkout includes the published content and assets.

```bash
npm ci
npm run dev
```

Open [localhost:4321](http://localhost:4321). To check a change and preview the production build with search, run the following.

```bash
npm run build
python3 scripts/check_links.py
npm run preview
```

## Contributing

The published chapters are in [`src/content/docs/tutorials/`](src/content/docs/tutorials/), the executable project is in [`public/notebooks/`](public/notebooks/), and the build and content tools are in [`scripts/`](scripts/).

[Open an issue](https://github.com/roryzhengzhang/cook-ml/issues) with a link to the chapter or notebook cell and a short description of what was unclear or how to reproduce the problem.

<details>
<summary>Maintainer workflow</summary>

- **Tutorials.** Edit the corresponding Obsidian notes, then regenerate the published chapters and figures with `npm run import:llm` or `npm run import:deep`. Export updated animation steps with `python3 scripts/export_animation_steps.py` before importing.
- **Notebook examples.** Edit the notebook cell and preserve its stable ID. The site's chapter examples are placed using the cell's `metadata.cookml`.
- **Publishing.** Run `npm run link-previews` after adding external links. Cloudflare Pages builds `main` with `npm run build` and publishes `dist/`.

</details>

## License

The code and documentation are available under the [MIT License](LICENSE). [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) have their own dataset terms.
