# Cook ML

Source for [cook-ml.com](https://cook-ml.com): machine learning tutorials and a blog, built with [Astro](https://astro.build) and [Starlight](https://starlight.astro.build).

## Run locally

```bash
npm install
npm run dev        # http://localhost:4321
npm run build      # static site in dist/, including the search index
npm run preview    # serve dist/ (search works here, not in dev)
python3 scripts/check_links.py   # after a build: broken anchors and KaTeX errors
```

Hovering a link in tutorial text opens a preview card (`public/link-preview.js`, styled in `src/styles/link-preview.css`). Links between pages are previewed from the target page itself. External links read `public/link-previews.json`, which `npm run link-previews` fills in for links that have no preview yet (arXiv papers through the arXiv API, other pages from their title, description and image tags). Run it after an import that adds external links; `npm run link-previews -- --all` refreshes every entry.

## Layout

| Path | What it holds |
|---|---|
| `src/content/docs/index.mdx` | Home page |
| `src/content/docs/tutorials/` | Tutorial series, one folder per series |
| `src/content/docs/blog/` | Blog posts (frontmatter needs `date`; optional `tags`, `excerpt`, `authors`) |
| `src/content/docs/about.md` | About page |
| `src/components/` | Header with section tabs, footer, and home page lists |
| `src/styles/custom.css` | Theme: colors, type, sidebar, figures, tables, math |
| `public/images/<series>/` | Tutorial figures copied from Obsidian |
| `public/images/steps/`, `src/data/animation-steps/` | Animation steps exported by `scripts/export_animation_steps.py` |
| `scripts/import-obsidian.mjs` | Obsidian to site converter |

## Updating LLM Fundamentals from Obsidian

The tutorial pages are generated from the Obsidian folder `Machine Learning Sheet/15 LLM Fundamentals`. Edit the notes in Obsidian, then run:

```bash
npm run import:llm
```

Animations are shown as numbered step-by-step figures rather than videos. Each step is one frame of the animation with its caption as real text. To regenerate the steps after the animation scripts in `~/.claude/scripts/ml-figures/<id>-gifs.py` change, run this before the import:

```bash
python3 scripts/export_animation_steps.py        # all 15x and 16x animations, or pass ids such as 15b 16c
```

It loads each script without touching the vault, saves every step as a PNG in `public/images/steps/<animation>/`, and writes the captions to `src/data/animation-steps/<animation>.json`. Frames that share a caption are merged into one step, keeping the final frame. The importer then replaces each `![[...mp4]]` embed that has step data with the step figure, and falls back to a looping video when it has none.

Collapsible "deep dive" cards come from collapsed Obsidian callouts. Put the body of a section in a `> [!deep-dive]-` callout right under its heading, and the site shows the heading as the card title with the body collapsed:

```md
## 8. Bits per byte across tokenizers

> [!deep-dive]-
> Comparing models is one of the most common reasons...
```

A `##` heading stays a real section heading inside the card, so the sidebar and links keep working. A callout with its own title (`> [!deep-dive]- Some title`) becomes a card with that title.

The tutorial sidebar is built by `scripts/series-sidebar.mjs`. Each chapter is a collapsible group listing its `##` sections. The Project notebook group also lists its subsections and the stages generated from the notebook. The sidebar is read when the dev server starts, so restart `npm run dev` after an import that changes headings.

The script regenerates `src/content/docs/tutorials/llm-fundamentals/` and `public/images/llm-fundamentals/`, so do not hand-edit those files. It converts `![[figure]]` embeds into images, links between chapters into site links, and links to unpublished notes into plain text.

### LLM Fundamentals project notebook

`public/notebooks/llm-fundamentals.ipynb` contains the runnable project and the source for the site’s codepads. An exposed Python cell has a stable Jupyter cell ID and a `metadata.cookml` object with its title, summary, chapter slug and section heading. The Markdown renderer inserts the cell beside that section and links it to the same cell in the web view. Leave the chapter and section out of the metadata for notebook-only cells. The Obsidian import can regenerate every chapter without rewriting the notebook or copying Python into Markdown. The project page `src/content/docs/story-model-notebook.md` holds the background, hardware and run instructions, and the renderer replaces its `<!-- notebook -->` marker with every cell in one notebook frame. Each markdown cell becomes the explanation above the code cell that follows it, and stage headings group the cells.

Use Python 3.12 and install the notebook tools with `python -m pip install torch jupyterlab`. The default CPU profile trains on an offline fixture. For the public-data run, install `datasets==3.6.0 "huggingface_hub<1"` as well and start Jupyter with `COOK_ML_PROFILE=story jupyter lab`. The story profile streams TinyStories for pretraining and [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) for supervised fine-tuning. It chooses CUDA when available or Apple MPS. Set `COOK_ML_DEVICE=cuda` or `COOK_ML_DEVICE=mps` to choose explicitly. The story profile runs 8,000 pretraining updates and 3,500 fine-tuning updates. The notebook saves separate `base.pt` and `sft-instruct.pt` checkpoints. `COOK_ML_RESUME=1` resumes a matching run, and `COOK_ML_LABS=1` enables the optional experiments.

LLM Deep Dive, the advanced follow-on series, comes from `Machine Learning Sheet/16 LLM Deep Dive` the same way:

```bash
npm run import:deep
```

Links between published series (`PUBLISHED` in `scripts/import-obsidian.mjs`) become site links, so a Deep Dive chapter can link a Fundamentals section. Each tutorial page shows only its own series in the sidebar (`src/routeData.ts`).

To import another series, pass the folder and a slug, then add a sidebar group in `astro.config.mjs` and a card in `src/components/SeriesGrid.astro`:

```bash
node scripts/import-obsidian.mjs --src "<vault>/06 RAG" --series rag --title "Retrieval and RAG"
```

## Writing a blog post

Add `src/content/docs/blog/<slug>.md`:

```md
---
title: My post
date: 2026-12-01T12:00:00Z
authors: zheng
tags: [llm]
excerpt: One sentence for the post list.
---
```

## Deploying to cook-ml.com

The site is hosted on Cloudflare Pages, connected to this GitHub repository. Every push to `main` builds and publishes the site, and every pull request gets its own preview URL. `.node-version` pins the Node version used by the Cloudflare build.

One-time setup in the Cloudflare dashboard:

1. Go to **Workers & Pages → Create → Pages → Connect to Git**, authorize GitHub, and pick this repository.
2. Set **Framework preset** to **Astro**, **Build command** to `npm run build`, and **Build output directory** to `dist`. Keep the production branch as `main`.
3. After the first deploy, open the project's **Custom domains** tab and add `cook-ml.com` and `www.cook-ml.com`. If the domain's DNS is on Cloudflare, the records are created for you. Otherwise, follow the CNAME instructions shown there.
