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
python3 scripts/export_animation_steps.py        # all 15x animations, or pass ids such as 15b 15c
```

It loads each script without touching the vault, saves every step as a PNG in `public/images/steps/<animation>/`, and writes the captions to `src/data/animation-steps/<animation>.json`. Frames that share a caption are merged into one step, keeping the final frame. The importer then replaces each `![[...mp4]]` embed that has step data with the step figure, and falls back to a looping video when it has none.

Collapsible "deep dive" cards come from collapsed Obsidian callouts. Put the body of a section in a `> [!deep-dive]-` callout right under its heading, and the site shows the heading as the card title with the body collapsed:

```md
## 8. Bits per byte across tokenizers

> [!deep-dive]-
> Comparing models is one of the most common reasons...
```

A `##` heading stays a real section heading inside the card, so the sidebar and links keep working. A callout with its own title (`> [!deep-dive]- Some title`) becomes a card with that title.

The tutorial sidebar is built from the chapter files by `scripts/series-sidebar.mjs`: each chapter is a collapsible group listing its `##` sections. It is read when the dev server starts, so restart `npm run dev` after an import that changes headings.

The script regenerates `src/content/docs/tutorials/llm-fundamentals/` and `public/images/llm-fundamentals/`, so do not hand-edit those files. It converts `![[figure]]` embeds into images, links between chapters into site links, and links to unpublished notes into plain text.

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
