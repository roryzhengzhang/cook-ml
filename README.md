# Cook ML

Source for [cook-ml.com](https://cook-ml.com): machine learning tutorials and a blog, built with [Astro](https://astro.build) and [Starlight](https://starlight.astro.build).

## Run locally

```bash
npm install
npm run dev        # http://localhost:4321
npm run build      # static site in dist/, including the search index
npm run preview    # serve dist/ (search works here, not in dev)
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
| `scripts/import-obsidian.mjs` | Obsidian to site converter |

## Updating LLM Fundamentals from Obsidian

The tutorial pages are generated from the Obsidian folder `Machine Learning Sheet/15 LLM Fundamentals`. Edit the notes in Obsidian, then run:

```bash
npm run import:llm
```

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
