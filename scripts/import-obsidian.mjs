#!/usr/bin/env node
// Import an Obsidian tutorial series into the Starlight content folder.
//
// Usage:
//   npm run import:llm
//   node scripts/import-obsidian.mjs --src <series folder> --figures <figures folder> --series llm-fundamentals
//
// What it does:
//   - Turns each note into a Markdown page with Starlight frontmatter.
//   - Copies `![[figure.png]]` embeds into public/images/<series>/ and rewrites them as images.
//   - Rewrites `[[Note#Heading|alias]]` links to pages of any published series as site links.
//     Links to notes outside the series become plain text, since those notes are not published.
//   - Drops Obsidian-only lines (tag line, Parent link, Last updated).

import fs from 'node:fs';
import path from 'node:path';
import GithubSlugger from 'github-slugger';
import katex from 'katex';

const VAULT =
  '/Users/zhengzhang/Library/CloudStorage/GoogleDrive-roryzhang95@gmail.com/My Drive/Obsidian/Personal/Machine Learning Sheet';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => {
    if (arg.startsWith('--')) pairs.push([arg.slice(2), all[i + 1]]);
    return pairs;
  }, []),
);

const SRC = args.src ?? path.join(VAULT, '15 LLM Fundamentals');
const FIGURES = args.figures ?? path.join(VAULT, 'figures');
const SERIES = args.series ?? 'llm-fundamentals';
const SERIES_TITLE = args.title ?? 'LLM Fundamentals';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const OUT_DOCS = path.join(ROOT, 'src/content/docs/tutorials', SERIES);
const OUT_IMAGES = path.join(ROOT, 'public/images', SERIES);

const slugify = (s) =>
  s
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

// "15a Language Modeling and Next Token Prediction" -> { code: "15a", title: "Language Modeling ..." }
function parseNoteName(name) {
  const m = name.match(/^(\d+[a-z]?)\s+(.*)$/);
  return m ? { code: m[1], title: m[2] } : { code: '', title: name };
}

// Short chapter names, used for page titles, the sidebar, Previous/Next and links that name a chapter.
// URLs keep the slug of the full note name, so existing links do not change.
const SHORT_TITLES = {
  '15a': 'Language Modeling',
  '15b': 'Tokenization',
  '15c': 'Attention',
  '15d': 'Transformer Block',
  '15e': 'Pretraining',
  '15f': 'Post-Training',
  '15g': 'Inference',
  '15h': 'LLM Applications',
  '15i': 'Evaluation and Safety',
  '16a': 'Efficient Attention',
  '16b': 'Long Context',
  '16c': 'Parallelism',
  '16d': 'Numerics and Stability',
  '16e': 'Mixture of Experts',
  '16f': 'Efficient Adaptation',
  '16g': 'Domain Adaptation',
  '16h': 'Serving Systems',
  '17a': 'Agent and Environment',
  '17b': 'Tasks and Verification',
  '17c': 'Model Baselines',
  '17d': 'Policy Gradients',
  '17e': 'GRPO',
  '17f': 'Multi-Turn Rollouts',
  '17g': 'Evaluation and Ablations',
  '17h': 'Training Systems',
  '17i': 'Continual Learning',
};

// Published series. Links from one published series to another become site links; links to
// any other note are rendered as plain text, since those notes are not on the site.
const PUBLISHED = [
  { folder: '15 LLM Fundamentals', series: 'llm-fundamentals', title: 'LLM Fundamentals' },
  { folder: '16 LLM Deep Dive', series: 'llm-deep-dive', title: 'LLM Deep Dive' },
  { folder: '17 RL for LLM Agents', series: 'rl-for-llm-agents', title: 'Training Tool-Using Agent With RL' },
];

// Build the page map first, so links between notes can be resolved. `own` marks the pages of
// the series being imported; the other published series are only mapped for their links.
const pages = new Map(); // note name -> { slug, title, order, isIndex, base, own }
function mapSeries(dir, series, seriesTitle, own) {
  if (!fs.existsSync(dir)) return;
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.md')).sort();
  const base = `/tutorials/${series}/`;
  let order = 1;
  for (const file of files) {
    const name = file.replace(/\.md$/, '');
    const isIndex = / - Index$/.test(name);
    if (isIndex) {
      pages.set(name, { name, file, slug: '', title: seriesTitle, order: 0, isIndex, base, own });
      continue;
    }
    const { code, title } = parseNoteName(name);
    pages.set(name, { name, file, slug: slugify(title), title: SHORT_TITLES[code] ?? title, order: order++, isIndex, base, own });
  }
}
for (const s of PUBLISHED) {
  if (s.series !== SERIES) mapSeries(path.join(VAULT, s.folder), s.series, s.title, false);
}
mapSeries(SRC, SERIES, SERIES_TITLE, true);

const pageUrl = (page, heading) => {
  const base = page.slug ? `${page.base}${page.slug}/` : page.base;
  return heading ? `${base}#${new GithubSlugger().slug(heading.replace(/^\d+\.\s*/, ''))}` : base;
};

// Display text for a link target that is not published.
const plainName = (target) => {
  const [note, heading] = target.split('#');
  if (heading) return heading.replace(/^\d+\.\s*/, '');
  return parseNoteName(note).title.replace(/\s+-\s+Index$/, '');
};

const figureName = (file) => {
  const ext = path.extname(file).toLowerCase();
  return slugify(file.slice(0, -ext.length)) + ext;
};
const FIGURE_ALTS = {
  '15c pronoun attention weights.png': 'Illustrative attention weights from “it”. With “lit”, the lantern gets 62.4%. With “wore”, the scarf gets 62.4%.',
  '15c scaled dot product attention.png': 'Query, key and value vectors for she, lit and it. The query for it scores the three keys. Softmax gives weights 0.0743, 0.6200 and 0.3057, which mix the values into (1.8514, 0.3800).',
};
const figureAlt = (file) => {
  if (FIGURE_ALTS[file]) return FIGURE_ALTS[file];
  const base = file.replace(/\.[a-z0-9]+$/i, '').replace(/^\d+[a-z]?\s+/, '');
  return base.charAt(0).toUpperCase() + base.slice(1);
};


// Animations exported as numbered steps by scripts/export_animation_steps.py.
const STEPS_DIR = path.join(ROOT, 'src/data/animation-steps');
const escapeHtml = (t) =>
  String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// Step captions are plain text from the animation scripts. Render "$...$" with KaTeX and
// plain exponents such as "0.8^5" or "(1 - p)^k" as superscripts.
function richText(t) {
  const parts = String(t).split(/(\$[^$]+\$)/);
  const html = parts.map((part) => {
    if (/^\$[^$]+\$$/.test(part)) {
      return katex.renderToString(part.slice(1, -1), { throwOnError: false, output: 'html' });
    }
    return escapeHtml(part).replace(/\^(\([^()]*\)|[\w.]+)/g, '<sup>$1</sup>');
  });
  return glueMath(parts, html);
}

// Card text is raw HTML, which the site's rehype-math-nowrap step never sees, so the same
// pairing happens here: an inline formula and the punctuation touching it go in one span that
// custom.css keeps on a single line. `parts` are the source pieces, `html` their rendered form.
function glueMath(parts, html) {
  const out = [...html];
  for (let i = 0; i < parts.length; i++) {
    if (!/^\$[^$]/.test(parts[i])) continue; // inline math only, not $$ display math
    const after = (out[i + 1] ?? '').match(/^[.,;:!?)\]%]+/)?.[0] ?? '';
    const before = (out[i - 1] ?? '').match(/[(\[]$/)?.[0] ?? '';
    if (!after && !before) continue;
    if (after) out[i + 1] = out[i + 1].slice(after.length);
    if (before) out[i - 1] = out[i - 1].slice(0, -before.length);
    out[i] = `<span class="cm-math-nowrap">${before}${out[i]}${after}</span>`;
  }
  return out.join('');
}
// Plain text for alt attributes.
const plainText = (t) => escapeHtml(String(t).replace(/\$/g, ''));

// One HTML block with no blank lines, so Markdown keeps it intact.
// Short names for step-by-step figures, shown as "Step-by-step example: <name>".
// Keyed by the animation name from the export; falls back to the animation's own title.
const STEP_TITLES = {
  '15a softmax walkthrough': 'Softmax',
  '15a perplexity walkthrough': 'Perplexity',
  '15b bpe merges': 'Byte pair encoding',
  '15b embedding lookup': 'Embedding lookup',
  '15c qkv walkthrough': 'Queries, keys and values',
  '15c rope relative angle': 'Rotary position embedding',
  '15d residual stream accumulation': 'Residual stream',
  '15d mixture of experts routing': 'Mixture-of-experts routing',
  '15e minhash shingles': 'MinHash deduplication',
  '15e sequence packing': 'Sequence packing',
  '15e adamw two steps': 'AdamW update',
  '15f dpo one pair': 'DPO on one pair',
  '15f grpo group advantages': 'GRPO group advantages',
  '15g temperature reshaping': 'Temperature',
  '15g paged block table': 'Paged KV cache',
  '15g speculative accept reject': 'Speculative decoding',
  '15h cosine retrieval': 'Cosine retrieval',
  '15h lora forward pass': 'LoRA forward pass',
  '15i pass at k walkthrough': 'pass@k',
  '15i calibration error walkthrough': 'Calibration error',
  '16a cache bytes per token': 'Cache bytes per token',
  '16a sliding window receptive field': 'Sliding-window reach',
  '16a online softmax rescaling': 'Online softmax',
  '16b interpolation versus ntk scaling': 'RoPE extension',
  '16b segment recurrence': 'Segment recurrence',
  '16b retrieval versus full context cost': 'Retrieval versus long context',
  '16c ring all reduce': 'Ring all-reduce',
  '16c zero3 sharded step': 'ZeRO-3 step',
  '16c tensor parallel mlp': 'Tensor-parallel MLP',
  '16c pipeline 1f1b': '1F1B pipeline',
  '16f pcgrad projection': 'PCGrad projection',
  '16f lora memory budget': 'LoRA memory budget',
  '16f nf4 block quantization': 'NF4 quantization',
  '16f batched adapters': 'Batched adapters',
  '16d rounding into formats': 'Rounding into bf16 and FP8',
  '16d dynamic loss scaling': 'Dynamic loss scaling',
  '16d z-loss three logits': 'z-loss',
  '16d checkpoint interval': 'Checkpoint interval',
  '16e load balancing loss': 'Load-balancing loss',
  '16e capacity and dropped tokens': 'Capacity and dropped tokens',
  '16e expert bias balancing': 'Bias-based balancing',
  '16e all to all dispatch': 'All-to-all dispatch',
  '16g new token embeddings': 'New token embeddings',
  '16g replay exposure': 'Replay exposure',
  '16g domain loss and task': 'Domain loss against task quality',
  '16g temporal split leak': 'Temporal split',
  '16h roofline placement': 'Roofline placement',
  '16h chunked prefill schedule': 'Chunked prefill',
  '16h prefix tree reuse': 'Prefix tree reuse',
  '16h capacity plan': 'Capacity plan',
};

function stepFigure(file) {
  const json = path.join(STEPS_DIR, `${slugify(file.replace(/\.[a-z0-9]+$/i, ''))}.json`);
  if (!fs.existsSync(json)) return null;
  const data = JSON.parse(fs.readFileSync(json, 'utf8'));
  // Steps without an image (empty opening frames) are left out; the rest are renumbered.
  const shown = data.steps.filter((step) => step.image);
  const total = shown.length;
  let lastHeading = '';
  const items = shown.map((step, i) => {
    const parts = [];
    if (step.heading && step.heading !== lastHeading) {
      parts.push(`<p class="cm-step-phase">${richText(replaceSectionRefsPlain(step.heading))}</p>`);
    }
    lastHeading = step.heading ?? lastHeading;
    parts.push(`<p class="cm-step-text"><span class="cm-step-num">Step ${i + 1}</span>${richText(replaceSectionRefsPlain(step.caption))}</p>`);
    if (step.image) {
      parts.push(
        `<img src="${step.image}" alt="Step ${i + 1} of ${total}: ${plainText(replaceSectionRefsPlain(step.caption))}" loading="lazy" decoding="async">`,
      );
    }
    return `<li class="cm-step">${parts.join('')}</li>`;
  });
  return [
    // Collapsed by default, like deep dives: the header shows what the example is and how long it runs.
    `<details class="cm-steps not-content" data-anim="${slugify(data.name)}">`,
    `<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> ${richText(STEP_TITLES[data.name] ?? data.title)}</span>` +
      `<span class="cm-collapse-hint" aria-hidden="true"></span>` +
      (stepIntro(slugify(data.name), data) ? `<span class="cm-steps-sub">${richText(replaceSectionRefsPlain(stepIntro(slugify(data.name), data)))}</span>` : '') +
      `</summary>`,
    `<ol class="cm-steps-list">`,
    ...items,
    `</ol>`,
    `</details>`,
  ].join('\n');
}



// The site shows sections without numbers, so "## 7. Perplexity" becomes "## Perplexity" and
// prose such as "as section 9 shows" becomes "as the [maximum likelihood ...](#...) section shows".
let currentSections = new Map(); // section number -> heading text, for the page being converted

function sectionMap(raw) {
  const map = new Map();
  for (const m of raw.matchAll(/^##\s+(\d+)\.\s+(.+?)\s*$/gm)) map.set(Number(m[1]), m[2]);
  return map;
}

// "The chain rule of probability" -> "chain rule of probability"; keep acronyms such as "WordPiece".
function sectionName(title) {
  const t = title.replace(/^The\s+/, '');
  const first = t.split(/\s/)[0];
  // Lowercase an ordinary first word ("Logits" -> "logits"), never a name such as "AdamW", "RLHF" or "ALiBi".
  return /^[A-Z][a-z]/.test(first) && !/[A-Z]/.test(first.slice(1)) ? t[0].toLowerCase() + t.slice(1) : t;
}
const sectionLink = (n) => {
  const title = currentSections.get(Number(n));
  return title ? `[${sectionName(title)}](#${new GithubSlugger().slug(title)})` : null;
};

function replaceSectionRefs(line) {
  return line
    .replace(/\b([Ss])ections (\d+) and (\d+)\b/g, (all, s, a, b) => {
      const [la, lb] = [sectionLink(a), sectionLink(b)];
      return la && lb ? `${s === 'S' ? 'The' : 'the'} ${la} and ${lb} sections` : all;
    })
    .replace(/\b([Ss])ections (\d+) to (\d+)\b/g, (all, s, a, b) => {
      const [la, lb] = [sectionLink(a), sectionLink(b)];
      return la && lb ? `${s === 'S' ? 'The' : 'the'} sections from ${la} to ${lb}` : all;
    })
    .replace(/(\b[Ii]ts )?\b([Ss])ection (\d+)('s)?\b/g, (all, its, s, n, poss) => {
      const link = sectionLink(n);
      if (!link) return all;
      const det = its ?? (s === 'S' ? 'The ' : 'the ');
      return `${det}${link} section${poss ?? ''}`;
    });
}

// Plain-text version for HTML captions, e.g. "Section 5 example: ..." in a step figure.
function replaceSectionRefsPlain(text) {
  return String(text)
    .replace(/\b([Tt])(his|he) note\b/g, '$1$2 chapter')
    .replace(/\bSection (\d+) example\b/g, (all, n) =>
      currentSections.has(Number(n)) ? `Example from the ${sectionName(currentSections.get(Number(n)))} section` : all)
    .replace(/\b([Ss])ection (\d+)\b/g, (all, s, n) =>
      currentSections.has(Number(n)) ? `${s === 'S' ? 'The' : 'the'} ${sectionName(currentSections.get(Number(n)))} section` : all);
}

function unnumberSections(body) {
  return body
    .split('\n')
    .map((line) => {
      if (/^##\s+\d+\.\s+/.test(line)) return line.replace(/^(##\s+)\d+\.\s+/, '$1');
      if (/^\s*</.test(line) || /^\s*\$\$/.test(line)) return line; // HTML blocks and display math
      return replaceSectionRefs(line);
    })
    .join('\n');
}


// Algorithm boxes. In the notes an algorithm is one display-math array in the style of a LaTeX
// algorithm environment:
//   \begin{array}{l}\hline \textbf{Algorithm 1: } \text{name} \\ \hline \textbf{Input: } ... \\
//   \textbf{Output: } ... \\ \hline \text{1:}\quad ... \\ \text{2:}\quad\quad ... \\ \hline\end{array}
// The site rebuilds it as a card: header, input/output rows, and numbered lines with indentation
// guides. Each line's math is still rendered by KaTeX.
const tex = (src) => katex.renderToString(src.trim(), { throwOnError: false, output: 'html' });

function algorithmHtml(src) {
  const inner = src.match(/\\begin\{array\}\{l\}([\s\S]*)\\end\{array\}/)?.[1];
  if (!inner || !/\\textbf\{Algorithm/.test(inner)) return null;
  const rows = inner
    .split(/\\\\/)
    .map((r) => r.replace(/\\hline/g, '').trim())
    .filter(Boolean);
  let tag = 'Algorithm';
  let name = '';
  const io = [];
  const lines = [];
  for (const row of rows) {
    let m;
    if ((m = row.match(/^\\textbf\{(Algorithm[^:}]*):?\s*\}\s*(.*)$/))) {
      tag = m[1].trim();
      name = m[2].replace(/^\\text\{(.*)\}$/, '$1').trim();
    } else if ((m = row.match(/^\\textbf\{(Input|Output|Require|Ensure)s?:?\s*\}\s*(.*)$/))) {
      io.push({ label: m[1], body: m[2] });
    } else if ((m = row.match(/^\\text\{(\d+):\}\s*((?:\\quad\s*)*)(.*)$/))) {
      const indent = Math.max(0, (m[2].match(/\\quad/g) ?? []).length - 1);
      let code = m[3];
      let comment = '';
      const c = code.match(/\\quad\s*\\text\{\(([^)]*)\)\}\s*$/);
      if (c) {
        comment = c[1];
        code = code.slice(0, c.index);
      }
      lines.push({ n: m[1], indent, code, comment });
    } else {
      return null; // Unknown row shape: keep the original math rather than guess.
    }
  }
  if (!lines.length) return null;
  const ioHtml = io
    .map((r) => `<div class="cm-algo-io-row"><span class="cm-algo-io-label">${r.label}</span><span class="cm-algo-io-body">${tex(r.body)}</span></div>`)
    .join('');
  const lineHtml = lines
    .map(
      (l) =>
        `<li style="--indent:${l.indent}"><span class="cm-algo-ln">${l.n}</span>` +
        `<span class="cm-algo-code">${tex(l.code)}</span>` +
        (l.comment ? `<span class="cm-algo-comment">${escapeHtml(l.comment)}</span>` : '') +
        `</li>`,
    )
    .join('');
  return (
    `<figure class="cm-algo not-content">` +
    `<figcaption class="cm-algo-head"><span class="cm-algo-tag">${escapeHtml(tag)}</span>` +
    `<span class="cm-algo-name">${escapeHtml(name)}</span></figcaption>` +
    (ioHtml ? `<div class="cm-algo-io">${ioHtml}</div>` : '') +
    `<ol class="cm-algo-lines">${lineHtml}</ol></figure>`
  );
}


// BPE walkthrough tables. Symbols, pairs and words are written as space-separated letters in the
// notes ("l o we s t"), which is hard to read. On tables with these column titles, cells become
// token chips: [w][e] 11 for a pair count, [w] + [e] -> [we] for a merge, [lo][we][r] for a word.
const chip = (t, cls = '') => `<span class="cm-tok${cls}">${escapeHtml(t)}</span>`;
const word = (w) => `<span class="cm-word">${w.split(' ').map((t) => chip(t)).join('')}</span>`;
const pair = (a, b, cls = '') => `<span class="cm-pair${cls}">${chip(a)}${chip(b)}</span>`;

const TOKEN_CELLS = {
  // "w e 11, l o 10": the first pair is the one this round merges.
  pairCounts: (cell) => {
    const items = cell.split(/,\s*/).map((x) => x.match(/^(\S+) (\S+) (\d+)$/));
    if (items.some((m) => !m)) return null;
    return items
      .map((m, i) => `<span class="cm-pc">${pair(m[1], m[2], i === 0 ? ' is-top' : '')}<span class="cm-count">${m[3]}</span></span>`)
      .join(' ');
  },
  // "l o (2), w e (1)": pairs with their position in the merge list.
  pairRefs: (cell) => {
    const items = cell.split(/,\s*/).map((x) => x.match(/^(\S+) (\S+) \((\d+)\)$/));
    if (items.some((m) => !m)) return null;
    return items.map((m) => `<span class="cm-pc">${pair(m[1], m[2])}<span class="cm-rank">#${m[3]}</span></span>`).join(' ');
  },
  // "w + e → we"
  merge: (cell) => {
    const m = cell.match(/^(\S+) \+ (\S+) → (\S+)$/);
    return m ? `<span class="cm-merge">${chip(m[1])}<span class="cm-op">+</span>${chip(m[2])}<span class="cm-op">→</span>${chip(m[3], ' is-new')}</span>` : null;
  },
  // "l o w, l o we r": words made of symbols.
  words: (cell) => {
    const ws = cell.split(/,\s*/);
    if (ws.some((w) => !/^[a-z]+( [a-z]+)*$/.test(w))) return null;
    return `<span class="cm-words">${ws.map(word).join('')}</span>`;
  },
};

function columnRole(header) {
  const h = header.toLowerCase();
  if (h.includes('largest pair counts')) return 'pairCounts';
  if (h.includes('pairs found in the merge list') || h.includes('earliest')) return 'pairRefs';
  if (/\bmerge$/.test(h)) return 'merge';
  if (h.includes('afterwards')) return 'words';
  return null;
}

function tokenTables(body) {
  const lines = body.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith('|')) continue;
    const headers = lines[i].split('|').slice(1, -1).map((c) => c.trim());
    const roles = headers.map(columnRole);
    if (!roles.some(Boolean) || !/^\|\s*-/.test(lines[i + 1] ?? '')) continue;
    // "Line 4, largest pair counts" -> "Largest pair counts": the algorithm line numbers are not shown on the site.
    lines[i] =
      '| ' +
      headers
        .map((h) => h.replace(/^Line \d+,\s*/, ''))
        .map((h) => h.charAt(0).toUpperCase() + h.slice(1))
        .join(' | ') +
      ' |';
    let j = i + 2;
    for (; j < lines.length && lines[j].startsWith('|'); j++) {
      const cells = lines[j].split('|').slice(1, -1).map((c) => c.trim());
      lines[j] = '| ' + cells.map((c, k) => (roles[k] && TOKEN_CELLS[roles[k]](c)) || c).join(' | ') + ' |';
    }
    i = j - 1;
  }
  return lines.join('\n');
}


// What each step-by-step example is about, shown under its title: the situation and its starting
// numbers in one or two plain sentences. Keyed by the exported step set (src/data/animation-steps/<key>.json).
// Falls back to the animation's own subtitle, which is written for the frame and is often just a list.
const stepIntro = (key, data) => STEP_INTROS[key] ?? data.subtitle;
const STEP_INTROS = {
  '15a-perplexity-walkthrough': 'A model scores the sentence "the cat sat", giving the token that really came next probability 0.2, then 0.1, then 0.3. The example turns those three probabilities into an average loss in nats and then into a perplexity.',
  '15a-softmax-walkthrough': 'After the context "the cat", four candidate tokens get raw scores of 2.0, 1.0, 0.5 and −1.0. The example turns them into probabilities that sum to one, then shows how temperature sharpens or flattens them.',
  '15b-bpe-merges': 'A tiny training corpus holds low 5 times, lower 2 times, lowest 3 times and newest 6 times. Byte pair encoding starts from single letters and keeps merging the most frequent neighbouring pair into a new token.',
  '15b-embedding-lookup': 'A five-token vocabulary gives each token a row of 3 learned numbers. The example fetches the vector for "cat", id 1, once by multiplying a one-hot vector by the table and once by reading the row directly.',
  '15c-qkv-walkthrough': 'In “she lit it”, every token starts with two numbers. Follow the query for “it” as it scores all three keys, then use softmax to find how much of each value enters the output.',
  '15c-rope-relative-angle': 'The query for “it” and the keys from “she lit it” are rotated by their positions, 45 degrees per step. Placing the same phrase at different points in a sentence shows that the score depends only on how far apart two tokens are.',
  '15d-mixture-of-experts-routing': 'A layer with four expert feed-forward networks keeps the top two for each token. The first token is followed from its router scores to its weighted output, then five more tokens show how unevenly the work spreads over the experts.',
  '15d-residual-stream-accumulation': 'The token “it” enters as the two-number vector (1, 0), and three blocks each add a small vector to it instead of replacing it. The example tracks the running sum and shows that each block\'s contribution can still be read off and removed.',
  '15e-adamw-two-steps': 'One parameter starts at 0.5 and sees a gradient of 2.0, then 0.5, with learning rate 0.01 and weight decay 0.1. Two AdamW updates are worked by hand and compared with plain gradient descent on the same gradients.',
  '15e-minhash-shingles': 'Two sentences that differ in a single word are cut into overlapping three-word shingles. Four hash functions turn each sentence into a short signature, and the matching minimums estimate how much the two overlap without comparing them word by word.',
  '15e-sequence-packing': 'Four short documents, each closed by an end-of-document token, have to fill training windows of 8 tokens. The example compares padding each document with packing them back to back, then adds the mask that stops packed documents from reading each other.',
  '15f-dpo-one-pair': 'One prompt has a preferred answer and a rejected one, each scored by the model being trained and by a frozen copy of where it started. The example works one DPO update from those four log probabilities, with no reward model and no sampling.',
  '15f-grpo-group-advantages': 'The model answers "what is 17 × 24?" four times, and a checker marks an answer right when it equals 408. Each answer\'s advantage comes from comparing its reward with the rest of the group, with no critic network.',
  '15g-paged-block-table': 'Three requests, A, B and C, share a key-value cache of 10 blocks that hold 4 tokens each. A block table hands each request a block only when it needs one, and the example follows it as requests grow, finish and free their blocks.',
  '15g-speculative-accept-reject': 'After "The attic was", a small draft model guesses four tokens ahead and the large target model checks them in one pass. Each guess is kept or rejected by comparing the two models\' probabilities, and the example counts how many tokens a round yields on average.',
  '15g-temperature-reshaping': 'After "The attic was too", six candidate words have logits from 3.0 down to 0.0. Their probabilities are recomputed at temperatures 1, 0.5, 2 and close to 0, which reshape the distribution but never change the order.',
  '15h-cosine-retrieval': 'A customer asks about returning a damaged item, and three help-center chunks compete to answer. Toy embeddings with three topic scores, returns, item condition and delivery, rank the chunks by cosine similarity and pass the top two to the model.',
  '15h-lora-forward-pass': 'A frozen 3 by 3 weight matrix gets a rank-1 LoRA path beside it, made of two thin matrices. One input runs through both paths before and after training, and the trained numbers are counted against a full update.',
  '15i-calibration-error-walkthrough': 'A model answers 100 questions and states a confidence for each. The answers are grouped into three confidence bins, and each bin\'s average confidence is compared with how often its answers were actually right.',
  '15i-pass-at-k-walkthrough': 'A coding model is sampled 10 times on one problem, and 3 of the answers pass the unit tests. The example works out the chance that a user who gets k attempts sees at least one passing answer.',
  '15w-feed-forward-pass': 'A token with the two-number vector (1, −1) goes through a feed-forward network that widens it to 4 hidden numbers and brings it back to 2. ReLU and GELU are applied side by side to show where they differ.',
  '15w-layernorm-and-rmsnorm': 'The vector (1, 3, −1, 5) is normalized twice, once by LayerNorm and once by RMSNorm, with a gain of 1 and no offset. LayerNorm subtracts the mean first and RMSNorm skips that step.',
  '15w-vocabulary-size-comparison': 'The story model\'s 4,096-token vocabulary is compared with a 16,384-token one trained the same way on the same 20,000 TinyStories stories, and both are measured on 1,500 held-out stories. The comparison counts tokens per word, compute per 1,000 words, logit memory and how often each row of the table gets trained.',
  '16a-cache-bytes-per-token': 'The key-value cache that one token leaves behind is priced in bytes under plain multi-head attention on the 7B shape, under grouped-query attention with 8 key-value heads, and under the latent attention of DeepSeek-V3.',
  '16a-online-softmax-rescaling': 'One query attends to four keys with scores 1, 2, 4 and 3 and values 10, 20, 30 and 40, but only two keys fit in fast memory at a time. The exact softmax output is built tile by tile by rescaling a running sum.',
  '16a-sliding-window-receptive-field': 'Twelve tokens pass through layers in which each token reads only itself and the two tokens before it. The example tracks how far back token 12 can see after each layer and how many pairs each layer computes.',
  '16b-interpolation-versus-ntk-scaling': 'A model trained on 4,096-token windows is stretched to read 131,072 tokens, 32 times longer. Three of the 64 rotation frequencies in one 128-wide attention head show how position interpolation and NTK-aware scaling keep the angles inside the range seen in training.',
  '16b-retrieval-versus-full-context-cost': 'One question about net debt in a 96,000-token annual filing is answered on one H100 once with the whole filing in the prompt and once with six retrieved chunks of 1,000 tokens. Both are priced in seconds and memory, then in the chance that retrieval misses evidence.',
  '16b-segment-recurrence': 'A debt figure sits in the first token of a document read in three-token segments, where each layer also sees a memory of the segment before. With two layers and attention that simply averages, the example traces how far the figure travels.',
  '16c-pipeline-1f1b': 'A model split into four pipeline stages, one per GPU, trains on a batch cut into eight micro-batches. Each forward and backward pass is scheduled slot by slot, and the example measures how long the stages sit idle.',
  '16c-ring-all-reduce': 'Four GPUs each hold a gradient vector of four numbers, and every GPU needs their sum. Chunks travel around a ring, summed in three steps and copied back in three more, and the example counts the bytes each GPU sends.',
  '16c-tensor-parallel-mlp': 'A two-layer MLP with a hidden width of 4 is split across two GPUs, each holding half of each weight matrix. The input (1, 2) runs through both halves, and one sum at the end recovers exactly what a single GPU would compute.',
  '16c-zero3-sharded-step': 'A two-layer model with four weights trains on two GPUs, and each GPU stores only two of the weights with their optimizer state. One training step is followed through gathering weights before use, sharing gradients, and updating only the owned weights.',
  '16d-checkpoint-interval': 'The 70B training run on 2,048 GPUs expects a failure about every 25 hours, and every checkpoint pauses training while it is written. The example finds the save interval that wastes the least time, then shows how saving in the background shortens it.',
  '16d-rounding-into-formats': 'Four numbers, 1 + 2^-9, 0.1, 500 and 0.0009, are rounded into bf16 and the two FP8 formats. The example shows how far apart neighbouring values sit in each format, and where a value overflows or rounds to zero.',
  '16d-dynamic-loss-scaling': 'Gradients as small as 2e-8 round to zero in fp16, whose positive numbers run from about 6e-8 to 65,504. The loss is multiplied by a scale that halves after an overflow and doubles after clean steps, followed over five training steps.',
  '16d-z-loss-three-logits': 'A three-token output with logits 2, 1 and 0 is trained toward the first token. The example computes the cross entropy and the small z-loss penalty on the softmax normalizer, then shows why letting the logits drift upward costs precision in bf16.',
  '16e-all-to-all-dispatch': 'Four GPUs each hold one expert and two of eight tokens, and every token has to reach the expert its router picked. The dispatch exchange sends the tokens out, the combine exchange brings the results back, and the example counts the bytes on the wire.',
  '16e-capacity-and-dropped-tokens': 'The same eight tokens are routed top-1 and land unevenly on four experts: 4, 2, 1 and 1. Each expert gets a fixed number of slots set by the capacity factor, and the example counts dropped tokens and wasted slots at two factors.',
  '16e-expert-bias-balancing': 'The same eight tokens start out overloading expert 1 with 4 tokens against a fair share of 2. Instead of an extra loss, each expert gets a bias on its routing score, nudged by 0.05 per step until the load evens out.',
  '16e-load-balancing-loss': 'Eight tokens are routed top-1 across four experts, and expert 1 receives half of them. The Switch Transformer balancing loss is computed from each expert\'s share of tokens and mean router probability, showing which expert it pushes hardest.',
  '16f-batched-adapters': 'Three tokens in one batch come from requests that use two different LoRA adapters, a legal one and a medical one, on the same frozen base matrix. The base is applied once to all three, and each token then adds its own adapter\'s path.',
  '16f-lora-memory-budget': 'Rank 16 LoRA adapters go on all seven weight matrices in each of the reference 7B model\'s 32 blocks. The example counts the trained parameters, then adds up training memory, optimizer state and the saved adapter size against full fine-tuning.',
  '16f-nf4-block-quantization': 'Eight small weights, between −0.031 and 0.045, are stored in 4 bits each with the NF4 format from QLoRA. The block is scaled by its largest weight, each value is rounded to one of 16 levels, and the example measures the error and what the scale costs in bits.',
  '16f-pcgrad-projection': 'Two tasks share one model, and their gradients (4, 0) and (−1, 1) point more than 90 degrees apart, so a plain summed step makes task B worse. PCGrad removes the conflicting part of each gradient before adding them.',
  '16g-domain-loss-and-task': 'The reference 7B model is scored before and after 20 billion tokens of continued pretraining on filings, once by held-out loss and once by accuracy at pulling figures from a 10-Q. Splitting the loss by token type shows why it falls while the task gets worse.',
  '16g-new-token-embeddings': 'Three finance terms such as EBITDA, each cut into three pieces by the Llama 2 tokenizer, get whole-word tokens of their own. Their new input and output rows are built from the rows of their pieces, using rows of 3 numbers in place of 4,096.',
  '16g-replay-exposure': 'A 20-billion-token continued pretraining run draws on 10 billion tokens of filings and call transcripts, mixed with replayed general text. The example works out how often the run reads the domain data at replay ratios of 0, 25 and 50 percent.',
  '16g-temporal-split-leak': 'A test set of 400 questions about second-quarter 2025 figures from 100 companies is answered from 10-Q filings. A date cutoff that looks clean still leaks most answers through earlier earnings calls, and the example finds the cutoff and holdout that give an honest score.',
  '16h-capacity-plan': 'A product sends the 70B model 50 requests a second, each with a 1,000-token prompt and a 500-token reply, and promises a first token within a second and a new token every 30 ms. The example sizes the number of 8-GPU replicas, including a traffic peak.',
  '16h-chunked-prefill-schedule': 'The 7B model on one H100 is generating for 32 users when a 2,048-token prompt arrives. Running the prompt in one piece is compared with cutting it into chunks between decode steps, measured by the longest stall the 32 users see.',
  '16h-prefix-tree-reuse': 'Three requests to the 7B model start with the same 1,024-token system prompt, and the third continues the first conversation. A prefix tree stores each shared run of cached tokens once, and the example counts the prompt tokens computed and the cache memory saved.',
  '16h-roofline-placement': 'The 7B model decodes on one H100 at batch sizes of 1, 64 and 256, with empty contexts so the weights are the only bytes read. Each batch is placed on the roofline to see whether memory bandwidth or arithmetic sets its speed.',
};

// Worked walkthroughs. A "###" walkthrough whose steps are a numbered list becomes one
// step-by-step card: each step keeps the note's own text (math rendered) and gets a figure,
// taken from the animation that replays the same example or from 15w-gifs.py. `frames` maps
// step k to a 1-based frame of the exported steps (null keeps the step as text only); `extra`
// appends further frames, with their captions, for content the text steps do not cover.
// `replaces` removes the separate animation card, which would otherwise repeat the example.
const WALKTHROUGHS = {
  'Comparison of 4,096 and 16,384 vocabularies': { title: 'Vocabulary size comparison', steps: '15w-vocabulary-size-comparison', frames: [1, 2, 3, 4, 5] },
  'A lookup computed both ways': { title: 'Embedding lookup', steps: '15b-embedding-lookup', frames: [1, null, 4, 5, 6], replaces: true },
  'A complete calculation on three tokens': { title: 'Attention on three tokens', steps: '15c-qkv-walkthrough', frames: [4, 5, 6, 7, 8, 10], replaces: true },
  'RoPE on a sentence': { title: 'Rotary position embedding', steps: '15c-rope-relative-angle', frames: [2, 3, 4, 5], extra: [6], replaces: true },
  'Three blocks on a two-number stream': { title: 'Residual stream', steps: '15d-residual-stream-accumulation', frames: [1, 2, 4, 5], replaces: true },
  'One vector normalized both ways': { title: 'LayerNorm versus RMSNorm', steps: '15w-layernorm-and-rmsnorm', frames: [1, 2, 3, 4] },
  'A feed-forward pass on a two-number token': { title: 'Feed-forward pass', steps: '15w-feed-forward-pass', frames: [1, 2, 3, 4] },
  'Routing one token through four experts': { title: 'Mixture-of-experts routing', steps: '15d-mixture-of-experts-routing', frames: [null, 1, 2], extra: [8, 9], replaces: true },
  'A near duplicate check by hand': { title: 'MinHash deduplication', steps: '15e-minhash-shingles', frames: [1, null, 5, 6], replaces: true },
  'Two AdamW steps on one parameter': { title: 'AdamW update', steps: '15e-adamw-two-steps', frames: [2, 4, 5, 7, 8, 9], replaces: true },
  'A cosine ranking worked by hand': { title: 'Cosine retrieval', steps: '15h-cosine-retrieval', frames: [2, 5, 9, 10], replaces: true },
  'A rank one LoRA layer worked by hand': { title: 'LoRA forward pass', steps: '15h-lora-forward-pass', frames: [4, 5, 6, 7, 10], extra: [11], replaces: true },
  'A pass at k calculation': { title: 'pass@k', steps: '15i-pass-at-k-walkthrough', frames: [1, 2, 3, 4, 5], extra: [6], replaces: true },
  // LLM Deep Dive
  'Cache bytes for one token under GQA and MLA': { title: 'Cache bytes per token', steps: '16a-cache-bytes-per-token', frames: [1, 2, 3, 4, 5] },
  'Reach of a sliding window through layers': { title: 'Sliding-window reach', steps: '16a-sliding-window-receptive-field', frames: [1, 2, 3, 4, 5], replaces: true },
  'Online softmax over two tiles': { title: 'Online softmax', steps: '16a-online-softmax-rescaling', frames: [1, 2, 3, 4, 5], replaces: true },
  'Three frequencies under two extension rules': { title: 'RoPE extension', steps: '16b-interpolation-versus-ntk-scaling', frames: [1, 2, 3, 4, 5], replaces: true },
  'A segment recurrence state update': { title: 'Segment recurrence', steps: '16b-segment-recurrence', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'The cost of one question both ways': { title: 'Retrieval versus long context', steps: '16b-retrieval-versus-full-context-cost', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'Ring all-reduce on four GPUs': { title: 'Ring all-reduce', steps: '16c-ring-all-reduce', frames: [1, 2, 3, 4, 5, 6, 7], replaces: true },
  'A sharded ZeRO-3 step': { title: 'ZeRO-3 step', steps: '16c-zero3-sharded-step', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'A tensor-parallel MLP on two GPUs': { title: 'Tensor-parallel MLP', steps: '16c-tensor-parallel-mlp', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'Filling a four-stage pipeline with 1F1B': { title: '1F1B pipeline', steps: '16c-pipeline-1f1b', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'PCGrad on two conflicting gradients': { title: 'PCGrad projection', steps: '16f-pcgrad-projection', frames: [1, 2, 3, 4, 5], replaces: true },
  'The 7B at rank 16 on every matrix': { title: 'LoRA memory budget', steps: '16f-lora-memory-budget', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'NF4 on a block of eight weights': { title: 'NF4 quantization', steps: '16f-nf4-block-quantization', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'Two adapters in one batch': { title: 'Batched adapters', steps: '16f-batched-adapters', frames: [1, 2, 3, 4, 5, 6], replaces: true },
  'Rounding numbers into bf16 and FP8': { title: 'Rounding into bf16 and FP8', steps: '16d-rounding-into-formats', frames: [1, 2, 3, 4, 5], replaces: true },
  'Dynamic loss scaling over five steps': { title: 'Dynamic loss scaling', steps: '16d-dynamic-loss-scaling', frames: [1, 2, 3, 4, 5], replaces: true },
  'z-loss on three logits': { title: 'z-loss', steps: '16d-z-loss-three-logits', frames: [1, 2, 3, 4, 5], replaces: true },
  'Picking a checkpoint interval': { title: 'Checkpoint interval', steps: '16d-checkpoint-interval', frames: [1, 2, 3, 4, 5], replaces: true },
  'A balancing loss on eight tokens': { title: 'Load-balancing loss', steps: '16e-load-balancing-loss', frames: [1, 2, 3, 4, 5], replaces: true },
  'Capacity on the same eight tokens': { title: 'Capacity and dropped tokens', steps: '16e-capacity-and-dropped-tokens', frames: [1, 2, 3, 4], replaces: true },
  'Bias updates over three steps': { title: 'Bias-based balancing', steps: '16e-expert-bias-balancing', frames: [1, 2, 3, 4], replaces: true },
  'Dispatch and combine across four GPUs': { title: 'All-to-all dispatch', steps: '16e-all-to-all-dispatch', frames: [1, 2, 3, 4, 5], replaces: true },
  'Three new finance tokens': { title: 'New token embeddings', steps: '16g-new-token-embeddings', frames: [1, 2, 3, 4, 5], replaces: true },
  'Replay exposure over a 20B-token run': { title: 'Replay exposure', steps: '16g-replay-exposure', frames: [1, 2, 3, 4, 5], replaces: true },
  'Two checkpoints scored two ways': { title: 'Domain loss against task quality', steps: '16g-domain-loss-and-task', frames: [1, 2, 3, 4, 5], replaces: true },
  'A temporal split that leaks and one that does not': { title: 'Temporal split', steps: '16g-temporal-split-leak', frames: [1, 2, 3, 4, 5], replaces: true },
  'Three decode batch sizes on the roofline': { title: 'Roofline placement', steps: '16h-roofline-placement', frames: [1, 2, 3, 4, 5], replaces: true },
  'A chunked prefill schedule': { title: 'Chunked prefill', steps: '16h-chunked-prefill-schedule', frames: [1, 2, 3, 4, 5], replaces: true },
  'Three requests sharing one system prompt': { title: 'Prefix tree reuse', steps: '16h-prefix-tree-reuse', frames: [1, 2, 3, 4], replaces: true },
  'A capacity plan for the 70B': { title: 'Capacity plan', steps: '16h-capacity-plan', frames: [1, 2, 3, 4, 5, 6] },
  'One valid exchange and two false completions': { title: 'Exchange verification', steps: '17b-exchange-verifier', frames: [1, 2, 3] },
  'Masked loss on a tool trace': { title: 'Masked supervised loss', steps: '17c-masked-loss', frames: [1, 2, 3] },
  'One rewarded exchange': { title: 'One policy gradient update', steps: '17d-one-rewarded-exchange', frames: [1, 2, 3] },
  'A baseline across two outcomes': { title: 'Two outcomes and a baseline', steps: '17d-baseline-two-outcomes', frames: [1, 2] },
  'Four exchange attempts': { title: 'Group-normalized rewards', steps: '17e-four-exchange-attempts', frames: [1, 2, 3] },
  'One inventory recovery': { title: 'Inventory recovery', steps: '17f-inventory-recovery', frames: [1, 2, 3, 4] },
  'A masked exchange prefix': { title: 'Token-level loss mask', steps: '17f-masked-exchange-prefix', frames: [1, 2, 3] },
  'Retry scores for one task': { title: 'Repeat one task', steps: '17g-retry-scores-for-one-task', frames: [1, 2, 3] },
  'A rollout batch on one GPU': { title: 'A colocated training round', steps: '17h-rollout-batch-on-one-gpu', frames: [1, 2, 3] },
  'Promotion gates for policy V2': { title: 'Release gates for a new policy', steps: '17i-promotion-gates-for-policy-v2', frames: [1, 2, 3] },
};

// Paragraphs that explain a step-by-step example go inside its card, so a reader who leaves the
// card closed does not meet text about steps they have not opened. Keyed by the card's data-anim
// or data-walkthrough value; the number is how many paragraphs right after the card move in.
// When a note paragraph mixes an explanation with new material, split it in the note so only
// the explanation moves.
const STEP_NOTES = {
  '15a-softmax-walkthrough': 1,
  '15a-perplexity-walkthrough': 1,
  'a-complete-calculation-on-three-tokens': 1,
  'routing-one-token-through-four-experts': 1,
  '15g-paged-block-table': 1,
  'a-cosine-ranking-worked-by-hand': 1,
  'a-pass-at-k-calculation': 1,
  '15i-calibration-error-walkthrough': 1,
};

function stepNotes(body) {
  const lines = body.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const id = lines[i].match(/^<details class="cm-steps not-content" data-(?:anim|walkthrough)="([^"]+)"/)?.[1];
    const want = id && STEP_NOTES[id];
    if (!want) continue;
    const close = lines.indexOf('</details>', i);
    const paras = [];
    let j = close + 1;
    while (paras.length < want) {
      while (j < lines.length && lines[j].trim() === '') j++;
      // Stop at anything that is not a plain paragraph: a heading, HTML, a figure, a list.
      if (j >= lines.length || /^(#|<|!\[|\||>|\d+\. |[-*] )/.test(lines[j])) break;
      const start = j;
      while (j < lines.length && lines[j].trim() !== '') j++;
      paras.push(lines.slice(start, j).join('\n'));
    }
    if (paras.length !== want) {
      console.warn(`step notes for "${id}": found ${paras.length} of ${want} paragraphs`);
      continue;
    }
    // Inside the card, "the step-by-step figure" is the card itself.
    const own = (p) => p.replace(/\bThe step-by-step figure\b/g, 'This example');
    const notes = ['<div class="cm-steps-notes">', ...paras.map((p) => `<p>${mdInline(own(p))}</p>`), '</div>'];
    lines.splice(close, j - close, ...notes, '</details>', '');
    i = close + notes.length;
  }
  return lines.join('\n');
}

// Markdown of one list item to HTML: display and inline math, bold, links, code.
function mdInline(md) {
  const parts = md.split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/);
  const html = parts
    .map((part) => {
      if (/^\$\$[\s\S]+\$\$$/.test(part)) {
        return katex.renderToString(part.slice(2, -2).trim(), { throwOnError: false, displayMode: true });
      }
      if (/^\$[^$\n]+\$$/.test(part)) {
        return katex.renderToString(part.slice(1, -1), { throwOnError: false });
      }
      return escapeHtml(part.replace(/\s*\n\s*/g, ' '))
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
    });
  return glueMath(parts, html);
}

function walkthroughCards(body) {
  let lines = body.split('\n');
  for (const [heading, cfg] of Object.entries(WALKTHROUGHS)) {
    // Worked examples are titled "### Practice: <name>" in the notes. WALKTHROUGHS is keyed by <name>,
    // which also gives the card its data-walkthrough id.
    let h = lines.indexOf(`### Practice: ${heading}`);
    if (h === -1) h = lines.indexOf(`### ${heading}`);
    if (h === -1) continue;
    const json = path.join(STEPS_DIR, `${cfg.steps}.json`);
    if (!fs.existsSync(json)) {
      console.warn(`walkthrough "${heading}": missing ${cfg.steps}.json`);
      continue;
    }
    const data = JSON.parse(fs.readFileSync(json, 'utf8'));
    // The numbered list: items start "N. ", continuation lines are indented.
    let a = h + 1;
    while (a < lines.length && !/^1\. /.test(lines[a]) && !/^#{2,3} /.test(lines[a])) a++;
    if (!/^1\. /.test(lines[a] ?? '')) continue;
    const items = [];
    let b = a;
    for (; b < lines.length; b++) {
      const line = lines[b];
      if (/^\d+\. /.test(line)) items.push(line.replace(/^\d+\.\s+/, ''));
      else if (/^\s+\S/.test(line) && items.length) items[items.length - 1] += '\n' + line.trim();
      else if (line.trim() === '' && /^\s+\S/.test(lines[b + 1] ?? '')) continue;
      else break;
    }
    if (items.length !== cfg.frames.length) {
      console.warn(`walkthrough "${heading}": ${items.length} steps but ${cfg.frames.length} frames mapped`);
      continue;
    }
    const steps = items.map((text, i) => ({ html: mdInline(text), image: cfg.frames[i] && data.steps[cfg.frames[i] - 1]?.image }));
    for (const k of cfg.extra ?? []) {
      const st = data.steps[k - 1];
      if (st) steps.push({ html: richText(replaceSectionRefsPlain(st.caption)), image: st.image });
    }
    const total = steps.length;
    const card = [
      `<details class="cm-steps not-content" data-walkthrough="${slugify(heading)}">`,
      `<summary class="cm-steps-head"><span class="cm-steps-title"><span class="cm-steps-label">Step-by-step example:</span> ${escapeHtml(cfg.title)}</span>` +
        `<span class="cm-collapse-hint" aria-hidden="true"></span>` +
        (stepIntro(cfg.steps, data) ? `<span class="cm-steps-sub">${richText(replaceSectionRefsPlain(stepIntro(cfg.steps, data)))}</span>` : '') +
        `</summary>`,
      `<ol class="cm-steps-list">`,
      ...steps.map(
        (st, i) =>
          `<li class="cm-step"><div class="cm-step-text"><span class="cm-step-num">Step ${i + 1}</span>${st.html}</div>` +
          (st.image ? `<img src="${st.image}" alt="Step ${i + 1} of ${total}, ${escapeHtml(cfg.title)}" loading="lazy" decoding="async">` : '') +
          `</li>`,
      ),
      `</ol>`,
      `</details>`,
    ];
    lines.splice(a, b - a, ...card);
    if (cfg.replaces) {
      // Drop the separate animation card for the same example, and the sentence that points to it.
      let end = lines.findIndex((l, i) => i > h && /^#{2,3} /.test(l));
      if (end === -1) end = lines.length;
      const start = lines.findIndex((l, i) => i > h && i < end && l.startsWith(`<details class="cm-steps not-content" data-anim="${cfg.steps}"`));
      if (start !== -1) {
        const stop = lines.findIndex((l, i) => i > start && l === '</details>');
        lines.splice(start, stop - start + 1);
        end -= stop - start + 1;
      }
      for (let i = h; i < end; i++) {
        if (lines[i].includes('The step-by-step figure ')) {
          lines[i] = lines[i].replace(/The step-by-step figure [^.]*\.\s*/, '').trimEnd();
          break;
        }
      }
    }
  }
  return lines.join('\n');
}

// Deep dives: collapsed Obsidian callouts ("> [!deep-dive]-") become collapsible cards.
//
//   ## 8. Bits per byte across tokenizers      A heading followed by an untitled callout
//                                              becomes one card titled by the heading. The
//   > [!deep-dive]-                            heading stays a real <h2>/<h3> inside the card,
//   > body ...                                 so the sidebar, numbering and links still work.
//
//   > [!deep-dive]- A title                    A titled callout becomes a card with that title.

// Lightbulb icon (Lucide) shown at the start of a card header.
const COLLAPSE_ICON =
  '<span class="cm-collapse-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 ' +
  '1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg></span>';

const DEEP_START = '<!--cm-deep-dive-start:';
const DEEP_END = '<!--cm-deep-dive-end-->';

// Pass 1, on the raw note: strip the "> " prefix from deep-dive callouts and mark where they start and end,
// so the body goes through the normal conversion (links, figures, math).
function unwrapCallouts(text) {
  const lines = text.split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    // A "> [!model] Title" callout becomes an always-open highlighted card. Its anchor is the
    // Obsidian block id on its own line after the callout ("^story-model"), so links
    // written as [[Note#^story-model|...]] work in the vault and on the site alike.
    const card = lines[i].match(/^>\s*\[!model\]\s*(.*)$/i);
    if (card) {
      let j = i + 1;
      const inner = [];
      while (j < lines.length && /^>/.test(lines[j])) inner.push(lines[j++].replace(/^> ?/, ''));
      let k = j;
      while (k < lines.length && lines[k].trim() === '') k++;
      const blockId = lines[k]?.match(/^\^([\w-]+)\s*$/)?.[1];
      const id = blockId ?? slugify(card[1]);
      const title = escapeHtml(card[1].trim());
      if (id === 'story-model') {
        const specs = inner.findIndex((line) => line.startsWith('|'));
        if (specs < 0) throw new Error('The story model card needs a specifications table');
        out.push(
          `<aside class="cm-model-card cm-notebook-card" id="${id}">`,
          '<div class="cm-notebook-kicker" id="story-model-project">Project</div>',
          `<p class="cm-notebook-title">${title}</p>`,
          '',
          ...inner.slice(0, specs),
          '<div class="cm-notebook-specs" role="region" aria-label="Model specifications" tabindex="0">',
          '',
          ...inner.slice(specs),
          '',
          '</div>',
          '',
          '<div class="cm-notebook-actions"><a href="/tutorials/llm-fundamentals/notebook/">Browse the notebook <span aria-hidden="true">↗</span></a></div>',
          '</aside>',
        );
        i = blockId ? k : j - 1;
        continue;
      }
      const label = id === 'scale-reference-models' ? '' : '<span class="cm-model-label">Running example</span>';
      out.push(
        `<aside class="cm-model-card" id="${id}">`,
        `<p class="cm-model-title">${label}${title}</p>`,
        '',
        ...inner,
        '',
        '</aside>',
      );
      i = blockId ? k : j - 1;
      continue;
    }
    const m = lines[i].match(/^>\s*\[!deep-dive\][-+]?\s*(.*)$/i);
    if (!m) {
      out.push(lines[i]);
      continue;
    }
    out.push(`${DEEP_START}${m[1].trim()}-->`, '');
    let j = i + 1;
    while (j < lines.length && /^>/.test(lines[j])) {
      out.push(lines[j].replace(/^> ?/, ''));
      j++;
    }
    out.push('', DEEP_END);
    i = j - 1;
  }
  return out.join('\n');
}

// Pass 2, on the converted Markdown: turn the markers into <details> cards.
// Blank lines around the inner Markdown keep it parsed as Markdown.
function buildDeepDives(body) {
  const out = [];
  for (const line of body.split('\n')) {
    if (line === DEEP_END) {
      out.push('', '</details>', '');
      continue;
    }
    if (!line.startsWith(DEEP_START)) {
      out.push(line);
      continue;
    }
    let title = line.slice(DEEP_START.length, -3).trim();
    let tag = 'span';
    if (!title) {
      // Untitled: absorb the heading right above the callout.
      while (out.length && out[out.length - 1].trim() === '') out.pop();
      const h = out[out.length - 1]?.match(/^(#{2,3})\s+(.+?)\s*$/);
      if (h) {
        out.pop();
        tag = `h${h[1].length}`;
        title = h[2];
      } else {
        title = 'Deep dive';
      }
    }
    const id = new GithubSlugger().slug(title);
    const titleHtml =
      tag === 'span'
        ? `<span class="cm-collapse-title">${escapeHtml(title)}</span>`
        : `<${tag} class="cm-collapse-title" id="${id}">${escapeHtml(title)}</${tag}>`;
    out.push(
      tag === 'span' ? `<details class="cm-collapse" id="${id}">` : '<details class="cm-collapse">',
      `<summary>${COLLAPSE_ICON}<span class="cm-collapse-text"><span class="cm-collapse-kicker">Deep dive</span>` +
        `${titleHtml}</span><span class="cm-collapse-hint" aria-hidden="true"></span></summary>`,
    );
  }
  return out.join('\n');
}

const copiedFigures = new Set();
const missingFigures = [];
const droppedLinks = new Set();

function convertLink(inner) {
  // inner is the text between [[ and ]], possibly with an escaped table pipe "\|".
  const [targetRaw, aliasRaw] = inner.replace(/\\\|/g, '|').split('|');
  const target = targetRaw.trim();
  const alias = aliasRaw?.trim();
  const [note, heading] = target.split('#');
  const page = pages.get(note.trim());
  if (!page) {
    droppedLinks.add(note.trim());
    return null;
  }
  const text = alias ?? (heading ? heading.replace(/^\d+\.\s*/, '') : page.title);
  return `[${text}](${pageUrl(page, heading)})`;
}

// Wording that only makes sense inside the Obsidian vault.
const WEB_WORDING = [
  // No divider between a chapter's introduction and its first section (non-global: first one only).
  [/\n---\n+(?=## )/, '\n'],
  [/\b(across|of|in) this folder\b/g, '$1 this series'],
  [/every note in this folder/g, 'every chapter in this series'],
  // Each note is a chapter on the site.
  [/\b([Tt])his note\b/g, '$1his chapter'],
  [/\b([Tt])he note\b/g, '$1he chapter'],
  [/\b([Tt])he whole note\b/g, '$1he whole chapter'],
  [/\b([Ee])ach note\b/g, '$1ach chapter'],
  [/\b([Ee])very note\b/g, '$1very chapter'],
  [/\b([Tt])hese nine notes\b/g, '$1hese nine chapters'],
  [/\bof all nine notes\b/g, 'of all nine chapters'],
  // Animations are shown as numbered step figures on the site.
  [/\b([Tt])he animation\b/g, '$1he step-by-step figure'],
  [/the roughly (\d+(?:\.\d+)?) million words of notes in this vault/g, 'about $1 million words of the author\'s study notes'],
  // Index-only references to unpublished notes.
  [/ Interview practice for this folder lives in the question bank at \[\[17 LLM Fundamentals\]\][^.]*\./g, ''],
  [/\n\nThis folder is the end-to-end entrance\.[^\n]*/g, ''],
  [/, then work through \[\[17 LLM Fundamentals\]\] and answer every collapsed question aloud before opening it/g, ''],
  [/\n## Related topics\n[\s\S]*?(?=\n\*\*Last updated|$)/g, '\n'],
  // The series overview on the site stops after the chapter list. The quick reference, interview
  // habits, design exercise and study order stay in the Obsidian index only.
  [/\n## Cross cutting quick reference\n[\s\S]*?(?=\n\*\*Last updated|$)/g, '\n'],
  // The site introduction describes the notebook project without an extra Overview heading.
  [/\n## Overview\n/g, ''],
  [/\bNumbered sections then build\b/g, 'Sections then build'],
  // The Obsidian index is an interview-prep index; the site speaks to learners and practitioners.
  [/in a product decision or an interview question start to make sense/g, 'in a product decision or a research paper start to make sense'],
  [/Everything here is written for someone new to the field\.[^\n]*/g,
    "This tutorial is built around a working language model we implement from scratch. In the [project notebook](/tutorials/llm-fundamentals/notebook/), we build a tokenizer and an 11-million-parameter Transformer, train it on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), and compare its responses before and after fine-tuning. The project is designed to run on a single consumer GPU, including the built-in GPU on an Apple Silicon Mac. That same implementation connects the chapters, from how text becomes tokens to how a model generates answers and how those answers are evaluated. Newcomers can follow the chapters in order, while engineers and researchers can revisit individual topics and explore the deep dives."],
  [/\n+Each chapter opens with the problem it answers[^\n]*/g, ''],
  [/ These nine chapters follow one model through its whole life[^.]*\./g, ''],
  // The site outlines the merged pair in the BPE table; the note's plain table does not.
  [/The first pair in each row is the one line 6 merges in that round\./g, 'The first pair in each row, outlined, is the one line 6 merges in that round.'],
];

// "15a" -> "chapter 1" in prose, for the codes of the series being imported (its folder number).
// Leaves image paths and URLs alone.
const SERIES_NO = path.basename(SRC).match(/^(\d+)/)?.[1] ?? '15';
const chapterNo = (letter) => letter.charCodeAt(0) - 96;
function chapterWords(text) {
  const c = () => `${SERIES_NO}([a-z])`;
  return text
    .replace(new RegExp(`(?<=[\\s(])${c()}, ${c()}, and ${c()}(?=[\\s,.;:)])`, 'g'), (_, a, b, d) =>
      `chapters ${chapterNo(a)}, ${chapterNo(b)}, and ${chapterNo(d)}`)
    .replace(new RegExp(`(?<=[\\s(])${c()} (to|and) ${c()}(?=[\\s,.;:)])`, 'g'), (_, a, w, b) =>
      `chapters ${chapterNo(a)} ${w} ${chapterNo(b)}`)
    .replace(new RegExp(`(?<=[\\s(])${c()}(?=[\\s,.;:)])`, 'g'), (_, a) => `chapter ${chapterNo(a)}`);
}

function convertBody(body) {
  for (const [pattern, replacement] of WEB_WORDING) body = body.replace(pattern, replacement);
  const out = [];
  let inFence = false;

  for (let line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;
    if (inFence) {
      out.push(line);
      continue;
    }

    // Single-line display math ("$$...$$" alone on a line) must be a fenced block for remark-math.
    const display = line.match(/^(\s*)\$\$(.+)\$\$\s*$/);
    const algorithm = display && algorithmHtml(display[2]);
    if (algorithm) {
      out.push(algorithm);
      continue;
    }
    if (display) {
      out.push(`${display[1]}$$`, `${display[1]}${display[2].trim()}`, `${display[1]}$$`);
      continue;
    }

    // Obsidian-only metadata lines.
    if (/^#[a-z][\w-]*(\s+#[a-z][\w-]*)*\s*$/i.test(line)) continue; // tag line
    if (/^Parent:\s*\[\[/.test(line)) continue;
    if (/^\*\*Last updated:/i.test(line)) continue;

    // Figure embeds.
    line = line.replace(/!\[\[([^\]]+)\]\]/g, (_, file) => {
      const clean = file.split('|')[0].trim();
      // An animation with exported steps becomes a step-by-step figure instead of a video.
      if (/\.(mp4|gif)$/i.test(clean)) {
        const steps = stepFigure(clean);
        if (steps) return steps;
      }
      const from = path.join(FIGURES, clean);
      if (!fs.existsSync(from)) {
        missingFigures.push(clean);
        return '';
      }
      const name = figureName(clean);
      if (!copiedFigures.has(name)) {
        fs.copyFileSync(from, path.join(OUT_IMAGES, name));
        copiedFigures.add(name);
      }
      const src = `/images/${SERIES}/${name}`;
      if (/\.(mp4|webm)$/i.test(name)) {
        // Animations: loop silently like a GIF, with controls so readers can pause or scrub.
        return `<video class="cm-figure-video" src="${src}" autoplay loop muted playsinline controls preload="metadata" aria-label="${figureAlt(clean)}"></video>`;
      }
      if (clean === '15c scaled dot product attention.png') {
        return `[![${figureAlt(clean)}](${src})](${src} "View the full-size attention figure")`;
      }
      return `![${figureAlt(clean)}](${src})`;
    });

    // "Related" footer: keep only links that resolve, drop the line if none remain.
    if (/^\*\*Related\*\*:/.test(line)) {
      const links = [...line.matchAll(/\[\[([^\]]+)\]\]/g)].map((m) => convertLink(m[1])).filter(Boolean);
      if (links.length) out.push(`**Related**: ${links.join(' · ')}`);
      continue;
    }

    // A "### [[Note]]" heading in the index becomes a linked heading.
    line = line.replace(/\[\[([^\]]+)\]\]/g, (_, inner) => {
      const converted = convertLink(inner);
      if (converted) return converted;
      const [target, alias] = inner.replace(/\\\|/g, '|').split('|');
      return alias?.trim() || plainName(target.trim());
    });

    out.push(line);
  }

  return chapterWords(out.join('\n'));
}

// Collapse separators like " |  | " left behind after dropping unpublished links.
function tidy(text) {
  return text
    .replace(/^(?:\s*\|\s*)+$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function firstParagraph(body) {
  const para = body
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .find((p) => p && !p.startsWith('#') && !p.startsWith('!') && !p.startsWith('---') && !p.startsWith('|'));
  if (!para) return '';
  const plain = para
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\*\*|__|`/g, '')
    .replace(/\$[^$]*\$/g, '')
    .replace(/\s+/g, ' ');
  const sentences = plain.match(/[^.!?]+[.!?]/g) ?? [plain];
  let desc = '';
  for (const s of sentences) {
    if ((desc + s).length > 200) break;
    desc += s;
  }
  return (desc || plain.slice(0, 197) + '...').trim();
}

// The index note's Topics table has one "| [[Note]] | summary |" row per chapter.
// Use that summary as the chapter's description.
const indexPage = [...pages.values()].find((p) => p.own && p.isIndex);
const summaries = new Map();
if (indexPage) {
  const indexRaw = fs.readFileSync(path.join(SRC, indexPage.file), 'utf8');
  for (const m of indexRaw.matchAll(/^\|\s*\[\[([^\]|\\]+)[^\]]*\]\]\s*\|\s*([^|\n]+?)\s*\|\s*$/gm)) {
    summaries.set(m[1].trim(), m[2].trim());
  }
}

const yamlString = (s) => JSON.stringify(s);

fs.rmSync(OUT_DOCS, { recursive: true, force: true });
fs.rmSync(OUT_IMAGES, { recursive: true, force: true });
fs.mkdirSync(OUT_DOCS, { recursive: true });
fs.mkdirSync(OUT_IMAGES, { recursive: true });

for (const page of pages.values()) {
  if (!page.own) continue;
  const raw = fs.readFileSync(path.join(SRC, page.file), 'utf8');
  const updated = raw.match(/\*\*Last updated:\s*(\d{4}-\d{2}-\d{2})\*\*/)?.[1];
  const withoutH1 = raw.replace(/^#\s+.*\n/, '');
  // Drop a leading horizontal rule that only separated the Obsidian header block.
  currentSections = sectionMap(withoutH1);
  const body = tidy(buildDeepDives(stepNotes(walkthroughCards(tokenTables(unnumberSections(convertBody(unwrapCallouts(withoutH1)))))))).replace(/^---\s*\n/, '').replace(/\n---\s*$/, '');

  const summary = summaries.get(page.name);
  const description = summary ? firstParagraph(convertBody(summary)) : firstParagraph(body);
  const frontmatter = [
    '---',
    `title: ${yamlString(page.title)}`,
    `description: ${yamlString(description)}`,
    'sidebar:',
    `  order: ${page.order}`,
    `  label: ${yamlString(page.isIndex ? 'Overview' : page.title)}`,
    updated ? `lastUpdated: ${updated}` : null,
    '---',
    '',
  ]
    .filter((l) => l !== null)
    .join('\n');

  const outFile = path.join(OUT_DOCS, page.isIndex ? 'index.md' : `${page.slug}.md`);
  fs.writeFileSync(outFile, `${frontmatter}\n${body}\n`);
}

console.log(`Imported ${[...pages.values()].filter((p) => p.own).length} pages into ${path.relative(ROOT, OUT_DOCS)}`);
console.log(`Copied ${copiedFigures.size} figures into ${path.relative(ROOT, OUT_IMAGES)}`);
if (missingFigures.length) console.warn(`Missing figures: ${missingFigures.join(', ')}`);
if (droppedLinks.size) console.log(`Unpublished link targets rendered as text: ${droppedLinks.size}`);
