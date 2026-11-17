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
//   - Rewrites `[[Note#Heading|alias]]` links to pages in the same series as site links.
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
const URL_BASE = `/tutorials/${SERIES}/`;

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
};

const files = fs
  .readdirSync(SRC)
  .filter((f) => f.endsWith('.md'))
  .sort();

// Build the page map first, so links between notes can be resolved.
const pages = new Map(); // note name -> { slug, title, order, isIndex }
let order = 1;
for (const file of files) {
  const name = file.replace(/\.md$/, '');
  const isIndex = / - Index$/.test(name);
  if (isIndex) {
    pages.set(name, { name, file, slug: '', title: SERIES_TITLE, order: 0, isIndex });
    continue;
  }
  const { code, title } = parseNoteName(name);
  pages.set(name, { name, file, slug: slugify(title), title: SHORT_TITLES[code] ?? title, order: order++, isIndex });
}

const pageUrl = (page, heading) => {
  const base = page.slug ? `${URL_BASE}${page.slug}/` : URL_BASE;
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
const figureAlt = (file) => {
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
      (data.subtitle ? `<span class="cm-steps-sub">${richText(replaceSectionRefsPlain(data.subtitle))}</span>` : '') +
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


// Worked walkthroughs. A "###" walkthrough whose steps are a numbered list becomes one
// step-by-step card: each step keeps the note's own text (math rendered) and gets a figure,
// taken from the animation that replays the same example or from 15w-gifs.py. `frames` maps
// step k to a 1-based frame of the exported steps (null keeps the step as text only); `extra`
// appends further frames, with their captions, for content the text steps do not cover.
// `replaces` removes the separate animation card, which would otherwise repeat the example.
const WALKTHROUGHS = {
  'The 32,000 against 128,000 comparison': { title: 'Vocabulary size comparison', steps: '15w-vocabulary-size-comparison', frames: [1, 2, 3, 4, 5] },
  'A lookup computed both ways': { title: 'Embedding lookup', steps: '15b-embedding-lookup', frames: [1, null, 4, 5, 6], replaces: true },
  'A complete calculation on three tokens': { title: 'Attention on three tokens', steps: '15c-qkv-walkthrough', frames: [4, 5, 6, 7, 8, 10], replaces: true },
  'A rotation worked to numbers': { title: 'Rotary position embedding', steps: '15c-rope-relative-angle', frames: [2, 3, 4, 5], extra: [6, 7], replaces: true },
  'Three blocks on a two-number stream': { title: 'Residual stream', steps: '15d-residual-stream-accumulation', frames: [1, 2, 4, 5], replaces: true },
  'One vector normalized both ways': { title: 'LayerNorm versus RMSNorm', steps: '15w-layernorm-and-rmsnorm', frames: [1, 2, 3, 4] },
  'A feed-forward pass on a two-number token': { title: 'Feed-forward pass', steps: '15w-feed-forward-pass', frames: [1, 2, 3, 4] },
  'Routing one token through four experts': { title: 'Mixture-of-experts routing', steps: '15d-mixture-of-experts-routing', frames: [null, 1, 2], extra: [8, 9], replaces: true },
  'A near duplicate check by hand': { title: 'MinHash deduplication', steps: '15e-minhash-shingles', frames: [1, null, 5, 6], replaces: true },
  'Two AdamW steps on one parameter': { title: 'AdamW update', steps: '15e-adamw-two-steps', frames: [2, 4, 5, 7, 8, 9], replaces: true },
  'A cosine ranking worked by hand': { title: 'Cosine retrieval', steps: '15h-cosine-retrieval', frames: [2, 5, 9, 10], replaces: true },
  'A rank one LoRA layer worked by hand': { title: 'LoRA forward pass', steps: '15h-lora-forward-pass', frames: [4, 5, 6, 7, 10], extra: [11], replaces: true },
  'A pass at k calculation': { title: 'pass@k', steps: '15i-pass-at-k-walkthrough', frames: [1, 2, 3, 4, 5], extra: [6], replaces: true },
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
    const h = lines.indexOf(`### ${heading}`);
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
        (data.subtitle ? `<span class="cm-steps-sub">${richText(replaceSectionRefsPlain(data.subtitle))}</span>` : '') +
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
  // The series overview reads as one introduction: no "Overview" heading, and no paragraph
  // spotlighting the running toy and 7B examples (each chapter introduces them where they appear).
  [/\n## Overview\n/g, ''],
  [/\n\nTwo running examples recur across the notes\.[^\n]*/g, ''],
  [/\bNumbered sections then build\b/g, 'Sections then build'],
  // The Obsidian index is an interview-prep index; the site speaks to learners and practitioners.
  [/in a product decision or an interview question start to make sense/g, 'in a product decision or a research paper start to make sense'],
  [/Everything here is written for someone new to the field\.[^\n]*/g,
    "This tutorial is for anyone who wants to understand how these models work, from readers new to large language models to engineers and researchers who already build with them. For newcomers, the chapters build on each other in order and cover the key ideas behind large language models step by step, from how text becomes tokens to how a finished model is trained, served and evaluated. Each idea starts with a small example before any math appears, so every step can be followed and checked by hand. For engineers and researchers who know the basics, the series works as a refresher: every chapter and section is one click away in the sidebar, and the deep dives hold the extra detail."],
  [/\n+Each chapter opens with the problem it answers[^\n]*/g, ''],
  [/ These nine chapters follow one model through its whole life[^.]*\./g, ''],
  // The site outlines the merged pair in the BPE table; the note's plain table does not.
  [/The first pair in each row is the one line 6 merges in that round\./g, 'The first pair in each row, outlined, is the one line 6 merges in that round.'],
];

// "15a" -> "chapter 1" in prose. Leaves image paths and URLs alone.
const chapterNo = (letter) => letter.charCodeAt(0) - 96;
function chapterWords(text) {
  return text
    .replace(/(?<=[\s(])15([a-i]), 15([a-i]), and 15([a-i])(?=[\s,.;:)])/g, (_, a, b, c) =>
      `chapters ${chapterNo(a)}, ${chapterNo(b)}, and ${chapterNo(c)}`)
    .replace(/(?<=[\s(])15([a-i]) (to|and) 15([a-i])(?=[\s,.;:)])/g, (_, a, w, b) =>
      `chapters ${chapterNo(a)} ${w} ${chapterNo(b)}`)
    .replace(/(?<=[\s(])15([a-i])(?=[\s,.;:)])/g, (_, a) => `chapter ${chapterNo(a)}`);
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
const indexPage = [...pages.values()].find((p) => p.isIndex);
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

console.log(`Imported ${pages.size} pages into ${path.relative(ROOT, OUT_DOCS)}`);
console.log(`Copied ${copiedFigures.size} figures into ${path.relative(ROOT, OUT_IMAGES)}`);
if (missingFigures.length) console.warn(`Missing figures: ${missingFigures.join(', ')}`);
if (droppedLinks.size) console.log(`Unpublished link targets rendered as text: ${droppedLinks.size}`);
