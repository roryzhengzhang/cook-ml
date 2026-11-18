// The notebook owns the code and the placement of each excerpt. The Obsidian
// importer can regenerate tutorial prose without copying or replacing cells.
import { readFileSync } from 'node:fs';
import { fromMarkdown } from 'mdast-util-from-markdown';
export const notebookFile = new URL('../public/notebooks/llm-fundamentals.ipynb', import.meta.url);
export const notebookPage = '/tutorials/llm-fundamentals/notebook/';

const html = (value) => ({ type: 'html', value });
const text = (value) => ({ type: 'text', value });
const code = (cell) => ({ type: 'code', lang: 'python', value: cell.source });
const escape = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

function loadNotebook(project) {
  return JSON.parse(readFileSync(project.file, 'utf8')).cells.map((item) => ({
    ...item, source: Array.isArray(item.source) ? item.source.join('') : item.source,
  }));
}

function loadCells(notebook) {
  const seen = new Set();
  const cells = [];
  for (const item of notebook) {
    if (item.cell_type !== 'code') continue;
    const { id } = item;
    const placement = item.metadata.cookml;
    if (!id || !/^[a-z0-9-]+$/.test(id) || seen.has(id)) throw new Error(`Invalid notebook cell ID: ${id}`);
    if (!placement?.title || !placement.summary || Boolean(placement.chapter) !== Boolean(placement.section)) {
      throw new Error(`Invalid codepad metadata for ${id}`);
    }
    seen.add(id);
    cells.push({ id, ...placement, source: item.source });
  }
  return cells;
}

// The notebook page shows every cell inside one notebook frame. Stage headings
// group the cells. Each explanation cell becomes prose above its code cell. Stage titles must
// not slug to a cell ID or another heading on the page, since Starlight assigns heading IDs.
export const notebookStages = [
  ['setup', 'Configuration'], ['corpus', 'Data and tokenizer'], ['causal-mask', 'Transformer'],
  ['adamw', 'Base training'], ['temperature', 'Generation'], ['instruction-data', 'Instruction tuning'],
  ['constraints', 'Evaluation'], ['preference-pairs', 'Experiments'],
];
export const rlNotebook = {
  file: new URL('../public/notebooks/rl-agent.ipynb', import.meta.url),
  page: '/tutorials/rl-for-llm-agents/notebook/',
  pageFile: 'rl-agent-notebook.md',
  series: 'rl-for-llm-agents',
  label: 'Order exchange agent notebook',
  stages: [
    ['store-fixture', 'Environment and verification'],
    ['store-sft-mask', 'Policy gradient foundations'],
    ['multiturn-trace', 'Multi-turn training'],
    ['repeated-evaluation', 'Evaluation and release'],
  ],
};
export const storyNotebook = {
  file: notebookFile,
  page: notebookPage,
  pageFile: 'story-model-notebook.md',
  series: 'llm-fundamentals',
  label: 'Story model notebook',
  stages: notebookStages,
};
export const notebookProjects = [storyNotebook, rlNotebook];
const STAGE_NOTES = {
  'preference-pairs': 'These cells extend the project past the main run. Cells that train or sample extra models run only when COOK_ML_LABS=1. The rest define small, checked functions that the chapters discuss.',
};

function explanation(markdown) {
  const lines = markdown.trim().split('\n')
    .filter((line) => !/^\[Read the tutorial section\]\(/.test(line.trim()));
  const title = lines[0].startsWith('## ') ? lines.shift().slice(3).trim() : null;
  return { title, body: fromMarkdown(lines.join('\n').trim()).children };
}

function notebookFrame(notebook, project) {
  const cells = loadCells(notebook);
  const byId = new Map(cells.map((cell) => [cell.id, cell]));
  const nodes = [html(`<div class="cm-nb" role="region" aria-label="${project.label}">`)];
  let pending = null, count = 0;
  for (const item of notebook) {
    if (item.cell_type === 'markdown') {
      if (item.id !== 'welcome') pending = explanation(item.source);
      continue;
    }
    const cell = byId.get(item.id);
    const stage = project.stages.find(([id]) => id === cell.id);
    if (stage) {
      nodes.push({ type: 'heading', depth: 3, data: { hProperties: { className: ['cm-nb-stage'] } },
        children: [text(stage[1])] });
      if (STAGE_NOTES[cell.id]) nodes.push({ type: 'paragraph', children: [text(STAGE_NOTES[cell.id])] });
    }
    count += 1;
    const link = cell.chapter
      ? `<a href="/tutorials/${project.series}/${cell.chapter}/#codepad-${cell.id}">Explained in ${escape(CHAPTERS[cell.chapter] ?? 'the tutorial')} <span aria-hidden="true">↗</span></a>`
      : '';
    nodes.push(html(`<section class="cm-nb-cell" id="${cell.id}">
<div class="cm-nb-head"><span class="cm-nb-title">${escape(pending?.title ?? cell.title)}</span>${link}</div>`));
    nodes.push(...(pending?.body ?? [{ type: 'paragraph', children: [text(cell.summary)] }]));
    nodes.push(html(`<div class="cm-nb-code"><span class="cm-nb-prompt" aria-label="Code cell ${count}">In [${count}]</span>`));
    nodes.push(code(cell));
    nodes.push(html('</div></section>'));
    pending = null;
  }
  nodes.push(html('</div>'));
  return nodes;
}

const CHAPTERS = {
  'language-modeling-and-next-token-prediction': 'Language Modeling',
  'tokenization-and-embeddings': 'Tokenization',
  'attention-and-positional-information': 'Attention',
  'the-transformer-block-and-model-anatomy': 'Transformer Block',
  'pretraining-at-scale': 'Pretraining',
  'post-training-from-base-model-to-assistant': 'Post-Training',
  'inference-and-decoding': 'Inference',
  'using-llms-in-applications': 'LLM Applications',
  'evaluation-reliability-and-safety': 'Evaluation and Safety',
};

function codepad(cell, project) {
  return [
    html(`<details class="cm-codepad" id="codepad-${cell.id}">
<summary>
  <span class="cm-codepad-icon" aria-hidden="true"></span>
  <span class="cm-codepad-text">
    <span class="cm-codepad-kicker">Code</span>
    <span class="cm-codepad-title">${escape(cell.title)}</span>
  </span>
  <span class="cm-collapse-hint" aria-hidden="true"></span>
</summary>
<div class="cm-codepad-body">
  <p class="cm-codepad-summary">${escape(cell.summary)}</p>
  <div class="cm-codepad-links"><a href="${project.page}#${cell.id}">Notebook cell <span aria-hidden="true">↗</span></a></div>`),
    code(cell),
    html('</div></details>'),
  ];
}

const headingText = (node) => node?.value ?? (node?.children ?? []).map(headingText).join('');

function chapterSection(node) {
  if (node.type === 'heading' && node.depth === 2) return headingText(node);
  if (node.type !== 'html') return null;
  return node.value.match(/<h2 class="cm-collapse-title" id="[^"]+">([^<]+)<\/h2>/)?.[1]?.replaceAll('&amp;', '&');
}

function replaceText(node, before, after) {
  if (node.type === 'text') node.value = node.value.replace(before, after);
  for (const child of node.children ?? []) replaceText(child, before, after);
}

export default function remarkStoryNotebook() {
  return (tree, file) => {
    const filename = (file.path ?? file.history?.[0] ?? '').replaceAll('\\', '/');
    const chapter = /\/tutorials\/([^/]+)\/([^/]+)\.md$/.exec(filename);
    const project = notebookProjects.find((entry) =>
      entry.series === chapter?.[1] || filename.endsWith(`/${entry.pageFile}`));
    if (!project) return;
    const fromChapter = chapter?.[2];
    const notebookView = filename.endsWith(`/${project.pageFile}`);
    if (notebookView) {
      const marker = tree.children.findIndex((node) => node.type === 'html' && node.value.trim() === '<!-- notebook -->');
      if (marker < 0) throw new Error(`${project.pageFile} needs a <!-- notebook --> marker`);
      // Keep prose subsections in the outline and accessible as headings.
      for (const node of tree.children) {
        if (node.type === 'heading' && node.depth === 3) {
          node.data = { ...node.data, hProperties: { className: ['sr-only'] } };
        }
      }
      tree.children.splice(marker, 1, ...notebookFrame(loadNotebook(project), project));
      return;
    }
    if (fromChapter === 'index') {
      replaceText(tree, 'Every example is still worked by hand rather than in code, so no programming is needed.',
        'The worked examples can be followed by hand. The optional codepads connect them to the runnable model.');
      return;
    }

    const selected = loadCells(loadNotebook(project)).filter((cell) => cell.chapter === fromChapter);
    const inserted = new Set();
    const rewritten = [];
    let section;
    function appendCodepads() {
      for (const cell of selected.filter((entry) => entry.section === section)) {
        rewritten.push(...codepad(cell, project));
        inserted.add(cell.id);
      }
    }
    for (const node of tree.children) {
      const nextSection = chapterSection(node);
      if (nextSection) {
        appendCodepads();
        section = nextSection;
      }
      // Keep the chapter's related links after its last implementation.
      if (node.type === 'paragraph' && headingText(node).startsWith('Related:')) {
        appendCodepads();
        section = undefined;
      }
      rewritten.push(node);
    }
    appendCodepads();
    for (const { id, section } of selected) {
      if (!inserted.has(id)) throw new Error(`Codepad ${id}: section "${section}" is missing from ${fromChapter}`);
    }
    tree.children = rewritten;
  };
}
