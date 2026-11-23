// The notebook owns the code and the placement of each excerpt. The Obsidian
// importer can regenerate tutorial prose without copying or replacing cells.
import { readFileSync } from 'node:fs';
import { slug } from 'github-slugger';
import { fromMarkdown } from 'mdast-util-from-markdown';
export const notebookFile = new URL('../public/notebooks/llm-fundamentals.ipynb', import.meta.url);
export const notebookPage = '/tutorials/llm-fundamentals/notebook/';

const html = (value) => ({ type: 'html', value });
const text = (value) => ({ type: 'text', value });
const code = (cell) => ({ type: 'code', lang: 'python', value: cell.source });
const escape = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

export function loadNotebook(project) {
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

// The notebook page groups cells by stage. Stage headings
// group the cells. Each explanation cell becomes prose above its code cell. Stage titles must
// not slug to a cell ID or another heading on the page, since Starlight assigns heading IDs.
export const notebookStages = [
  ['setup', 'Configuration', 'foundations/config.py'],
  ['corpus', 'Data and tokenizer', 'foundations/data_and_tokenizer.py'],
  ['causal-mask', 'Transformer', 'foundations/transformer.py'],
  ['adamw', 'Base training', 'base_model/pretrain.py'],
  ['temperature', 'Generation', 'base_model/generation.py'],
  ['instruction-data', 'Instruction tuning', 'post_training/instruction_tuning.py'],
  ['constraints', 'Evaluation', 'post_training/evaluation.py'],
  ['preference-pairs', 'Experiments', 'post_training/preferences.py'],
];
export const rlNotebook = {
  name: 'rl-agent',
  file: new URL('../public/notebooks/rl-agent.ipynb', import.meta.url),
  page: '/tutorials/rl-for-llm-agents/notebook/',
  pageFile: 'rl-agent-notebook.md',
  series: 'rl-for-llm-agents',
  label: 'Order exchange agent notebook',
  stages: [
    ['store-fixture', 'Environment and verification', 'environment/store.py'],
    ['store-sft-mask', 'Policy gradient foundations', 'training/policy_gradients.py'],
    ['multiturn-trace', 'Multi-turn training', 'training/multi_turn.py'],
    ['repeated-evaluation', 'Evaluation and release', 'evaluation/release.py'],
  ],
};
export const storyNotebook = {
  name: 'llm-fundamentals',
  file: notebookFile,
  readme: 'notebooks/llm-fundamentals/README.md',
  page: notebookPage,
  pageFile: 'story-model-notebook.md',
  series: 'llm-fundamentals',
  label: 'Story model notebook',
  stages: notebookStages,
  // The notebook keeps one Experiments stage, while the explorer splits its topics.
  extraSourceFiles: [
    ['beam-search', 'Decoding and adaptation', 'experiments/decoding_and_adaptation.py'],
    ['retrieval', 'Retrieval and tool use', 'experiments/retrieval_and_tools.py'],
    ['calibration', 'Reliability and evaluation', 'experiments/reliability.py'],
    ['lookup', 'Model internals', 'experiments/model_internals.py'],
    ['cache-budget', 'Inference and serving', 'experiments/serving.py'],
    ['rag', 'Retrieval-augmented generation and privacy', 'experiments/rag_and_privacy.py'],
    ['layernorm-reference', 'Reference implementations', 'experiments/reference.py'],
  ],
};
export const notebookProjects = [storyNotebook, rlNotebook];

export function notebookExpandLink(project, cellId) {
  return `<a class="cm-code-tool cm-notebook-expand" href="/notebooks/${project.name}/source/#${cellId}" data-notebook-expand="${project.name}" aria-label="Open Python source"></a>`;
}
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
  const outline = project.stages.map(([, title]) => `<a href="#${slug(title)}">${escape(title)}</a>`).join('');
  const nodes = [html(`<div class="cm-nb" role="region" aria-label="${escape(project.label)}">
<div class="cm-nb-toolbar not-content">
  <details class="cm-nb-outline">
    <summary><span class="cm-nb-outline-label">Jump to stage</span></summary>
    <nav aria-label="Notebook stages">${outline}</nav>
  </details>
  <button type="button" class="cm-nb-toggle-all" hidden>Show all code</button>
</div>`)];
  let pending = null;
  let sectionOpen = false;
  for (const item of notebook) {
    if (item.cell_type === 'markdown') {
      if (item.id !== 'welcome') pending = explanation(item.source);
      continue;
    }
    const cell = byId.get(item.id);
    const stage = project.stages.find(([id]) => id === cell.id);
    if (stage) {
      if (sectionOpen) nodes.push(html('</section>'));
      nodes.push(html('<section class="cm-nb-section">'));
      sectionOpen = true;
      nodes.push({ type: 'heading', depth: 3, data: { hProperties: { className: ['cm-nb-stage'], tabIndex: -1 } },
        children: [text(stage[1])] });
      if (STAGE_NOTES[cell.id]) nodes.push({ type: 'paragraph', children: [text(STAGE_NOTES[cell.id])] });
    }
    const link = cell.chapter
      ? `<a class="cm-nb-chapter-link" href="/tutorials/${project.series}/${cell.chapter}/#codepad-${cell.id}">Explained in ${escape(CHAPTERS[cell.chapter] ?? 'the tutorial')} <span aria-hidden="true">↗</span></a>`
      : '';
    nodes.push(html(`<section class="cm-nb-cell" id="${cell.id}">
<h4 class="cm-nb-title">${escape(pending?.title ?? cell.title)}</h4>
<div class="cm-nb-explanation">`));
    nodes.push(...(pending?.body ?? [{ type: 'paragraph', children: [text(cell.summary)] }]));
    nodes.push(html(`</div>
<details class="cm-nb-source">
  <summary><span class="cm-nb-show">View code</span><span class="cm-nb-hide">Hide code</span></summary>
  <div class="cm-nb-code">${link}`));
    nodes.push({ ...code(cell), meta: 'wrap hangingIndent=2' });
    nodes.push(html('</div></details></section>'));
    pending = null;
  }
  if (sectionOpen) nodes.push(html('</section>'));
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
  <div class="cm-codepad-frame not-content">`),
    code(cell),
    html(`${notebookExpandLink(project, cell.id)}</div></div></details>`),
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
