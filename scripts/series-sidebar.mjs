// Build a Starlight sidebar for a tutorial series where every chapter is a
// collapsible group listing its "##" sections as anchor links.
//
// Starlight opens the group that contains the current page and leaves the
// others collapsed, so readers see the outline of the chapter they are in.

import fs from 'node:fs';
import path from 'node:path';
import GithubSlugger from 'github-slugger';
import { notebookPage, notebookStages } from './remark-story-notebook.mjs';

const DOCS = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../src/content/docs');

function readFrontmatter(text) {
  const block = text.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
  const get = (key) => block.match(new RegExp(`^\\s*${key}:\\s*(.+)$`, 'm'))?.[1].trim().replace(/^"|"$/g, '');
  return { title: get('title'), order: Number(get('order') ?? 0), label: get('label') };
}

// Heading text as it is rendered: Markdown links and emphasis stripped.
const headingText = (raw) =>
  raw
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_`]/g, '')
    .trim();

// Slug every heading in document order, exactly as Astro does, so ids match.
function sections(markdown, maxDepth = 2) {
  const slugger = new GithubSlugger();
  const found = [];
  let inFence = false;
  for (const line of markdown.replace(/^---\n[\s\S]*?\n---\n/, '').split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;
    if (inFence) continue;
    // A "##" section shown as a deep-dive card keeps its heading as raw HTML with a fixed id.
    const html = line.match(/<h2 class="cm-collapse-title" id="([^"]+)">(.*?)<\/h2>/);
    if (html) {
      found.push({ text: html[2].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&'), id: html[1], depth: 2 });
      continue;
    }
    const m = line.match(/^(#{2,6})\s+(.*?)\s*#*\s*$/);
    if (!m) continue;
    const text = headingText(m[2]);
    const id = slugger.slug(text);
    const depth = m[1].length;
    if (depth <= maxDepth) found.push({ text, id, depth });
  }
  return found;
}

export const notebookPageFile = path.join(DOCS, 'story-model-notebook.md');

export function notebookSidebar() {
  const markdown = fs.readFileSync(notebookPageFile, 'utf8').replace(
    '<!-- notebook -->',
    notebookStages.map(([, title]) => `### ${title}`).join('\n'),
  );
  return {
    label: 'Project notebook',
    collapsed: true,
    items: [
      { label: 'Introduction', link: notebookPage },
      ...sections(markdown, 3).map(({ text, id, depth }) => ({
        label: text,
        link: `${notebookPage}#${id}`,
        ...(depth === 3 && { attrs: { class: 'cm-sidebar-subsection' } }),
      })),
    ],
  };
}

export function seriesSidebar(series, { overviewLabel = 'Overview' } = {}) {
  const dir = path.join(DOCS, 'tutorials', series);
  const base = `/tutorials/${series}/`;
  const chapters = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.md') && f !== 'index.md')
    .map((file) => {
      const text = fs.readFileSync(path.join(dir, file), 'utf8');
      const fm = readFrontmatter(text);
      const href = `${base}${file.replace(/\.md$/, '')}/`;
      return {
        order: fm.order,
        label: fm.label ?? fm.title,
        collapsed: true,
        items: [
          { label: 'Introduction', link: href },
          ...sections(text).map((s) => ({ label: s.text, link: `${href}#${s.id}` })),
        ],
      };
    })
    .sort((a, b) => a.order - b.order)
    .map(({ order, ...group }) => group);

  return [{ label: overviewLabel, link: base }, ...chapters];
}
