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
  const { title } = parseNoteName(name);
  pages.set(name, { name, file, slug: slugify(title), title, order: order++, isIndex });
}

const pageUrl = (page, heading) => {
  const base = page.slug ? `${URL_BASE}${page.slug}/` : URL_BASE;
  return heading ? `${base}#${new GithubSlugger().slug(heading)}` : base;
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
  [/\b(across|of|in) this folder\b/g, '$1 this series'],
  [/every note in this folder/g, 'every chapter in this series'],
  [/the roughly 1\.8 million words of notes in this vault/g, 'about 1.8 million words of the author\'s study notes'],
  // Index-only references to unpublished notes.
  [/ Interview practice for this folder lives in the question bank at \[\[17 LLM Fundamentals\]\][^.]*\./g, ''],
  [/\n\nThis folder is the end-to-end entrance\.[^\n]*/g, ''],
  [/, then work through \[\[17 LLM Fundamentals\]\] and answer every collapsed question aloud before opening it/g, ''],
  [/\n## Related topics\n[\s\S]*?(?=\n\*\*Last updated|$)/g, '\n'],
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

// The index note has one "### [[Note]]" heading per chapter followed by a summary paragraph.
// Use that paragraph as the chapter's description.
const indexPage = [...pages.values()].find((p) => p.isIndex);
const summaries = new Map();
if (indexPage) {
  const indexRaw = fs.readFileSync(path.join(SRC, indexPage.file), 'utf8');
  for (const m of indexRaw.matchAll(/^### \[\[([^\]|]+)\]\]\s*\n\s*\n([^\n]+)/gm)) {
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
  const body = tidy(convertBody(withoutH1)).replace(/^---\s*\n/, '').replace(/\n---\s*$/, '');

  const summary = summaries.get(page.name);
  const description = summary ? firstParagraph(convertBody(summary)) : firstParagraph(body);
  const frontmatter = [
    '---',
    `title: ${yamlString(page.title)}`,
    `description: ${yamlString(description)}`,
    'sidebar:',
    `  order: ${page.order}`,
    `  label: ${yamlString(page.isIndex ? 'Overview' : `${page.order}. ${page.title}`)}`,
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
