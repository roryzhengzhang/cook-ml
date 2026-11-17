// Collects the external links in the tutorials and saves a short preview for each one to
// public/link-previews.json, which public/link-preview.js reads when a reader hovers a link.
//
//   npm run link-previews            fetch previews for links that have none yet
//   npm run link-previews -- --all   fetch every link again
//
// A browser cannot read another site's page (cross-origin), so previews are gathered here,
// ahead of time. arXiv papers come from the arXiv API (title, authors, date, abstract);
// other pages from their <title>, description and og:image tags. A link whose fetch fails
// keeps no entry, and the hover card then shows only its address.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CONTENT = path.join(ROOT, 'src/content/docs');
const OUT = path.join(ROOT, 'public/link-previews.json');
const REFRESH_ALL = process.argv.includes('--all');
const UA = 'Mozilla/5.0 (compatible; CookML-link-previews/1.0; +https://cook-ml.com)';

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? walk(p) : /\.mdx?$/.test(e.name) ? [p] : [];
  });
}

const urls = new Set();
for (const file of walk(CONTENT)) {
  const text = fs.readFileSync(file, 'utf8');
  for (const m of text.matchAll(/\]\((https?:\/\/[^)\s]+)\)/g)) urls.add(m[1]);
  for (const m of text.matchAll(/href="(https?:\/\/[^"]+)"/g)) urls.add(m[1]);
}

const previews = !REFRESH_ALL && fs.existsSync(OUT) ? JSON.parse(fs.readFileSync(OUT, 'utf8')) : {};
const todo = [...urls].filter((u) => !previews[u]).sort();
console.log(`${urls.size} external links, ${todo.length} to fetch`);

const decode = (s) =>
  s
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&');
const clean = (s) => decode(s ?? '').replace(/\s+/g, ' ').trim();
const clip = (s, n) => (s.length > n ? s.slice(0, n - 1).replace(/\s+\S*$/, '') + '…' : s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function get(url, timeout = 15000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    return await fetch(url, { headers: { 'user-agent': UA }, redirect: 'follow', signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ---- arXiv: one API request per 50 papers, three seconds apart as the API asks.
const ARXIV = /^https?:\/\/arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})(?:v\d+)?/;
const arxivTodo = todo.filter((u) => ARXIV.test(u));
const byId = new Map();
for (const u of arxivTodo) {
  const id = u.match(ARXIV)[1];
  byId.set(id, [...(byId.get(id) ?? []), u]);
}
const ids = [...byId.keys()];
for (let i = 0; i < ids.length; i += 50) {
  const batch = ids.slice(i, i + 50);
  if (i) await sleep(3000);
  try {
    const res = await get(`https://export.arxiv.org/api/query?id_list=${batch.join(',')}&max_results=50`, 30000);
    const xml = await res.text();
    for (const entry of xml.split('<entry>').slice(1)) {
      const id = entry.match(/<id>[^<]*\/abs\/(\d{4}\.\d{4,5})/)?.[1];
      if (!id || !byId.has(id)) continue;
      const authors = [...entry.matchAll(/<author>\s*<name>([^<]+)<\/name>/g)].map((m) => clean(m[1]));
      const preview = {
        site: 'arXiv',
        title: clean(entry.match(/<title>([\s\S]*?)<\/title>/)?.[1]),
        authors: authors.length > 3 ? `${authors.slice(0, 3).join(', ')} et al.` : authors.join(', '),
        date: entry.match(/<published>(\d{4}-\d{2})/)?.[1],
        description: clip(clean(entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]), 420),
      };
      if (preview.title) for (const u of byId.get(id)) previews[u] = preview;
    }
    console.log(`arXiv: ${Math.min(i + 50, ids.length)}/${ids.length}`);
  } catch (e) {
    console.warn(`arXiv batch ${i / 50 + 1} failed: ${e.message}`);
  }
}

// ---- Other pages: title, description and og:image from the page head.
const meta = (html, ...names) => {
  for (const n of names) {
    const re1 = new RegExp(`<meta[^>]+(?:property|name)=["']${n}["'][^>]*content=["']([^"']*)["']`, 'i');
    const re2 = new RegExp(`<meta[^>]+content=["']([^"']*)["'][^>]*(?:property|name)=["']${n}["']`, 'i');
    const m = html.match(re1) ?? html.match(re2);
    if (m && m[1].trim()) return clean(m[1]);
  }
  return '';
};
const siteName = (u) => new URL(u).hostname.replace(/^www\./, '');

for (const u of todo.filter((x) => !ARXIV.test(x))) {
  try {
    const res = await get(u);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const type = res.headers.get('content-type') ?? '';
    if (type.includes('pdf') || /\.pdf$/i.test(new URL(res.url).pathname)) {
      previews[u] = { site: siteName(u), title: decodeURIComponent(path.basename(new URL(u).pathname)), kind: 'PDF' };
    } else if (type.includes('html')) {
      const html = (await res.text()).slice(0, 400_000);
      const title = meta(html, 'og:title', 'twitter:title') || clean(html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]);
      let image = meta(html, 'og:image', 'og:image:url', 'twitter:image');
      if (image) image = new URL(image, res.url).href;
      const p = {
        site: meta(html, 'og:site_name') || siteName(u),
        title: clip(title, 160),
        description: clip(meta(html, 'og:description', 'description', 'twitter:description'), 300),
      };
      // Site-wide taglines say nothing about the linked page.
      if (/^We.re on a journey to advance and democratize/.test(p.description)) delete p.description;
      if (image) p.image = image;
      if (p.title || p.description) previews[u] = p;
    } else if (type.includes('json')) {
      previews[u] = { site: siteName(u), title: decodeURIComponent(path.basename(new URL(u).pathname)), kind: 'JSON file' };
    }
    console.log(`${previews[u] ? 'ok  ' : 'none'} ${u}`);
  } catch (e) {
    console.warn(`fail ${u}: ${e.message}`);
  }
}

// Keep only links still used, sorted, so the file diffs cleanly.
const out = Object.fromEntries(Object.keys(previews).filter((u) => urls.has(u)).sort().map((u) => [u, previews[u]]));
fs.writeFileSync(OUT, JSON.stringify(out, null, 1) + '\n');
console.log(`Saved ${Object.keys(out).length} of ${urls.size} previews to ${path.relative(ROOT, OUT)}`);
