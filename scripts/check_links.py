#!/usr/bin/env python3
"""Check the built site for broken anchors and math errors.

Run after `npx astro build`:
    python3 scripts/check_links.py

Checks every `href="...#id"` inside article content and in the tutorial sidebar resolves to an
element id on the target page, and that no page contains a KaTeX error. Exits 1 on any problem.
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dist')
NOTEBOOKS = {
    'llm-fundamentals': 'llm-fundamentals.ipynb',
    'rl-for-llm-agents': 'rl-agent.ipynb',
}


def check_notebook(series, filename, ids, problems):
    series_path = f'/tutorials/{series}/'
    source = (Path(DIST).parent / 'public/notebooks' / filename).read_bytes()
    download = Path(DIST) / 'notebooks' / filename
    if not download.is_file() or download.read_bytes() != source:
        problems.append(f'{filename}: download does not match the source notebook')
    web_cells = ids.get(series_path + 'notebook/', set())
    for cell in json.loads(source)['cells']:
        if cell['cell_type'] != 'code':
            continue
        cell_id = cell['id']
        if cell_id not in web_cells:
            problems.append(f'Notebook web view is missing cell {cell_id}')
        chapter = cell['metadata']['cookml'].get('chapter')
        if chapter and f'codepad-{cell_id}' not in ids.get(f'{series_path}{chapter}/', set()):
            problems.append(f'{chapter}: missing codepad for {cell_id}')


def main():
    pages = {}
    for f in glob.glob(os.path.join(DIST, 'tutorials', '**', 'index.html'), recursive=True):
        url = '/' + os.path.relpath(os.path.dirname(f), DIST).replace(os.sep, '/') + '/'
        pages[url] = open(f, encoding='utf-8').read()
    if not pages:
        sys.exit('No built tutorial pages found. Run `npx astro build` first.')

    ids = {u: set(re.findall(r'id="([^"]+)"', h)) for u, h in pages.items()}
    problems = []
    checked = 0

    for url, html in pages.items():
        content = html.split('class="sl-markdown-content"', 1)[-1]
        for href in re.findall(r'href="([^"]*#[^"]+)"', content):
            checked += 1
            path, frag = href.split('#', 1)
            target = path or url
            if target in ids and frag not in ids[target]:
                problems.append(f'{url}: broken link {href}')
        if 'katex-error' in html:
            problems.append(f'{url}: KaTeX error')

    # The sidebar is the same on every chapter page; /tutorials/ itself has none.
    sidebar = pages.get('/tutorials/llm-fundamentals/', '')
    for path, frag in re.findall(r'href="(/tutorials/[^"#]+/)#([^"]+)"', sidebar):
        checked += 1
        if frag not in ids.get(path, set()):
            problems.append(f'sidebar: broken link {path}#{frag}')

    for series, filename in NOTEBOOKS.items():
        check_notebook(series, filename, ids, problems)
    print(f'{len(pages)} pages, {checked} anchors checked, {len(problems)} problems')
    for p in problems:
        print('  ' + p)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
