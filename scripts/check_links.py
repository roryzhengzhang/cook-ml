#!/usr/bin/env python3
"""Check the built site for broken anchors and math errors.

Run after `npx astro build`:
    python3 scripts/check_links.py

Checks every `href="...#id"` inside article content and in the tutorial sidebar resolves to an
element id on the target page, and that no page contains a KaTeX error. Exits 1 on any problem.
"""

import glob
import os
import re
import sys

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dist')


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

    print(f'{len(pages)} pages, {checked} anchors checked, {len(problems)} problems')
    for p in problems:
        print('  ' + p)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
