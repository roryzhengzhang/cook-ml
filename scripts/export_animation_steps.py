#!/usr/bin/env python3
"""Export each tutorial animation as a sequence of step images plus caption text.

The vault's animations are drawn by ~/.claude/scripts/ml-figures/<id>-gifs.py with a small
`Gif` helper: every frame is one matplotlib figure with a title, a subtitle, and a caption.
This script loads those scripts without running their main block, patches the helper so
each frame is saved as a clean PNG (title and caption left out), and records the title,
subtitle and per-step captions as JSON. The site then shows the animation as numbered
steps with real text, instead of a video.

Nothing is written to the vault.

Usage:
    python3 scripts/export_animation_steps.py            # every 15x- and 16x-gifs.py script
    python3 scripts/export_animation_steps.py 15b 15c    # only these notes
"""

import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mpl-cook-ml'))

import numpy as np  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

SCRIPTS = Path.home() / '.claude/scripts/ml-figures'
ROOT = Path(__file__).resolve().parent.parent
SERIES = 'llm-fundamentals'
# Kept apart from public/images/<series>/, which the Obsidian importer clears on every run.
OUT_IMAGES = ROOT / 'public/images/steps'
OUT_DATA = ROOT / 'src/data/animation-steps'


# Opening frames that only show empty headers or a bare prompt. Their caption already says
# everything, so the step is shown as text without an image. Keyed by animation name, 1-based.
TEXT_ONLY_STEPS = {
    '15a perplexity walkthrough': {1},
    '15g speculative accept reject': {1},
    '15i calibration error walkthrough': {1},
}


def slugify(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


def content_box(img):
    bg = Image.new('RGB', img.size, (255, 255, 255))
    return ImageChops.difference(img, bg).point(lambda v: 255 if v > 8 else 0).getbbox()


def shared_crop(images, pad=12):
    """One crop box for every step of an animation, so the picture does not shift between steps.
    It removes the white margin left where the title and caption used to be."""
    boxes = [b for b in (content_box(i) for i in images) if b]
    if not boxes:
        return lambda img: img
    w, h = images[0].size
    l = max(0, min(b[0] for b in boxes) - pad)
    t = max(0, min(b[1] for b in boxes) - pad)
    r = min(w, max(b[2] for b in boxes) + pad)
    b = min(h, max(b[3] for b in boxes) + pad)
    return lambda img: img.crop((l, t, r, b))


def load(path):
    spec = importlib.util.spec_from_file_location(f'anim_{path.stem.replace("-", "_")}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ is not '__main__', so nothing renders yet
    return mod


def patch(mod, results):
    Gif = mod.Gif
    plt = mod.plt

    def fig(self, title=None, subtitle=None):
        # Same canvas as the original, without the title and subtitle text.
        f = plt.figure(figsize=self.size, dpi=getattr(mod, 'DPI', 100) * 2)
        # Frames that pass their own title or subtitle (e.g. a closing summary) keep it as a step heading.
        f._cm_heading = title if title and title != self.title else (subtitle if subtitle is not None else '')
        return f

    def add(self, f, caption, hold=None):
        f.canvas.draw()
        a = np.asarray(f.canvas.buffer_rgba())[..., :3].copy()
        plt.close(f)
        self.frames.append((Image.fromarray(a), caption or '', getattr(f, '_cm_heading', '')))
        return self.frames[-1][0]

    def save(self):
        steps = []
        for img, caption, heading in self.frames:
            # A run of frames with the same caption is one step drawn in motion; keep its final state.
            if steps and caption == steps[-1]['caption'] and heading == steps[-1]['heading']:
                steps[-1]['image'] = img
            else:
                steps.append({'caption': caption, 'heading': heading, 'image': img})
        results.append({'name': self.name, 'title': self.title, 'subtitle': self.subtitle, 'steps': steps})
        return [s['image'] for s in steps]

    Gif.fig, Gif.add, Gif.save = fig, add, save


def animation_functions(mod):
    fns = getattr(mod, 'ALL', None)
    if fns:
        return list(fns)
    return [
        fn for name, fn in inspect.getmembers(mod, inspect.isfunction)
        if fn.__module__ == mod.__name__ and 'Gif(' in inspect.getsource(fn)
    ]


def export(script):
    mod = load(script)
    results = []
    patch(mod, results)
    for fn in animation_functions(mod):
        fn()

    for anim in results:
        slug = slugify(anim['name'])
        img_dir = OUT_IMAGES / slug
        shutil.rmtree(img_dir, ignore_errors=True)
        img_dir.mkdir(parents=True)
        skip = TEXT_ONLY_STEPS.get(anim['name'], set())
        crop = shared_crop([st['image'] for k, st in enumerate(anim['steps'], 1) if k not in skip])
        steps = []
        text_only = TEXT_ONLY_STEPS.get(anim['name'], set())
        for i, step in enumerate(anim['steps'], 1):
            if i in text_only:
                entry = {'caption': step['caption']}
                if step['heading']:
                    entry['heading'] = step['heading']
                steps.append(entry)
                continue
            name = f'step-{i:02d}.png'
            crop(step['image']).quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
                img_dir / name, optimize=True)
            entry = {'caption': step['caption'], 'image': f'/images/steps/{slug}/{name}'}
            if step['heading']:
                entry['heading'] = step['heading']
            steps.append(entry)
        data = {'name': anim['name'], 'title': anim['title'], 'subtitle': anim['subtitle'], 'steps': steps}
        (OUT_DATA / f'{slug}.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        print(f'{anim["name"]}: {len(steps)} steps')


def main():
    wanted = sys.argv[1:]
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    scripts = sorted([*SCRIPTS.glob('15?-gifs.py'), *SCRIPTS.glob('16?-gifs.py'), *SCRIPTS.glob('17?-gifs.py')])
    for script in scripts:
        if wanted and not any(script.name.startswith(w) for w in wanted):
            continue
        export(script)


if __name__ == '__main__':
    main()
