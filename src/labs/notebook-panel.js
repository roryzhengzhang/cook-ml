const panel = document.getElementById('cm-notebook-panel');
const triggers = document.querySelectorAll('a[data-notebook-expand]');
if (panel && triggers.length) connectNotebookPanel(panel, triggers);

function connectNotebookPanel(panel, triggers) {
  const root = document.documentElement;
  const compact = matchMedia('(width < 56rem)');
  const chapter = panel.closest('.main-frame');
  const header = document.querySelector('.page > .header');
  const frame = panel.querySelector('iframe');
  const title = panel.querySelector('h2');
  const download = panel.querySelector('[data-notebook-download]');
  const close = panel.querySelector('[data-notebook-close]');
  const divider = panel.querySelector('[role="separator"]');
  let ratio = 46;
  let drag;
  let opener;

  function chapterScroller() {
    return root.hasAttribute('data-cm-notebook-split') ? chapter : window;
  }

  function captureReadingPosition() {
    const bounds = chapter.getBoundingClientRect();
    const edge = header.getBoundingClientRect().bottom;
    const openerTop = opener?.getBoundingClientRect().top;
    const anchor = openerTop >= edge && openerTop < innerHeight
      ? opener
      : document.elementFromPoint(bounds.left + chapter.clientWidth / 2, edge + 8);
    return chapter.contains(anchor) ? { anchor, top: anchor.getBoundingClientRect().top } : null;
  }

  function restoreReadingPosition(position) {
    if (!position) return;
    const top = position.anchor.getBoundingClientRect().top - position.top;
    chapterScroller().scrollBy({ top, behavior: 'instant' });
  }

  function setChapterSplit(split) {
    if (split === root.hasAttribute('data-cm-notebook-split')) return;
    const position = captureReadingPosition();
    const top = split ? window.scrollY : chapter.scrollTop;
    root.toggleAttribute('data-cm-notebook-split', split);
    if (split) {
      chapter.tabIndex = 0;
      window.scrollTo({ top: 0, behavior: 'instant' });
    } else {
      chapter.removeAttribute('tabindex');
    }
    chapterScroller().scrollTo({ top, behavior: 'instant' });
    restoreReadingPosition(position);
  }

  function resize(requested = ratio) {
    if (compact.matches) return;
    const position = panel.open ? captureReadingPosition() : null;
    const minimum = Math.max(25, Math.ceil(320 / innerWidth * 100));
    const maximum = 100 - minimum;
    ratio = Math.min(maximum, Math.max(minimum, requested));
    root.style.setProperty('--cm-notebook-panel-width', `${ratio}vw`);
    const notebookPercent = Math.round(ratio);
    divider.setAttribute('aria-valuemin', minimum);
    divider.setAttribute('aria-valuemax', maximum);
    divider.setAttribute('aria-valuenow', notebookPercent);
    divider.setAttribute('aria-valuetext', `Notebook ${notebookPercent}%, chapter ${100 - notebookPercent}%`);
    restoreReadingPosition(position);
  }

  function show() {
    resize();
    setChapterSplit(!compact.matches);
    root.setAttribute('data-cm-notebook-open', '');
    if (compact.matches) panel.showModal();
    else panel.show();
  }

  function closeOnEscape(event) {
    if (event.key !== 'Escape' || !panel.open || event.defaultPrevented) return;
    event.preventDefault();
    panel.close();
  }

  function setExpanded(trigger, expanded) {
    trigger.setAttribute('aria-expanded', String(expanded));
    if (trigger.classList.contains('cm-notebook-expand')) {
      trigger.setAttribute('aria-label', expanded ? 'Close notebook panel' : 'Open Python source');
    }
  }

  for (const trigger of triggers) {
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.setAttribute('aria-controls', panel.id);
    setExpanded(trigger, false);
    trigger.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

      const target = new URL(trigger.href);
      if (target.origin !== location.origin) return;
      event.preventDefault();
      if (panel.open && opener === trigger) {
        panel.close();
        return;
      }
      if (opener) setExpanded(opener, false);
      opener = trigger;
      setExpanded(opener, true);
      const project = trigger.dataset.notebookExpand;
      title.textContent = project;
      download.href = `/notebooks/${project}.py`;
      if (!panel.open) show();
      const frameDocument = frame.contentDocument;
      if (frameDocument?.location.pathname === target.pathname) {
        if (frameDocument.location.hash !== target.hash) frame.contentWindow.location.replace(target.href);
        else frameDocument.dispatchEvent(new Event('cookml:source-jump'));
      } else {
        frame.src = target.href;
      }
      close.focus({ preventScroll: true });
    });
  }

  close.addEventListener('click', () => panel.close());
  document.addEventListener('keydown', closeOnEscape);
  frame.addEventListener('load', () => {
    frame.contentDocument?.addEventListener('keydown', closeOnEscape);
  });

  divider.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || !event.isPrimary) return;
    drag = { x: event.clientX, ratio };
    divider.setPointerCapture(event.pointerId);
    root.setAttribute('data-cm-notebook-resizing', '');
  });
  divider.addEventListener('pointermove', (event) => {
    if (drag && divider.hasPointerCapture(event.pointerId)) {
      resize(drag.ratio + (drag.x - event.clientX) / innerWidth * 100);
    }
  });
  divider.addEventListener('lostpointercapture', () => {
    drag = undefined;
    root.removeAttribute('data-cm-notebook-resizing');
  });
  divider.addEventListener('keydown', (event) => {
    const widths = { ArrowLeft: ratio + 3, ArrowRight: ratio - 3, Home: 0, End: 100 };
    if (!Object.hasOwn(widths, event.key)) return;
    event.preventDefault();
    resize(widths[event.key]);
  });
  window.addEventListener('resize', () => {
    if (panel.open) resize();
  });

  panel.addEventListener('close', () => {
    if (panel.open) return;
    drag = undefined;
    root.removeAttribute('data-cm-notebook-resizing');
    root.removeAttribute('data-cm-notebook-open');
    setChapterSplit(false);
    if (opener) {
      setExpanded(opener, false);
      opener.focus({ preventScroll: true });
    }
  });

  compact.addEventListener('change', () => {
    if (panel.open) {
      panel.close();
      show();
    }
  });
}
