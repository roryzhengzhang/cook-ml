// Hover previews for links in tutorial text.
//
// Hovering (or keyboard-focusing) a link in a page's text opens a small card about where the
// link goes:
//   - a section of a tutorial page: the page and section title, the section's first figure as a
//     thumbnail, and its opening paragraph;
//   - the model card: its title and spec table;
//   - a whole tutorial page: its title, description and first figure;
//   - an external page: the title, authors, date and abstract (arXiv) or the title, description
//     and image (other sites), read from /link-previews.json, which
//     scripts/fetch-link-previews.mjs builds ahead of time.
// Pages are fetched once and cached. Touch screens get no previews.
(() => {
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

  const SHOW_DELAY = 350;
  const HIDE_DELAY = 200;
  const docs = new Map();
  let external = null;
  let card = null;
  let activeLink = null;
  let showTimer = 0;
  let hideTimer = 0;
  let request = 0;

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  };

  function previewLink(target) {
    const a = target instanceof Element ? target.closest('a[href]') : null;
    if (!a || !a.closest('.sl-markdown-content') || a.closest('.cm-link-preview')) return null;
    if (a.classList.contains('sl-anchor-link')) return null;
    const href = a.getAttribute('href');
    if (!href || href.startsWith('mailto:') || href.startsWith('javascript:')) return null;
    return a;
  }

  function getDoc(path) {
    if (path === location.pathname) return Promise.resolve(document);
    if (!docs.has(path)) {
      docs.set(
        path,
        fetch(path)
          .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
          .then((html) => new DOMParser().parseFromString(html, 'text/html')),
      );
    }
    return docs.get(path);
  }

  function getExternal() {
    external ??= fetch('/link-previews.json')
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({}));
    return external;
  }

  // An image that fails to load (an expired or blocked thumbnail) is removed, not shown broken.
  function thumb(src, alt) {
    const img = el('img', 'cm-lp-thumb');
    img.src = src;
    img.alt = alt || '';
    img.decoding = 'async';
    img.addEventListener('error', () => img.remove());
    return img;
  }

  // A copy of a paragraph from the target page, with links made plain so the card stays a preview.
  function excerpt(p) {
    const copy = p.cloneNode(true);
    copy.className = 'cm-lp-text';
    copy.removeAttribute('id');
    copy.querySelectorAll('a').forEach((a) => a.replaceWith(...a.childNodes));
    return copy;
  }

  const isTextParagraph = (p) => p.tagName === 'P' && p.textContent.trim().length > 40 && !p.querySelector(':scope > img');
  const pageTitle = (doc) => doc.querySelector('main h1')?.textContent.trim() || doc.title.split('|')[0].trim();
  const absolute = (src, base) => new URL(src, base).href;

  // Everything after a heading up to the next heading of the same or a higher level.
  function sectionBlocks(heading) {
    const level = Number(heading.tagName.slice(1));
    const start = heading.closest('.sl-heading-wrapper') ?? heading;
    const blocks = [];
    for (let n = start.nextElementSibling; n; n = n.nextElementSibling) {
      const h = n.matches('.sl-heading-wrapper') ? n.querySelector('h2, h3, h4, h5, h6') : n.matches('h2, h3, h4, h5, h6') ? n : null;
      if (h && Number(h.tagName.slice(1)) <= level) break;
      blocks.push(n);
    }
    return blocks;
  }

  function firstFigure(blocks) {
    for (const b of blocks) {
      const img = b.matches('p') ? b.querySelector(':scope > img') : null;
      if (img) return img;
    }
    for (const b of blocks) {
      const img = b.querySelector?.('img');
      if (img) return img;
    }
    return null;
  }

  function firstParagraph(blocks) {
    for (const b of blocks) {
      if (isTextParagraph(b)) return b;
      const inner = [...(b.querySelectorAll?.('p') ?? [])].find(isTextParagraph);
      if (inner) return inner;
    }
    return null;
  }

  function header(kicker, title) {
    const head = el('div', 'cm-lp-head');
    if (kicker) head.append(el('div', 'cm-lp-kicker', kicker));
    head.append(el('div', 'cm-lp-title', title));
    return head;
  }

  async function internalCard(url) {
    const doc = await getDoc(url.pathname);
    const page = pageTitle(doc);
    const id = decodeURIComponent(url.hash.slice(1));
    const target = id && id !== '_top' ? doc.getElementById(id) : null;
    const body = el('div', 'cm-lp-body');

    const modelCard = target?.closest('.cm-model-card');
    if (modelCard) {
      const title = modelCard.querySelector('.cm-model-title, .cm-notebook-title')?.cloneNode(true);
      if (title) {
        title.className = 'cm-lp-model-title';
        body.append(title);
      }
      const table = modelCard.querySelector('table');
      if (table) {
        const copy = table.cloneNode(true);
        copy.querySelectorAll('a').forEach((a) => a.replaceWith(...a.childNodes));
        copy.className = 'cm-lp-table';
        body.append(copy);
      }
      return body;
    }

    if (target && /^H[2-6]$/.test(target.tagName)) {
      const blocks = sectionBlocks(target);
      const img = firstFigure(blocks);
      if (img) body.append(thumb(absolute(img.getAttribute('src'), url), img.alt));
      body.append(header(page, target.textContent.trim()));
      const p = firstParagraph(blocks);
      if (p) body.append(excerpt(p));
      return body;
    }

    if (target) {
      body.append(header(page, ''));
      const p = target.matches('p') ? target : target.querySelector('p');
      if (p) body.append(excerpt(p));
      return body;
    }

    const content = doc.querySelector('.sl-markdown-content');
    const img = content?.querySelector('p > img');
    if (img) body.append(thumb(absolute(img.getAttribute('src'), url), img.alt));
    body.append(header('', page));
    const description = doc.querySelector('meta[name="description"]')?.content;
    if (description) body.append(el('p', 'cm-lp-text', description));
    return body;
  }

  async function externalCard(url) {
    const data = (await getExternal())[url.href];
    const body = el('div', 'cm-lp-body cm-lp-external');
    if (!data) {
      body.append(header(url.hostname.replace(/^www\./, ''), decodeURIComponent(url.pathname + url.search)));
      return body;
    }
    if (data.image) body.append(thumb(data.image, ''));
    const kicker = [data.site, data.kind, data.date].filter(Boolean).join(' · ');
    body.append(header(kicker, data.title || url.hostname));
    if (data.authors) body.append(el('div', 'cm-lp-authors', data.authors));
    if (data.description) body.append(el('p', 'cm-lp-text', data.description));
    return body;
  }

  function place(a) {
    const r = a.getBoundingClientRect();
    const gap = 8;
    const w = card.offsetWidth;
    const h = card.offsetHeight;
    const left = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
    const below = r.bottom + gap + h <= window.innerHeight || r.top - gap - h < 0;
    const top = below ? r.bottom + gap : r.top - gap - h;
    card.style.left = `${left + window.scrollX}px`;
    card.style.top = `${top + window.scrollY}px`;
    card.dataset.side = below ? 'below' : 'above';
  }

  async function show(a) {
    const id = ++request;
    let url;
    try {
      url = new URL(a.getAttribute('href'), location.href);
    } catch {
      return;
    }
    if (!/^https?:$/.test(url.protocol)) return;
    let body;
    try {
      body = url.origin === location.origin ? await internalCard(url) : await externalCard(url);
    } catch {
      return;
    }
    if (id !== request || activeLink !== a) return;
    if (!card) {
      card = el('div', 'cm-link-preview');
      card.setAttribute('role', 'tooltip');
      card.id = 'cm-link-preview';
      card.addEventListener('mouseenter', () => clearTimeout(hideTimer));
      card.addEventListener('mouseleave', scheduleHide);
      document.body.append(card);
    }
    card.replaceChildren(body);
    card.hidden = false;
    card.classList.remove('is-open');
    place(a);
    a.setAttribute('aria-describedby', card.id);
    void card.offsetWidth; // start the fade from the closed state
    card.classList.add('is-open');
    // Re-place once the thumbnail knows its height.
    card.querySelector('img')?.addEventListener('load', () => activeLink === a && place(a), { once: true });
  }

  function hide() {
    request++;
    activeLink?.removeAttribute('aria-describedby');
    activeLink = null;
    if (card) {
      card.classList.remove('is-open');
      card.hidden = true;
    }
  }

  function scheduleHide() {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, HIDE_DELAY);
  }

  function scheduleShow(a) {
    clearTimeout(hideTimer);
    if (activeLink === a) return;
    clearTimeout(showTimer);
    if (activeLink) hide();
    activeLink = a;
    showTimer = setTimeout(() => show(a), SHOW_DELAY);
  }

  document.addEventListener('mouseover', (e) => {
    const a = previewLink(e.target);
    if (a) scheduleShow(a);
  });
  document.addEventListener('mouseout', (e) => {
    const a = previewLink(e.target);
    if (a && !a.contains(e.relatedTarget)) scheduleHide();
  });
  document.addEventListener('focusin', (e) => {
    const a = previewLink(e.target);
    if (a) scheduleShow(a);
  });
  document.addEventListener('focusout', (e) => {
    if (previewLink(e.target)) scheduleHide();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && activeLink) hide();
  });
  document.addEventListener('click', (e) => {
    if (previewLink(e.target)) hide();
  });
  window.addEventListener('resize', hide);
})();
