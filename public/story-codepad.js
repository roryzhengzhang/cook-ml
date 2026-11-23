// Open a linked codepad even when it sits inside a closed tutorial disclosure.
function revealLinkedCode() {
  const hash = window.location.hash.slice(1);
  if (!/^[a-z0-9-]+$/.test(hash)) return;
  const pad = document.getElementById(hash);
  const source = pad?.matches('.cm-nb-cell') && pad.querySelector('.cm-nb-source');
  if (source) {
    source.open = true;
    requestAnimationFrame(() => pad.scrollIntoView({ block: 'start', behavior: 'instant' }));
    return;
  }
  if (!pad?.matches('details.cm-codepad')) return;
  for (let node = pad; node; node = node.parentElement?.closest('details')) node.open = true;
  requestAnimationFrame(() => pad.scrollIntoView({ block: 'start' }));
}

revealLinkedCode();
window.addEventListener('hashchange', revealLinkedCode);

const notebook = document.querySelector('.cm-nb');
if (notebook) setupNotebookControls(notebook);

function setupNotebookControls(notebook) {
  const toolbar = notebook.querySelector('.cm-nb-toolbar');
  if (!toolbar) return;
  const outline = toolbar.querySelector('.cm-nb-outline');
  const outlineButton = outline.querySelector('summary');
  const outlineLabel = outline.querySelector('.cm-nb-outline-label');
  const outlineLinks = [...outline.querySelectorAll('nav a')];
  const stages = [...notebook.querySelectorAll('.cm-nb-stage')];
  const toggle = toolbar.querySelector('.cm-nb-toggle-all');
  const sources = [...notebook.querySelectorAll('.cm-nb-source')];
  const cells = [...notebook.querySelectorAll('.cm-nb-cell')];
  const allOpen = () => sources.every((source) => source.open);
  let activeStage;
  let stageUpdate = 0;

  function updateStage() {
    stageUpdate = 0;
    const edge = toolbar.getBoundingClientRect().bottom + 32;
    const stage = stages.findLast((heading) => heading.getBoundingClientRect().top <= edge);
    if (stage === activeStage) return;
    activeStage = stage;
    outlineLabel.textContent = stage?.textContent ?? 'Jump to stage';
    if (stage) outlineButton.setAttribute('aria-label', `Jump to stage. Current stage ${stage.textContent}.`);
    else outlineButton.removeAttribute('aria-label');
    for (const link of outlineLinks) {
      if (link.hash.slice(1) === stage?.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }

  function scheduleStageUpdate() {
    if (!stageUpdate) stageUpdate = requestAnimationFrame(updateStage);
  }

  window.addEventListener('scroll', scheduleStageUpdate, { passive: true });
  window.addEventListener('resize', scheduleStageUpdate);
  scheduleStageUpdate();

  outline.addEventListener('click', (event) => {
    const link = event.target.closest('a[href^="#"]');
    if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    outline.open = false;
    const heading = document.getElementById(link.hash.slice(1));
    if (heading) {
      requestAnimationFrame(() => {
        heading.scrollIntoView({ block: 'start', behavior: 'instant' });
        heading.focus({ preventScroll: true });
      });
    }
  });
  outline.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !outline.open) return;
    event.preventDefault();
    outline.open = false;
    outlineButton.focus({ preventScroll: true });
  });
  document.addEventListener('click', (event) => {
    if (!outline.contains(event.target)) outline.open = false;
  });

  toggle.hidden = false;
  toggle.addEventListener('click', () => {
    const toolbarBottom = toolbar.getBoundingClientRect().bottom;
    const anchor = cells.find((cell) => cell.getBoundingClientRect().bottom >= toolbarBottom);
    const top = anchor?.getBoundingClientRect().top;
    const open = !allOpen();
    for (const source of sources) source.open = open;
    if (anchor) window.scrollBy({ top: anchor.getBoundingClientRect().top - top, behavior: 'instant' });
  });
  notebook.addEventListener('toggle', (event) => {
    if (!event.target.matches('.cm-nb-source')) return;
    toggle.textContent = allOpen() ? 'Hide all code' : 'Show all code';
    scheduleStageUpdate();
  }, true);
}
