const workspace = document.querySelector('.workspace');
const explorer = workspace.querySelector('.source-explorer');
const editor = workspace.querySelector('.source-editor');
const toggle = workspace.querySelector('[data-explorer-toggle]');
const tab = workspace.querySelector('[data-active-file]');
const fileName = tab.querySelector('[data-file-name]');
const fileIcon = tab.querySelector('[data-file-icon]');
const download = workspace.querySelector('[data-file-download]');
const files = [...editor.querySelectorAll('[data-file-path]')];
const fileLinks = new Map([...explorer.querySelectorAll('[data-source-file]')]
  .map((link) => [link.hash.slice(1), link]));
const narrow = matchMedia('(width < 34rem)');
let activeFile;
let highlight;

function showExplorer(open) {
  if (!open && explorer.contains(document.activeElement)) toggle.focus({ preventScroll: true });
  explorer.hidden = !open;
  toggle.setAttribute('aria-expanded', String(open));
  workspace.toggleAttribute('data-explorer-closed', !open);
}

function scrollToSource() {
  const top = highlight
    ? editor.scrollTop + highlight.getBoundingClientRect().top - editor.getBoundingClientRect().top - 16
    : 0;
  editor.scrollTo({ top, behavior: 'instant' });
}

function showLocation() {
  const target = document.getElementById(location.hash.slice(1));
  const file = target?.closest('[data-file-path]') ?? files[0];
  const link = fileLinks.get(file.id);

  if (file !== activeFile) {
    for (const section of files) section.hidden = section !== file;
    fileLinks.get(activeFile?.id)?.removeAttribute('aria-current');
    link.setAttribute('aria-current', 'location');
    const path = file.dataset.filePath;
    fileName.textContent = path.split('/').at(-1);
    fileIcon.textContent = path.endsWith('.md') ? 'MD' : 'Py';
    tab.title = path;
    download.href = file.dataset.fileUrl;
    activeFile = file;
  }

  const folder = link.closest('details');
  if (folder) folder.open = true;
  highlight?.classList.remove('is-jump-target');
  highlight = target?.matches('.line') ? target : undefined;
  highlight?.classList.add('is-jump-target');
  scrollToSource();
}

toggle.hidden = false;
tab.hidden = false;
download.hidden = false;
workspace.setAttribute('data-enhanced', '');
showExplorer(!narrow.matches);
showLocation();
// Font metrics may change wrapping, including lines above the initial code link.
const initialHash = location.hash;
void document.fonts.ready.then(() => {
  if (location.hash === initialHash && highlight) scrollToSource();
});

toggle.addEventListener('click', () => showExplorer(explorer.hidden));
narrow.addEventListener('change', () => showExplorer(!narrow.matches));
window.addEventListener('hashchange', showLocation);
document.addEventListener('cookml:source-jump', showLocation);
explorer.addEventListener('click', (event) => {
  const link = event.target.closest('[data-source-file]');
  if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

  event.preventDefault();
  if (location.hash !== link.hash) {
    if (window.parent === window) history.pushState(null, '', link.hash);
    else history.replaceState(null, '', link.hash);
  }
  showLocation();
  if (narrow.matches) {
    editor.focus({ preventScroll: true });
    showExplorer(false);
  }
});
