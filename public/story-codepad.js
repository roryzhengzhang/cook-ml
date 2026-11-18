// Open a linked codepad even when it sits inside a closed tutorial disclosure.
function revealCodepad() {
  const hash = window.location.hash.slice(1);
  if (!/^[a-z0-9-]+$/.test(hash)) return;
  const pad = document.getElementById(hash);
  if (!pad?.matches('details.cm-codepad')) return;
  for (let node = pad; node; node = node.parentElement?.closest('details')) node.open = true;
  requestAnimationFrame(() => pad.scrollIntoView({ block: 'start' }));
}

revealCodepad();
window.addEventListener('hashchange', revealCodepad);
