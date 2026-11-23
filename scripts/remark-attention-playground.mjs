import { readFileSync } from 'node:fs';
import { loadNotebook, notebookExpandLink, storyNotebook } from './remark-story-notebook.mjs';

const template = new URL('../src/labs/attention.html', import.meta.url);
export const attentionFiles = [template, new URL(import.meta.url)];
export const attentionMarker = '<!-- attention-playground -->';

export default function remarkAttentionPlayground() {
  return (tree) => {
    const marker = tree.children.find(
      (node) => node.type === 'html' && node.value.trim() === attentionMarker,
    );
    if (!marker) return;

    const cellId = 'attention-playground';
    const cell = loadNotebook(storyNotebook).find((item) => item.id === cellId && item.cell_type === 'code');
    const exampleCall = /\nattention\([^\n]+\)$/;
    const notebookCode = cell?.source.trimEnd();
    if (!notebookCode || !exampleCall.test(notebookCode)) {
      throw new Error(`The ${cellId} notebook cell must end with a call to attention().`);
    }
    const source = notebookCode.replace(exampleCall, '\nattention(**inputs)')
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;');
    marker.value = readFileSync(template, 'utf8')
      .replace('<!-- python-source -->', () => source)
      .replace('<!-- notebook-expand -->', () => notebookExpandLink(storyNotebook, cellId));
  };
}
