// Keep an inline formula on the same line as the punctuation that touches it.
// Inline math is drawn as a chip (an inline block), so a browser may break the line between a
// formula and the "." or "," right after it, leaving the punctuation alone on the next line.
// This wraps the formula together with closing punctuation after it, and an opening bracket or
// quote just before it, in a span that custom.css marks as unbreakable.
const CLOSE = /^[.,;:!?)\]}%”’"']+/;
const OPEN = /[(\[{“‘"']$/;

const classes = (node) => node.properties?.className ?? [];
const isInlineMath = (node) =>
  node.type === 'element' && classes(node).includes('katex') && !classes(node).includes('katex-display');

function walk(node) {
  if (!node.children) return;
  for (let i = 0; i < node.children.length; i++) {
    const child = node.children[i];
    if (!isInlineMath(child)) {
      walk(child);
      continue;
    }
    const prev = node.children[i - 1];
    const next = node.children[i + 1];
    const after = next?.type === 'text' ? next.value.match(CLOSE)?.[0] : undefined;
    const before = prev?.type === 'text' ? prev.value.match(OPEN)?.[0] : undefined;
    if (!after && !before) continue;
    if (after) next.value = next.value.slice(after.length);
    if (before) prev.value = prev.value.slice(0, -before.length);
    node.children[i] = {
      type: 'element',
      tagName: 'span',
      properties: { className: ['cm-math-nowrap'] },
      children: [
        ...(before ? [{ type: 'text', value: before }] : []),
        child,
        ...(after ? [{ type: 'text', value: after }] : []),
      ],
    };
  }
}

export default function rehypeMathNowrap() {
  return (tree) => walk(tree);
}
