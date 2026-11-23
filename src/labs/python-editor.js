import { EditorView, minimalSetup } from 'codemirror';
import { python } from '@codemirror/lang-python';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { tags } from '@lezer/highlight';

// Use the Night Owl token colors used by the dark Starlight codepads.
const nightOwl = HighlightStyle.define([
  { tag: [tags.keyword, tags.operatorKeyword], color: '#c792ea' },
  { tag: [tags.string, tags.special(tags.string)], color: '#ecc48d' },
  { tag: [tags.number, tags.bool, tags.null], color: '#f78c6c' },
  { tag: [tags.function(tags.variableName), tags.definition(tags.className)], color: '#82aaff' },
  { tag: tags.comment, color: '#969baa' },
]);

const editorTheme = EditorView.theme({
  '.cm-content': { padding: '0.75rem 0', caretColor: '#f5f0fb' },
  '.cm-line': { padding: '0 0.75rem' },
  '.cm-scroller': { fontFamily: 'inherit', lineHeight: 'inherit', overflow: 'auto' },
  '&.cm-focused': { outline: '2px solid var(--sl-color-text-accent)', outlineOffset: '2px' },
}, { dark: true });

export function mountPythonEditor(textarea) {
  const view = new EditorView({
    doc: textarea.value,
    extensions: [
      minimalSetup,
      python(),
      syntaxHighlighting(nightOwl),
      editorTheme,
      EditorView.contentAttributes.of({
        'aria-label': textarea.getAttribute('aria-label'),
        'aria-describedby': textarea.getAttribute('aria-describedby'),
        autocapitalize: 'off',
        spellcheck: 'false',
      }),
    ],
  });
  textarea.before(view.dom);
  if (textarea === document.activeElement) view.focus();
  textarea.hidden = true;

  return {
    get value() {
      return view.state.doc.toString();
    },
    reset() {
      const original = textarea.defaultValue;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: original } });
    },
  };
}
