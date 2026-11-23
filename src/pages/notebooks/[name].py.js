import { notebookPythonPaths } from '../../labs/notebook-python.mjs';

export const getStaticPaths = notebookPythonPaths;

export function GET({ props }) {
  return new Response(props.source, {
    headers: { 'Content-Type': 'text/x-python; charset=utf-8' },
  });
}
