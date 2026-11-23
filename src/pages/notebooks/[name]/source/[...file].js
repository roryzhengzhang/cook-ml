import { notebookWorkspacePaths } from '../../../../labs/notebook-python.mjs';

export function getStaticPaths() {
  return notebookWorkspacePaths().flatMap(({ params, props }) =>
    props.files.map(({ path, source }) => ({
      params: { ...params, file: path },
      props: { source },
    })),
  );
}

export function GET({ props }) {
  return new Response(props.source, {
    headers: { 'Content-Type': 'text/x-python; charset=utf-8' },
  });
}
