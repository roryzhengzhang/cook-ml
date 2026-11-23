import { publicDir } from 'astro:config/server';
import { fromMarkdown } from 'mdast-util-from-markdown';
import { loadNotebook, notebookProjects } from '../../scripts/remark-story-notebook.mjs';

function plainText(node) {
  switch (node.type) {
    case 'root':
    case 'listItem':
      return node.children.map(plainText).join('\n\n');
    case 'list':
      return node.children.map((item, index) => {
        const bullet = node.ordered ? `${node.start + index}. ` : '• ';
        const indent = ' '.repeat(bullet.length);
        return bullet + plainText(item).replaceAll('\n', `\n${indent}`);
      }).join('\n');
    case 'link': {
      const label = node.children.map(plainText).join('');
      return label === node.url ? label : `${label} (${node.url})`;
    }
    case 'break':
      return '\n';
    default:
      return node.value ?? node.children?.map(plainText).join('') ?? '';
  }
}

function renderCell(cell) {
  if (cell.cell_type === 'code') return cell.source.split('\n');
  const prose = cell.cell_type === 'markdown' ? plainText(fromMarkdown(cell.source)) : cell.source;
  return prose.trimEnd().split('\n').map((line) => line.trimEnd() ? `# ${line.trimEnd()}` : '#');
}

function renderPython(blocks) {
  const lines = [];
  const anchors = {};
  for (const { cell, content } of blocks) {
    if (cell.cell_type === 'code') anchors[lines.length + 1] = cell.id;
    lines.push(...content, '');
  }
  return { source: lines.join('\n'), anchors };
}

function notebookBlocks(project) {
  const file = new URL(`notebooks/${project.name}.ipynb`, publicDir);
  return loadNotebook({ file }).map((cell) => ({ cell, content: renderCell(cell) }));
}

export function notebookPythonPaths() {
  return notebookProjects.map((project) => ({
    params: { name: project.name },
    props: renderPython(notebookBlocks(project)),
  }));
}

export function notebookWorkspacePaths() {
  return notebookProjects.map((project) => {
    const blocks = notebookBlocks(project);
    const paths = new Set();
    const sourceFiles = [...project.stages, ...(project.extraSourceFiles ?? [])];
    const stages = sourceFiles.map(([id, title, path]) => {
      let start = blocks.findIndex(({ cell }) => cell.cell_type === 'code' && cell.id === id);
      if (start < 0 || !/^[a-z_]+\/[a-z_]+\.py$/.test(path) || paths.has(path)) {
        throw new Error(`Invalid source file for ${project.name} stage ${id}: ${path}`);
      }
      paths.add(path);
      // Keep the explanation immediately before a stage with its first code cell.
      while (start > 0 && blocks[start - 1].cell.cell_type !== 'code') start -= 1;
      return { id: `file-${id}`, title, path, start };
    });
    if (stages[0]?.start !== 0 || stages.some(({ start }, i) => i > 0 && start <= stages[i - 1].start)) {
      throw new Error(`Source stages for ${project.name} must follow notebook order and start at the first cell`);
    }
    const files = stages.map(({ start, ...file }, i) => ({
      ...file, ...renderPython(blocks.slice(start, stages[i + 1]?.start)),
    }));
    const readme = project.readme ? new URL(project.readme, publicDir) : undefined;
    return { params: { name: project.name }, props: { files, readme } };
  });
}
