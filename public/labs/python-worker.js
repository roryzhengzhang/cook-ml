import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.8/full/pyodide.mjs';

const pyodide = await loadPyodide();
const dict = pyodide.globals.get('dict');
const outputLimit = 8000;

self.onmessage = async ({ data: { id, code, inputs } }) => {
  const globals = dict();
  const parameters = pyodide.toPy(inputs);
  let result;
  let output = '';
  const capture = (line) => {
    if (output.length > outputLimit) return;
    output += (line + '\n').slice(0, outputLimit + 1 - output.length);
  };

  pyodide.setStdout({ batched: capture });
  pyodide.setStderr({ batched: capture });
  try {
    globals.set('inputs', parameters);
    result = await pyodide.runPythonAsync(code, { globals });
    const value = result?.toJs
      ? result.toJs({ dict_converter: Object.fromEntries, create_pyproxies: false })
      : result;
    self.postMessage({ id, value, output: formatOutput(output) });
  } catch (error) {
    self.postMessage({ id, error: String(error).slice(0, outputLimit), output: formatOutput(output) });
  } finally {
    result?.destroy?.();
    parameters.destroy();
    globals.destroy();
  }
};

function formatOutput(output) {
  return output.length > outputLimit
    ? output.slice(0, outputLimit) + '\n[Output truncated]'
    : output.trimEnd();
}

self.postMessage({ ready: true });
