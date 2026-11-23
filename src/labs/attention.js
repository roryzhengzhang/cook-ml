const lab = document.getElementById('attention-playground');
const controls = lab.querySelector('form');
const sliders = [...controls.querySelectorAll('input[type="range"]')];
const lanes = [...lab.querySelectorAll('[data-lane]')].map((lane) => ({
  lane,
  dot: lane.querySelector('[data-dot]'),
  score: lane.querySelector('[data-score]'),
  weight: lane.querySelector('[data-weight]'),
  track: lane.querySelector('.cm-attention-track'),
}));
const editor = lab.querySelector('textarea');
const editorPanel = editor.closest('details');
const vector = lab.querySelector('[data-vector]');
const consoleOutput = lab.querySelector('[data-console]');
const status = lab.querySelector('[role="status"]');
const stop = lab.querySelector('[data-action="stop"]');
const defaults = {
  dot_products: lanes.map(({ dot }) => Number(dot.textContent.replace('−', '-'))),
  scores: lanes.map(({ score }) => Number(score.textContent.replace('−', '-'))),
  weights: lanes.map(({ track }) => Number(track.style.getPropertyValue('--weight'))),
  output: vector.textContent.slice(1, -1).split(', ').map(Number),
};

let source = editor.defaultValue;
let pythonEditor;
let worker;
let ready = false;
let pending;
let active;
let timer;
let nextId = 0;

function showStatus(message) {
  if (status.textContent !== message) status.textContent = message;
}

function showConsole(message) {
  consoleOutput.textContent = message;
  consoleOutput.hidden = !message;
}

function cancel(message) {
  clearTimeout(timer);
  worker?.terminate();
  worker = undefined;
  ready = false;
  pending = undefined;
  active = undefined;
  stop.disabled = true;
  showStatus(message);
}

function number(value, digits) {
  return value.toFixed(digits).replace('-', '−');
}

function render({ dot_products, scores, weights, output }) {
  lanes.forEach(({ lane, dot, score, weight, track }, index) => {
    const masked = scores[index] == null;
    dot.textContent = number(dot_products[index], 3);
    score.textContent = masked ? '−∞ (masked)' : number(scores[index], 3);
    weight.textContent = number(weights[index] * 100, 1) + '%';
    track.style.setProperty('--weight', weights[index]);
    lane.toggleAttribute('data-masked', masked);
  });
  vector.textContent = '(' + output.map((value) => number(value, 4)).join(', ') + ')';
}

function isVector(value, length, valid = Number.isFinite) {
  return Array.isArray(value) && value.length === length && value.every(valid);
}

function isAttentionResult(value) {
  return value &&
    isVector(value.dot_products, lanes.length) &&
    isVector(value.scores, lanes.length, (score) => score == null || Number.isFinite(score)) &&
    isVector(value.weights, lanes.length, (weight) => Number.isFinite(weight) && weight >= 0 && weight <= 1) &&
    isVector(value.output, 2);
}

function dispatch() {
  if (!ready || active || !pending) return;
  active = pending;
  pending = undefined;
  stop.disabled = false;
  timer = setTimeout(() => cancel('Python timed out. Run again or reset the example.'), 10000);
  worker.postMessage(active);
}

function handleMessage({ data }) {
  if (data.ready) {
    clearTimeout(timer);
    ready = true;
    showStatus('Python ready.');
  } else {
    if (data.id !== active?.id) return;
    clearTimeout(timer);
    active = undefined;
    stop.disabled = true;
    if (!pending) {
      if (data.error) {
        showStatus('Python raised an error. The chart shows the last result.');
        showConsole([data.output, data.error].filter(Boolean).join('\n\n'));
      } else if (isAttentionResult(data.value)) {
        render(data.value);
        showStatus('Python ready.');
        showConsole(data.output);
      } else {
        showStatus('The chart needs dot products, scores, weights, and an output vector.');
        showConsole(data.output);
      }
    }
  }
  dispatch();
}

function queueRun() {
  const fields = controls.elements;
  sliders.forEach((input) => {
    controls.querySelector(`output[for="${input.id}"]`).textContent = number(input.valueAsNumber, 1);
    const minimum = Number(input.min);
    const position = (input.valueAsNumber - minimum) / (Number(input.max) - minimum);
    input.style.setProperty('--position', position * 100 + '%');
  });
  pending = {
    id: ++nextId,
    code: source,
    inputs: {
      query_index: Number(fields.query_index.value),
      key_first: fields.key_first.valueAsNumber,
      value_first: fields.value_first.valueAsNumber,
      causal: fields.causal.checked,
      scaled: fields.scaled.checked,
    },
  };

  if (!worker) {
    try {
      worker = new Worker('/labs/python-worker.js', { type: 'module' });
      worker.onmessage = handleMessage;
      worker.onerror = () => cancel('Python could not start. Run again to retry.');
      showStatus('Loading Python…');
      stop.disabled = false;
      timer = setTimeout(() => cancel('Python took too long to load. Run again to retry.'), 45000);
    } catch {
      cancel('Python could not start in this browser.');
    }
  }
  dispatch();
}

controls.addEventListener('submit', (event) => event.preventDefault());
controls.addEventListener('input', queueRun);
const copyFeedback = lab.querySelector('[data-copy-feedback]');
let copyFeedbackTimer;
lab.querySelector('[data-action="copy"]').addEventListener('click', async () => {
  clearTimeout(copyFeedbackTimer);
  try {
    await navigator.clipboard.writeText(pythonEditor?.value ?? editor.value);
    copyFeedback.textContent = 'Copied';
  } catch {
    copyFeedback.textContent = 'Unable to copy';
  }
  copyFeedbackTimer = setTimeout(() => { copyFeedback.textContent = ''; }, 1800);
});
lab.querySelector('[data-action="run"]').addEventListener('click', () => {
  source = pythonEditor?.value ?? editor.value;
  showConsole('');
  if (ready) showStatus('Running Python…');
  queueRun();
});
stop.addEventListener('click', () => cancel('Python stopped. Change a control or run again.'));
lab.querySelector('[data-action="reset"]').addEventListener('click', () => {
  if (active) cancel('Resetting Python…');
  controls.reset();
  editor.value = editor.defaultValue;
  pythonEditor?.reset();
  source = editor.defaultValue;
  showConsole('');
  render(defaults);
  queueRun();
});
lab.addEventListener('toggle', () => {
  if (lab.open && !worker) queueRun();
});
if (location.hash === '#attention-playground') lab.open = true;
if (lab.open) queueRun();

async function enhanceEditor() {
  try {
    const { mountPythonEditor } = await import('./python-editor.js');
    pythonEditor = mountPythonEditor(editor);
  } catch (error) {
    // The original textarea remains editable if the editor bundle fails to load.
    console.warn('Could not load Python syntax highlighting.', error);
  }
}

if (editorPanel.open) void enhanceEditor();
else editorPanel.addEventListener('toggle', () => void enhanceEditor(), { once: true });
