/**
 * Runs the forecast analyst (ml/src/auraguard/agent) for one question.
 *
 * The analyst is Python: an LLM restricted to typed forecast tools, whose answer is
 * checked deterministically before it is returned. This spawns
 * `python -m auraguard.agent --json <question>` and returns its JSON record.
 * Set AURAGUARD_PYTHON to the interpreter that has the ml package installed.
 */

const { spawn } = require('child_process');
const path = require('path');

const ML_DIR = path.join(__dirname, '../../../../ml');
const DEFAULT_PYTHON = process.platform === 'win32' ? 'python' : 'python3';

function askAnalyst(question, { timeoutMs = 120000, python = process.env.AURAGUARD_PYTHON || DEFAULT_PYTHON } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, ['-m', 'auraguard.agent', '--json', question], {
      cwd: ML_DIR,
      env: { ...process.env, PYTHONPATH: path.join(ML_DIR, 'src'), PYTHONIOENCODING: 'utf-8' },
    });
    let out = '';
    let err = '';
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error('forecast analyst timed out'));
    }, timeoutMs);
    child.stdout.on('data', (d) => { out += d.toString('utf8'); });
    child.stderr.on('data', (d) => { err += d.toString('utf8'); });
    child.on('error', (e) => {
      clearTimeout(timer);
      reject(new Error(`could not start ${python}: ${e.message}`));
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      try {
        const result = JSON.parse(out);
        if (result.error && !result.answer) return reject(new Error(result.error));
        return resolve(result);
      } catch (e) {
        return reject(new Error(`forecast analyst failed (exit ${code}): ${err.slice(0, 300)}`));
      }
    });
  });
}

module.exports = { askAnalyst };
