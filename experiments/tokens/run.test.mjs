import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createState, distributionFor, chooseToken } from './mechanics.mjs';

const run = (...args) => execFileSync(process.execPath, [fileURLToPath(new URL('./run.mjs', import.meta.url)), ...args], { encoding: 'utf8' });

test('--seed reproduces the sampler draw shown on the companion page', () => {
  const pick = chooseToken(distributionFor(createState()), { mode: 'sample', seed: 600 });
  const out = run('--sample', '--seed', '600');
  assert.match(out, new RegExp(`# draw ${(pick.seed / 4294967296).toFixed(3)} -> "${pick.piece}"`));
  assert.ok((out.match(/^# draw /gm) || []).length >= 1, 'one line per sampled choice');
});

test('default output still starts from seed 42', () => {
  assert.match(run('--sample'), /"seed":42/);
});

test('--draws chains seeds and counts every draw', () => {
  const out = run('--seed', '1', '--draws', '100');
  const counts = [...out.matchAll(/^# .+?\s+(\d+)\s+expected/gm)].map(m => Number(m[1]));
  assert.equal(counts.length, 6);
  assert.equal(counts.reduce((a, b) => a + b, 0), 100);
  assert.match(out, /^# seed 1 draws 0\.236 -> " mat"/m);
});

test('invalid flags fail loudly', () => {
  assert.throws(() => run('--seed', '-1', '--draws', '10'));
  assert.throws(() => run('--draws', '0'));
});

test('--subject and --verb change the computed outcome; --trace shows the legs', () => {
  assert.match(run('--subject', 'king'), /"text":"The king sat on the throne\."/);
  assert.match(run('--subject', 'king', '--verb', 'slept'), /"text":"The king slept on the bed\."/);
  const trace = run('--trace');
  assert.match(trace, /^# heard " cat"/m);
  assert.match(trace, /^# worked out from cat \+ sitting \+ sleeping/m);
  assert.throws(() => run('--subject', 'dog'));
});
