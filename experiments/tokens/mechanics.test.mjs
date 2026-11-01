import test from 'node:test';
import assert from 'node:assert/strict';
import { FIXTURE, PHASES, createState, step, reset, decode, distributionFor, chooseToken, matricesFor } from './mechanics.mjs';

test('versioned real tokenization, whitespace and subword fixture', () => {
  assert.equal(FIXTURE.version, '0.14.0');
  assert.equal(FIXTURE.encoding, 'cl100k_base');
  assert.deepEqual(FIXTURE.examples[0].ids, [791, 8415, 7731, 389, 279]);
  assert.deepEqual(FIXTURE.examples.find(e => e.text === 'unhappiness').pieces, ['un', 'h', 'appiness']);
  for (const example of FIXTURE.examples) {
    assert.equal(decode(example.ids), example.text);
    assert.equal(example.pieces.join(''), example.text);
  }
  assert.throws(() => decode([999999]), RangeError);
});

test('five stages, two bounded greedy cycles, immutable history and reset', () => {
  const initial = createState();
  let state = initial;
  for (let cycle = 0; cycle < 2; cycle++) {
    for (const phase of PHASES) {
      assert.equal(state.phase, phase);
      const before = structuredClone(state);
      const next = step(state);
      assert.deepEqual(state, before);
      state = next;
    }
  }
  assert.equal(state.text, 'The cat sat on the mat.');
  assert.equal(state.generated, 2);
  assert.equal(state.stopped, true);
  assert.equal(state.stopReason, 'demo-limit');
  assert.deepEqual(step(state), state);
  assert.deepEqual(reset(state), initial);
  assert.equal(initial.ids.length, 5);
});

test('probability fixture and seeded sampling are reproducible', () => {
  const distribution = distributionFor(createState());
  assert.deepEqual(distribution.map(item => item.probability), [.45, .25, .15, .15]);
  const counts = new Set();
  for (let seed = 0; seed < 2000; seed++) {
    const a = chooseToken(distribution, { mode: 'sample', seed });
    assert.deepEqual(a, chooseToken(distribution, { mode: 'sample', seed }));
    counts.add(a.id);
  }
  assert.deepEqual(counts, new Set([5634, 6558, 32169, null]));
  assert.equal(chooseToken(distribution).id, 5634);
});

test('sampled aggregate stops honestly without inventing or decoding a token', () => {
  let state = createState({ mode: 'sample', seed: 1700 });
  for (let i = 0; i < 4; i++) state = step(state);
  assert.equal(state.selection.aggregate, true);
  assert.equal(state.stopReason, 'outside-fixture');
  assert.equal(state.generated, 0);
  assert.equal(state.text, FIXTURE.examples[0].text);
  assert.deepEqual(state.ids, FIXTURE.examples[0].ids);
  assert.deepEqual(distributionFor(state), []);
  let replay = reset(state);
  for (let i = 0; i < 4; i++) replay = step(replay);
  assert.deepEqual(replay, state);
});

test('display matrices are deterministic and aligned, not model measurements', () => {
  const state = createState();
  const matrices = matricesFor(state);
  assert.deepEqual(matricesFor(state), matrices);
  assert.equal(matrices.input.length, state.ids.length);
  assert.ok(matrices.input.every(row => row.length === 6));
  assert.ok(matrices.contextual.every(row => row.length === 6));
  assert.notDeepEqual(matrices.input, matrices.contextual);
  const repeated = matricesFor({ ids: [8415, 8415] });
  assert.deepEqual(repeated.input[0], repeated.input[1]);
  assert.notDeepEqual(repeated.contextual[0], repeated.contextual[1]);
});

test('invalid controls and probability arrays reject', () => {
  for (const seed of [-1, 1.5, NaN, 4294967296]) assert.throws(() => createState({ seed }), RangeError);
  assert.throws(() => createState({ mode: 'unknown' }), RangeError);
  for (const probabilities of [[], [0.2], [-1, 2], [NaN], [Infinity]]) {
    assert.throws(() => chooseToken(probabilities.map(probability => ({ probability }))), RangeError);
  }
});
