import test from 'node:test';
import assert from 'node:assert/strict';
import { FIXTURE, PHASES, createState, step, reset, decode, distributionFor, chooseToken, matricesFor, traceFor, causalWeights, EMBEDDINGS, MLP_MATRIX } from './mechanics.mjs';

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
  assert.deepEqual(repeated.contextual[0], repeated.contextual[1]);
});

test('invalid controls and probability arrays reject', () => {
  for (const seed of [-1, 1.5, NaN, 4294967296]) assert.throws(() => createState({ seed }), RangeError);
  assert.throws(() => createState({ mode: 'unknown' }), RangeError);
  for (const probabilities of [[], [0.2], [-1, 2], [NaN], [Infinity]]) {
    assert.throws(() => chooseToken(probabilities.map(probability => ({ probability }))), RangeError);
  }
});

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 1e-10, `${actual} != ${expected}`);

test('explicit lookup keeps the original six components; numeric IDs only select rows', () => {
  assert.deepEqual(EMBEDDINGS[8415], [-0.9, -0.2, 0.5, -0.7, 0, 0.7]);
  assert.deepEqual(traceFor({ ids: [8415, 8415] }).input[0], EMBEDDINGS[8415]);
  assert.throws(() => traceFor({ ids: [999999] }), RangeError);
  assert.throws(() => { EMBEDDINGS[8415][0] = 5; }, TypeError);
});

test('causal coefficients normalize, include self, and exclude all future rows', () => {
  for (let length = 1; length <= 7; length++) {
    for (let row = 0; row < length; row++) {
      const weights = causalWeights(row, length);
      assert.equal(weights.reduce((a, b) => a + b), 1);
      assert.ok(weights[row] > 0);
      assert.ok(weights.slice(row + 1).every(weight => weight === 0));
    }
  }
  assert.deepEqual(causalWeights(4, 5), [.0625, .0625, .125, .25, .5]);
  for (const args of [[-1, 5], [5, 5], [0, 0], [0, 8], [1.5, 5]]) {
    assert.throws(() => causalWeights(...args), RangeError);
  }
});

test('all contributions, shared row transform and both residual equations reconcile', () => {
  const state = createState();
  const matrices = matricesFor(state);
  for (let row = 0; row < state.ids.length; row++) {
    const trace = traceFor(state, row);
    for (let column = 0; column < 6; column++) {
      trace.contributions.forEach((values, source) => close(values[column], trace.input[source][column] * trace.weights[source]));
      close(trace.contextMix[column], trace.contributions.reduce((sum, values) => sum + values[column], 0));
      close(trace.attentionDelta[column], trace.contextMix[column] * .5);
      close(trace.afterAttention[column], trace.before[column] + trace.attentionDelta[column]);
      close(trace.mlpPreactivation[column], trace.afterAttention.reduce((sum, value, index) => sum + value * MLP_MATRIX[index][column], 0));
      close(trace.mlpDelta[column], .25 * Math.max(0, trace.mlpPreactivation[column]));
      close(trace.output[column], trace.afterAttention[column] + trace.mlpDelta[column]);
    }
    assert.deepEqual(trace.output, matrices.contextual[row]);
    assert.deepEqual(trace.afterAttention, matrices.afterAttention[row]);
    assert.deepEqual(trace.mlpMatrix, MLP_MATRIX);
  }
  const final = traceFor(state);
  assert.deepEqual(final.attentionDelta, [.021875, .015625, -.16875, .121875, .115625, -.06875]);
  assert.deepEqual(final.output, [-.1681640625, .52578125, -.86875, .1880859375, .88203125, -.56875]);
});

test('earlier context affects a row; future changes and appends cannot leak backward', () => {
  const state = createState();
  const changedPast = { ...state, ids: [5634, ...state.ids.slice(1)] };
  assert.notDeepEqual(traceFor(state).output, traceFor(changedPast).output);
  for (let row = 0; row < 4; row++) {
    const changedFuture = { ...state, ids: [...state.ids.slice(0, row + 1), ...state.ids.slice(row + 1).map(() => 5634)] };
    assert.deepEqual(traceFor(state, row).output, traceFor(changedFuture, row).output);
    assert.deepEqual(traceFor(state, row).output, traceFor({ ids: [...state.ids, 5634, 13] }, row).output);
  }
  const before = structuredClone(state);
  assert.deepEqual(traceFor(state), traceFor(state));
  assert.deepEqual(state, before);
  // The scripted probabilities are independent of the toy representations.
  assert.deepEqual(distributionFor(state), distributionFor(changedPast));
});
