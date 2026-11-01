/** Fixed-fixture next-token teaching mechanics. No tokenizer or model runs here. */
import { FIXTURE } from './fixture.mjs';
export { FIXTURE };
export const PHASES = Object.freeze(['tokens', 'embedding', 'layers', 'distribution', 'append']);
export const MAX_GENERATED = 2;
const vocabulary = new Map(FIXTURE.examples.flatMap(example => example.ids.map((id, index) => [id, example.pieces[index]])));
const prompt = FIXTURE.examples[0];

/** Decode only known, individually UTF-8-decodable fixture tokens. */
export function decode(ids) {
  return ids.map(id => {
    if (!vocabulary.has(id)) throw new RangeError(`Token ${id} is outside this fixture`);
    return vocabulary.get(id);
  }).join('');
}

/** Start the bounded demonstration with a reproducible unsigned 32-bit seed. */
export function createState({ seed = 1, mode = 'greedy' } = {}) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new RangeError('Seed must be uint32');
  if (!['greedy', 'sample'].includes(mode)) throw new RangeError('Unknown selection mode');
  return { phase: 'tokens', ids: [...prompt.ids], text: prompt.text, generated: 0,
    mode, seed, initialSeed: seed, stopped: false, stopReason: null, selection: null };
}

/** Return synthetic probabilities, never measured model probabilities. */
export function distributionFor(state) {
  if (state.generated >= MAX_GENERATED || state.stopped) return [];
  if (state.generated === 1) return [{ id: 13, piece: '.', probability: 1, aggregate: false }];
  return [
    { id: 5634, piece: ' mat', probability: 0.45, aggregate: false },
    { id: 6558, piece: ' floor', probability: 0.25, aggregate: false },
    { id: 32169, piece: ' sofa', probability: 0.15, aggregate: false },
    { id: null, piece: null, label: 'All other tokens', probability: 0.15, aggregate: true }
  ];
}

/** Greedy choice or seeded LCG sample; the aggregate bucket has no decodable ID. */
export function chooseToken(distribution, { mode = 'greedy', seed = 1 } = {}) {
  createState({ seed, mode });
  if (!distribution.length || distribution.some(item => !Number.isFinite(item.probability) || item.probability < 0)
      || Math.abs(distribution.reduce((sum, item) => sum + item.probability, 0) - 1) > 1e-9) {
    throw new RangeError('Probabilities must be nonnegative and sum to one');
  }
  if (mode === 'greedy') {
    return { ...distribution.reduce((best, item) => item.probability > best.probability ? item : best), seed };
  }
  const nextSeed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  const draw = nextSeed / 4294967296;
  let cumulative = 0;
  for (const item of distribution) {
    cumulative += item.probability;
    if (draw < cumulative) return { ...item, seed: nextSeed };
  }
  return { ...distribution.at(-1), seed: nextSeed };
}

/** Advance one stage without mutating the previous state. */
export function step(state) {
  if (state.stopped) return { ...state, ids: [...state.ids] };
  const index = PHASES.indexOf(state.phase);
  if (index < 0) throw new RangeError('Unknown phase');
  if (state.phase === 'append') {
    return state.generated >= MAX_GENERATED
      ? { ...state, ids: [...state.ids], stopped: true, stopReason: 'demo-limit' }
      : { ...state, ids: [...state.ids], phase: 'tokens', selection: null };
  }
  const next = { ...state, ids: [...state.ids], phase: PHASES[index + 1] };
  if (next.phase !== 'append') return next;
  const selection = chooseToken(distributionFor(state), state);
  next.selection = selection;
  next.seed = selection.seed;
  if (selection.aggregate) return { ...next, stopped: true, stopReason: 'outside-fixture' };
  next.ids.push(selection.id);
  next.text = decode(next.ids);
  next.generated += 1;
  return next;
}

/** Replay from the same original seed and selection mode. */
export function reset(state) {
  return createState({ mode: state.mode, seed: state.initialSeed });
}

/** Hand-authored teaching lookup; IDs select rows, never determine vector values. */
export const EMBEDDINGS = Object.freeze({
  791: Object.freeze([0.0, 0.7, -0.5, 0.2, 0.9, -0.3]),
  8415: Object.freeze([-0.9, -0.2, 0.5, -0.7, 0.0, 0.7]),
  7731: Object.freeze([0.4, -0.8, -0.1, 0.6, -0.6, 0.1]),
  389: Object.freeze([0.6, -0.6, 0.1, 0.8, -0.4, 0.3]),
  279: Object.freeze([-0.2, 0.5, -0.7, 0.0, 0.7, -0.5]),
  5634: Object.freeze([-0.2, 0.5, -0.7, 0.0, 0.7, -0.5]),
  6558: Object.freeze([0.4, -0.8, -0.1, 0.6, -0.6, 0.1]),
  32169: Object.freeze([-0.4, 0.3, -0.9, -0.2, 0.5, -0.7]),
  13: Object.freeze([0.4, -0.8, -0.1, 0.6, -0.6, 0.1]),
  359: Object.freeze([-0.7, 0.0, 0.7, -0.5, 0.2, 0.9]),
  71: Object.freeze([-0.6, 0.1, 0.8, -0.4, 0.3, -0.9]),
  67391: Object.freeze([-0.6, 0.1, 0.8, -0.4, 0.3, -0.9]),
  519: Object.freeze([0.0, 0.7, -0.5, 0.2, 0.9, -0.3]),
  85342: Object.freeze([-0.7, 0.0, 0.7, -0.5, 0.2, 0.9]),
  34500: Object.freeze([-0.2, 0.5, -0.7, 0.0, 0.7, -0.5]),
  479: Object.freeze([-0.6, 0.1, 0.8, -0.4, 0.3, -0.9]),
  8997: Object.freeze([-0.5, 0.2, 0.9, -0.3, 0.4, -0.8]),
  2191: Object.freeze([0.6, -0.6, 0.1, 0.8, -0.4, 0.3]),
});

/** Shared hand-authored row transform; not learned weights from a real model. */
export const MLP_MATRIX = Object.freeze([
  [0.5, 0, 0, 0, 0, 0.25],
  [0.25, 0.5, 0, 0, 0, 0],
  [0, 0.25, 0.5, 0, 0, 0],
  [0, 0, 0.25, 0.5, 0, 0],
  [0, 0, 0, 0.25, 0.5, 0],
  [0, 0, 0, 0, 0.25, 0.5]
].map(row => Object.freeze(row)));

// Round arithmetic to ten decimal places to remove binary floating-point tails.
// Consumers should show sufficient decimals or explicitly mark rounded values.
const rounded = value => Number(value.toFixed(10));
const add = (a, b) => a.map((value, index) => rounded(value + b[index]));

/** Fixed causal, recent-weighted coefficients; no query/key scores or softmax. */
export function causalWeights(rowIndex, length) {
  if (!Number.isInteger(length) || length < 1 || length > 7
      || !Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= length) {
    throw new RangeError('Expected a row in a one-to-seven-token fixture');
  }
  return Array.from({ length }, (_, source) => source > rowIndex ? 0
    : source === 0 ? 2 ** -rowIndex : 2 ** (source - rowIndex - 1));
}

/** Explain one toy block: gather earlier/self rows, then process this row. */
export function traceFor(state, rowIndex = state.ids.length - 1) {
  const weights = causalWeights(rowIndex, state.ids.length);
  const input = state.ids.map(id => {
    if (!Object.hasOwn(EMBEDDINGS, id)) throw new RangeError(`Token ${id} is outside this fixture`);
    return [...EMBEDDINGS[id]];
  });
  const before = [...input[rowIndex]];
  const contributions = input.map((row, source) => row.map(value => rounded(value * weights[source])));
  const contextMix = Array.from({ length: 6 }, (_, column) =>
    rounded(contributions.reduce((sum, row) => sum + row[column], 0)));
  const attentionDelta = contextMix.map(value => rounded(0.5 * value));
  const afterAttention = add(before, attentionDelta);
  const mlpPreactivation = Array.from({ length: 6 }, (_, column) =>
    rounded(afterAttention.reduce((sum, value, index) => sum + value * MLP_MATRIX[index][column], 0)));
  const mlpActivation = mlpPreactivation.map(value => Math.max(0, value));
  const mlpDelta = mlpActivation.map(value => rounded(0.25 * value));
  const output = add(afterAttention, mlpDelta);
  return { rowIndex, tokenId: state.ids[rowIndex], input, weights, contributions,
    contextMix, before, attentionDelta, afterAttention,
    mlpMatrix: MLP_MATRIX.map(row => [...row]), mlpPreactivation, mlpActivation, mlpDelta, output };
}

/** Six-component toy arrays, with every contextual value explained by traceFor. */
export function matricesFor(state) {
  const traces = state.ids.map((_, rowIndex) => traceFor(state, rowIndex));
  if (!traces.length) throw new RangeError('Expected at least one fixture token');
  return { input: traces[0].input, contextual: traces.map(trace => trace.output),
    attentionWeights: traces.map(trace => trace.weights),
    afterAttention: traces.map(trace => trace.afterAttention) };
}
