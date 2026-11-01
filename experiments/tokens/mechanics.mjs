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

/** Six-component display arrays: deterministic invented numbers, not embeddings. */
export function matricesFor(state) {
  const input = state.ids.map((id) => Array.from({ length: 6 }, (_, column) =>
    (((id % 17) + column * 7) % 19 - 9) / 10));
  const contextual = input.map((row, index) => row.map((value, column) =>
    Math.round((value * 0.5 + (index + column) / 20) * 100) / 100));
  return { input, contextual };
}
