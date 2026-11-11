/** Fixed-fixture next-token teaching mechanics (v2). No tokenizer or model runs here.
 *  One hand-authored card of notes is carried from text to next-token scores, and every
 *  number is computed from the step before it. See README.md for what is simplified. */
import { FIXTURE } from './fixture.mjs';
export { FIXTURE };
export const PHASES = Object.freeze(['tokens', 'embedding', 'layers', 'distribution', 'append']);
export const MAX_GENERATED = 2;

// Round arithmetic to ten decimal places to remove binary floating-point tails.
const rounded = value => { const result = Number(value.toFixed(10)); return result === 0 ? 0 : result; };
const add = (a, b) => a.map((value, index) => rounded(value + b[index]));
const scale = (a, k) => a.map(value => rounded(value * k));
const dot = (a, b) => rounded(a.reduce((sum, value, index) => sum + value * b[index], 0));
const freeze = object => Object.freeze(object);

// ---- Real tokenization: pieces and IDs come only from the tiktoken fixture ----
const vocabulary = new Map(FIXTURE.examples.flatMap(example => example.ids.map((id, index) => [id, example.pieces[index]])));
const idByPiece = new Map([...vocabulary].map(([id, piece]) => [piece, id]));
/** Fixture ID for a piece, or null while the fixture has not been regenerated with it. */
export const idFor = piece => idByPiece.has(piece) ? idByPiece.get(piece) : null;

/** Decode only known, individually UTF-8-decodable fixture tokens. */
export function decode(ids) {
  return ids.map(id => {
    if (!vocabulary.has(id)) throw new RangeError(`Token ${id} is outside this fixture`);
    return vocabulary.get(id);
  }).join('');
}

// ---- Hand-authored teaching model ----
export const WIDTH = 6;
/** Eight named notes. A real model's directions are unlabelled; these labels are ours. */
export const NOTES = freeze([
  freeze({ key: 'thing', label: 'a thing comes next' }),
  freeze({ key: 'surface', label: 'on top of something' }),
  freeze({ key: 'sitting', label: 'sitting' }),
  freeze({ key: 'sleeping', label: 'sleeping' }),
  freeze({ key: 'cat', label: 'a cat' }),
  freeze({ key: 'royal', label: 'royalty' }),
  freeze({ key: 'cosy', label: 'low, cosy spot', blendOf: freeze({ cat: 0.7, sitting: 0.5, sleeping: 0.5, surface: 0.1 }) }),
  freeze({ key: 'grand', label: 'grand seat', blendOf: freeze({ royal: 0.8, sitting: 0.6 }) })
]);
const PRIMARY = NOTES.filter(note => !note.blendOf).map(note => note.key);

// Six primary notes are six orthogonal directions that line up with no single stored number:
// the rows of the order-6 conference matrix, scaled to length one. The two blended notes
// point between their ingredients, so eight notes share six numbers and overlap.
const CONFERENCE = [
  [0, 1, 1, 1, 1, 1], [1, 0, 1, -1, -1, 1], [1, 1, 0, 1, -1, -1],
  [1, -1, 1, 0, 1, -1], [1, -1, -1, 1, 0, 1], [1, 1, -1, -1, 1, 0]
];
const unit = 1 / Math.sqrt(5);
const directions = {};
PRIMARY.forEach((key, index) => { directions[key] = freeze(CONFERENCE[index].map(value => rounded(value * unit))); });
const combine = amounts => Object.entries(amounts).reduce(
  (sum, [key, amount]) => add(sum, scale(directions[key], amount)), Array(WIDTH).fill(0));
NOTES.filter(note => note.blendOf).forEach(note => { directions[note.key] = freeze(combine(note.blendOf)); });
/** Each note's direction in the six stored numbers. */
export const DIRECTIONS = freeze(directions);

/** How strongly every note reads from six stored numbers (a dot product per note). */
export function readNotes(row) {
  return Object.fromEntries(NOTES.map(note => [note.key, dot(row, DIRECTIONS[note.key])]));
}
/** The six stored numbers that hold the given note amounts. */
export function writeNotes(amounts) { return combine(amounts); }

/** What each piece means on its own (`starts`) and how well it answers a listener (`match`). */
export const PIECES = freeze({
  'The': freeze({ starts: freeze({ thing: 1 }), match: -1 }),
  ' the': freeze({ starts: freeze({ thing: 1 }), match: -1 }),
  ' cat': freeze({ starts: freeze({ cat: 1 }), match: 1 }),
  ' king': freeze({ starts: freeze({ royal: 1 }), match: 1 }),
  ' sat': freeze({ starts: freeze({ sitting: 1 }), match: 1 }),
  ' slept': freeze({ starts: freeze({ sleeping: 1 }), match: 1 }),
  ' on': freeze({ starts: freeze({ surface: 1 }), match: 0.8 }),
  ' mat': freeze({ starts: freeze({ cosy: 1 }), match: 1 }),
  ' sofa': freeze({ starts: freeze({ cosy: 0.7, sitting: 0.3 }), match: 1 }),
  ' floor': freeze({ starts: freeze({ surface: 1 }), match: 1 }),
  ' bed': freeze({ starts: freeze({ sleeping: 1 }), match: 1 }),
  ' throne': freeze({ starts: freeze({ grand: 1 }), match: 1 }),
  '.': freeze({ starts: freeze({}), match: -1 })
});
/** A piece's starting row: the six numbers its ID looks up. */
export function embeddingFor(piece) {
  if (!Object.hasOwn(PIECES, piece)) throw new RangeError(`Piece ${JSON.stringify(piece)} is outside this fixture`);
  return writeNotes(PIECES[piece].starts);
}

/** Listening shares are a softmax of match scores; this gain turns a share into an amount. */
export const LISTEN_GAIN = 3;
/** Work-out units: each fires only when its inputs together pass a threshold, then writes one note. */
export const UNITS = freeze([
  freeze({ key: 'cosy', reads: freeze({ cat: 1, sitting: 1, sleeping: 1 }), threshold: 1.2, gain: 1.5, writes: 'cosy' }),
  freeze({ key: 'grand', reads: freeze({ royal: 1, sitting: 1 }), threshold: 1.2, gain: 1.5, writes: 'grand' })
]);
/** How much each candidate's score rises per unit of each note read from the final card. */
export const CANDIDATES = freeze([
  freeze({ piece: ' mat', wants: freeze({ thing: 1, surface: 1, sitting: 0.5, sleeping: 0.2, cat: 0.4, cosy: 1.1 }) }),
  freeze({ piece: ' sofa', wants: freeze({ thing: 1, surface: 1, sitting: 0.6, sleeping: 0.6, cat: 0.2, royal: 0.1, cosy: 0.7, grand: 0.2 }) }),
  freeze({ piece: ' floor', wants: freeze({ thing: 1, surface: 1.3, sitting: 0.3, sleeping: 0.2, cosy: 0.1 }) }),
  freeze({ piece: ' bed', wants: freeze({ thing: 1, surface: 0.8, sleeping: 2, royal: 0.3, cosy: 0.3, grand: 0.2 }) }),
  freeze({ piece: ' throne', wants: freeze({ thing: 1, surface: 0.6, sitting: 0.6, royal: 0.6, grand: 1 }) })
]);
/** The score direction a candidate is compared along (its `wants`, as six numbers). */
export function scoreDirection(candidate) { return writeNotes(candidate.wants); }
/** Fixed stand-in score for the rest of a real vocabulary. An aggregate, never a token. */
export const OTHER_SCORE = 3.4;

/** A fixed three-number view of the six stored numbers, for drawing a path. */
export const VIEW = freeze({
  thing: freeze([0, 0.3, 0]), surface: freeze([0.35, 0.8, 0.35]), sitting: freeze([0.2, 0.8, -0.4]),
  sleeping: freeze([-0.1, 0.55, 0.95]), cat: freeze([1, 0.5, -0.5]), royal: freeze([-0.9, 0.6, 0.4])
});
/** Where six stored numbers sit in the three-number view (linear, so paths stay honest). */
export function viewOf(row) {
  const reads = readNotes(row);
  return [0, 1, 2].map(axis => rounded(PRIMARY.reduce((sum, key) => sum + reads[key] * VIEW[key][axis], 0)));
}

// ---- Prompts ----
export const SUBJECTS = freeze(['cat', 'king']);
export const VERBS = freeze(['sat', 'slept']);
const promptPieces = (subject, verb) => ['The', ` ${subject}`, ` ${verb}`, ' on', ' the'];

/** Start the bounded demonstration with a reproducible unsigned 32-bit seed. */
export function createState({ seed = 1, mode = 'greedy', subject = 'cat', verb = 'sat', covered = [] } = {}) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new RangeError('Seed must be uint32');
  if (!['greedy', 'sample'].includes(mode)) throw new RangeError('Unknown selection mode');
  if (!SUBJECTS.includes(subject) || !VERBS.includes(verb)) throw new RangeError('Unknown prompt variant');
  const pieces = promptPieces(subject, verb);
  if (!Array.isArray(covered) || covered.some(index => !Number.isInteger(index) || index < 0 || index >= pieces.length - 1)) {
    throw new RangeError('Only earlier prompt pieces can be covered');
  }
  return { phase: 'tokens', pieces, ids: pieces.map(idFor), text: pieces.join(''), generated: 0,
    mode, seed, initialSeed: seed, subject, verb, covered: [...new Set(covered)].sort((a, b) => a - b),
    stopped: false, stopReason: null, selection: null };
}

/** Explain one row end to end: start, listen to earlier pieces, work out, final card. */
export function traceFor(state, rowIndex = state.pieces.length - 1) {
  if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= state.pieces.length) throw new RangeError('Unknown row');
  const piece = state.pieces[rowIndex];
  const start = embeddingFor(piece);
  const sources = state.pieces.slice(0, rowIndex).map((source, index) => ({ index, piece: source,
    covered: state.covered.includes(index), match: PIECES[source].match }));
  const total = sources.reduce((sum, source) => sum + Math.exp(source.match), 0);
  const legs = [{ kind: 'start', piece, index: rowIndex, wrote: { ...PIECES[piece].starts }, delta: start, after: start }];
  let row = start;
  // Listening really is one move; it is split into one leg per piece, nearest piece first.
  for (const source of [...sources].reverse()) {
    const share = rounded(Math.exp(source.match) / total);
    const amount = rounded(LISTEN_GAIN * share);
    const offers = source.covered ? {} : PIECES[source.piece].starts;
    const wrote = Object.fromEntries(Object.entries(offers).map(([key, value]) => [key, rounded(value * amount)]));
    const delta = writeNotes(wrote);
    row = add(row, delta);
    legs.push({ kind: 'heard', piece: source.piece, index: source.index, covered: source.covered,
      match: source.match, share, amount, wrote, delta, after: row });
  }
  const afterListening = row;
  const heard = readNotes(afterListening);
  const units = UNITS.map(unit => {
    const input = rounded(Object.entries(unit.reads).reduce((sum, [key, weight]) => sum + weight * heard[key], 0));
    const activation = rounded(Math.max(0, input - unit.threshold));
    return { key: unit.key, reads: { ...unit.reads }, input, threshold: unit.threshold, activation,
      amount: rounded(unit.gain * activation), writes: unit.writes };
  });
  for (const unit of units.filter(unit => unit.amount > 0)) {
    const wrote = { [unit.writes]: unit.amount };
    const delta = writeNotes(wrote);
    row = add(row, delta);
    legs.push({ kind: 'worked', unit: unit.key, from: Object.keys(unit.reads), wrote, delta, after: row });
  }
  let previous = viewOf(Array(WIDTH).fill(0));
  for (const leg of legs) { leg.viewFrom = previous; leg.viewTo = viewOf(leg.after); previous = leg.viewTo; }
  return { rowIndex, piece, id: state.ids[rowIndex], start, legs, afterListening, units, output: row,
    notes: { start: readNotes(start), afterListening: heard, output: readNotes(row) } };
}

/** Every candidate's score from the final card, split by the leg that contributed it. */
export function scoresFor(state) {
  const trace = traceFor(state);
  const candidates = CANDIDATES.map(candidate => {
    const direction = scoreDirection(candidate);
    const parts = trace.legs.map(leg => rounded(dot(leg.delta, direction)));
    return { piece: candidate.piece, id: idFor(candidate.piece), direction, parts, score: dot(trace.output, direction) };
  });
  return { trace, candidates, other: OTHER_SCORE };
}

/** Next-token probabilities. First step: softmax of computed scores. Second step: scripted. */
export function distributionFor(state) {
  if (state.generated >= MAX_GENERATED || state.stopped) return [];
  if (state.generated === 1) return [{ id: idFor('.'), piece: '.', probability: 1, aggregate: false, scripted: true }];
  const { candidates, other } = scoresFor(state);
  const weights = [...candidates.map(candidate => Math.exp(candidate.score)), Math.exp(other)];
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const probabilities = weights.map(weight => rounded(weight / total));
  probabilities[probabilities.length - 1] = rounded(1 - probabilities.slice(0, -1).reduce((sum, p) => sum + p, 0));
  return [
    ...candidates.map((candidate, index) => ({ id: candidate.id, piece: candidate.piece, score: candidate.score,
      probability: probabilities[index], aggregate: false })),
    { id: null, piece: null, label: 'All other tokens', score: other, probability: probabilities.at(-1), aggregate: true }
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

const copy = state => ({ ...state, pieces: [...state.pieces], ids: [...state.ids], covered: [...state.covered] });

/** Advance one stage without mutating the previous state. */
export function step(state) {
  if (state.stopped) return copy(state);
  const index = PHASES.indexOf(state.phase);
  if (index < 0) throw new RangeError('Unknown phase');
  if (state.phase === 'append') {
    return state.generated >= MAX_GENERATED
      ? { ...copy(state), stopped: true, stopReason: 'demo-limit' }
      : { ...copy(state), phase: 'tokens', selection: null };
  }
  const next = { ...copy(state), phase: PHASES[index + 1] };
  if (next.phase !== 'append') return next;
  const selection = chooseToken(distributionFor(state), state);
  next.selection = selection;
  next.seed = selection.seed;
  if (selection.aggregate) return { ...next, stopped: true, stopReason: 'outside-fixture' };
  next.pieces.push(selection.piece);
  next.ids.push(selection.id);
  next.text = next.pieces.join('');
  next.generated += 1;
  return next;
}

/** Replay from the same original seed, selection mode and prompt. */
export function reset(state) {
  return createState({ mode: state.mode, seed: state.initialSeed, subject: state.subject, verb: state.verb, covered: state.covered });
}
