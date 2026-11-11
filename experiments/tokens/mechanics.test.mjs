import test from 'node:test';
import assert from 'node:assert/strict';
import { FIXTURE, PHASES, MAX_GENERATED, WIDTH, NOTES, DIRECTIONS, PIECES, UNITS, CANDIDATES, OTHER_SCORE, LISTEN_GAIN,
  SUBJECTS, VERBS, VIEW, createState, step, reset, decode, idFor, embeddingFor, readNotes, writeNotes, scoreDirection,
  viewOf, traceFor, scoresFor, distributionFor, chooseToken } from './mechanics.mjs';

const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const top = distribution => distribution.reduce((best, item) => item.probability > best.probability ? item : best);
const advance = (state, steps) => { for (let i = 0; i < steps; i++) state = step(state); return state; };

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
  assert.deepEqual(createState().ids, [791, 8415, 7731, 389, 279]);
  assert.equal(idFor(' mat'), 5634);
  assert.equal(idFor('not a fixture piece'), null);
});

test('every model piece has a real fixture ID once the fixture is regenerated', t => {
  const missing = Object.keys(PIECES).filter(piece => idFor(piece) === null);
  if (missing.length) {
    t.skip(`regenerate_fixture.py has not been run for: ${missing.map(piece => JSON.stringify(piece)).join(', ')}`);
    return;
  }
  for (const piece of Object.keys(PIECES)) assert.equal(decode([idFor(piece)]), piece);
});

test('five stages, two bounded greedy cycles, immutable history and reset', () => {
  const initial = createState();
  let state = initial;
  for (let cycle = 0; cycle < MAX_GENERATED; cycle++) {
    for (const phase of PHASES) {
      assert.equal(state.phase, phase);
      const before = structuredClone(state);
      const next = step(state);
      assert.deepEqual(state, before);
      state = next;
    }
  }
  assert.equal(state.text, 'The cat sat on the mat.');
  assert.deepEqual(state.ids, [791, 8415, 7731, 389, 279, 5634, 13]);
  assert.equal(state.generated, 2);
  assert.equal(state.stopped, true);
  assert.equal(state.stopReason, 'demo-limit');
  assert.deepEqual(step(state), state);
  assert.deepEqual(reset(state), initial);
});

test('six primary notes are orthogonal unit directions aligned with no stored number', () => {
  const primary = NOTES.filter(note => !note.blendOf);
  assert.equal(primary.length, WIDTH);
  for (const a of primary) {
    assert.equal(DIRECTIONS[a.key].length, WIDTH);
    assert.ok(DIRECTIONS[a.key].filter(value => value !== 0).length >= 5, 'a note is spread across the stored numbers');
    for (const b of primary) close(dot(DIRECTIONS[a.key], DIRECTIONS[b.key]), a === b ? 1 : 0);
  }
  assert.throws(() => { DIRECTIONS.cat[0] = 5; }, TypeError);
});

test('eight notes share six numbers: blended notes are unit length and overlap their ingredients', () => {
  assert.equal(NOTES.length, 8);
  for (const note of NOTES.filter(note => note.blendOf)) {
    close(dot(DIRECTIONS[note.key], DIRECTIONS[note.key]), 1);
    for (const [key, amount] of Object.entries(note.blendOf)) close(dot(DIRECTIONS[note.key], DIRECTIONS[key]), amount);
  }
  // Writing one note makes an overlapping note read partly lit, without anything writing it.
  const reads = readNotes(writeNotes({ cat: 1 }));
  close(reads.cat, 1);
  close(reads.cosy, 0.7);
  close(reads.royal, 0);
});

test('writing then reading primary notes round-trips; IDs only select rows', () => {
  const amounts = { thing: 0.3, surface: -1.2, sitting: 2, sleeping: 0, cat: 0.25, royal: 1 };
  const reads = readNotes(writeNotes(amounts));
  for (const [key, amount] of Object.entries(amounts)) close(reads[key], amount);
  assert.deepEqual(embeddingFor(' cat'), [...DIRECTIONS.cat]);
  assert.deepEqual(embeddingFor('The'), embeddingFor(' the'));
  assert.throws(() => embeddingFor(' dog'), RangeError);
});

test('listening shares are a softmax of match scores over earlier pieces only', () => {
  const state = createState();
  for (let row = 0; row < state.pieces.length; row++) {
    const heard = traceFor(state, row).legs.filter(leg => leg.kind === 'heard');
    assert.equal(heard.length, row);
    assert.ok(heard.every(leg => leg.index < row));
    if (row) close(heard.reduce((sum, leg) => sum + leg.share, 0), 1);
    for (const leg of heard) {
      close(leg.amount, LISTEN_GAIN * leg.share);
      const total = state.pieces.slice(0, row).reduce((sum, piece) => sum + Math.exp(PIECES[piece].match), 0);
      close(leg.share, Math.exp(PIECES[leg.piece].match) / total);
    }
  }
  const legs = traceFor(state).legs.filter(leg => leg.kind === 'heard');
  assert.deepEqual(legs.map(leg => leg.piece), [' on', ' sat', ' cat', 'The'], 'nearest piece first');
  assert.ok(legs.find(leg => leg.piece === 'The').share < 0.06, '"The" is barely heard');
  assert.ok(legs.find(leg => leg.piece === ' cat').share > 0.3, 'content is heard by relevance, not distance');
});

test('a covered piece keeps its share but offers nothing', () => {
  const open = traceFor(createState());
  const shut = traceFor(createState({ covered: [1] }));
  const openLeg = open.legs.find(leg => leg.index === 1 && leg.kind === 'heard');
  const shutLeg = shut.legs.find(leg => leg.index === 1 && leg.kind === 'heard');
  assert.equal(shutLeg.share, openLeg.share);
  assert.deepEqual(shutLeg.wrote, {});
  assert.deepEqual(shutLeg.delta, Array(WIDTH).fill(0));
  close(shut.notes.output.cat, 0);
});

test('work-out units fire only when their ingredients are heard together', () => {
  const amount = (options, key) => traceFor(createState(options)).units.find(unit => unit.key === key).amount;
  assert.ok(amount({}, 'cosy') > 1);
  assert.equal(amount({}, 'grand'), 0);
  assert.equal(amount({ covered: [1] }, 'cosy'), 0);
  assert.equal(amount({ covered: [2] }, 'cosy'), 0);
  assert.ok(amount({ subject: 'king' }, 'grand') > 1);
  assert.equal(amount({ subject: 'king' }, 'cosy'), 0);
  assert.equal(amount({ subject: 'king', verb: 'slept' }, 'grand'), 0);
  const trace = traceFor(createState());
  for (const unit of trace.units) {
    const spec = UNITS.find(candidate => candidate.key === unit.key);
    close(unit.input, Object.entries(spec.reads).reduce((sum, [key, weight]) => sum + weight * trace.notes.afterListening[key], 0));
    close(unit.amount, spec.gain * Math.max(0, unit.input - spec.threshold));
  }
  assert.equal(trace.legs.filter(leg => leg.kind === 'worked').length, 1);
});

test('legs add up to the final card, in stored numbers and in the view', () => {
  for (const subject of SUBJECTS) for (const verb of VERBS) for (const covered of [[], [1], [2], [0, 1, 2, 3]]) {
    const trace = traceFor(createState({ subject, verb, covered }));
    const sum = trace.legs.reduce((row, leg) => row.map((value, index) => value + leg.delta[index]), Array(WIDTH).fill(0));
    sum.forEach((value, index) => close(value, trace.output[index]));
    assert.deepEqual(trace.legs.at(-1).after, trace.output);
    assert.deepEqual(trace.legs[0].viewFrom, [0, 0, 0]);
    trace.legs.forEach((leg, index) => {
      if (index) assert.deepEqual(leg.viewFrom, trace.legs[index - 1].viewTo);
      const moved = viewOf(leg.delta);
      moved.forEach((value, axis) => close(leg.viewTo[axis] - leg.viewFrom[axis], value));
    });
  }
  assert.deepEqual(viewOf(embeddingFor(' cat')), [...VIEW.cat]);
});

test('scores are computed from the final card and split exactly by leg', () => {
  const { trace, candidates, other } = scoresFor(createState());
  assert.equal(other, OTHER_SCORE);
  assert.deepEqual(candidates.map(candidate => candidate.piece), CANDIDATES.map(candidate => candidate.piece));
  for (const candidate of candidates) {
    close(candidate.score, dot(trace.output, scoreDirection(CANDIDATES.find(c => c.piece === candidate.piece))));
    close(candidate.parts.reduce((sum, part) => sum + part, 0), candidate.score);
    assert.equal(candidate.parts.length, trace.legs.length);
    assert.ok(candidate.parts.every(part => part >= 0));
  }
});

test('the next-token distribution follows the context', () => {
  const expected = { 'cat sat': ' mat', 'king sat': ' throne', 'king slept': ' bed', 'cat slept': ' mat' };
  for (const subject of SUBJECTS) for (const verb of VERBS) {
    const distribution = distributionFor(createState({ subject, verb }));
    close(distribution.reduce((sum, item) => sum + item.probability, 0), 1);
    assert.equal(distribution.length, CANDIDATES.length + 1);
    assert.equal(distribution.at(-1).aggregate, true);
    assert.equal(top(distribution).piece, expected[`${subject} ${verb}`]);
    const scores = distribution.map(item => Math.exp(item.score));
    const total = scores.reduce((sum, value) => sum + value, 0);
    distribution.forEach((item, index) => close(item.probability, scores[index] / total, 1e-9));
  }
  assert.ok(top(distributionFor(createState())).probability > 0.5);
  // With the content covered, nothing in the fixture stands out.
  for (const covered of [[1, 2], [1, 2, 3], [0, 1, 2, 3]]) assert.equal(top(distributionFor(createState({ covered }))).aggregate, true);
  assert.notDeepEqual(distributionFor(createState()), distributionFor(createState({ subject: 'king' })));
});

test('the second step is scripted punctuation, then the demo stops', () => {
  const state = advance(createState(), 5);
  assert.equal(state.generated, 1);
  assert.deepEqual(distributionFor(state), [{ id: 13, piece: '.', probability: 1, aggregate: false, scripted: true }]);
  assert.deepEqual(distributionFor(advance(state, 5)), []);
});

test('seeded sampling is reproducible and reaches every bucket', () => {
  const distribution = distributionFor(createState({ subject: 'king', verb: 'slept' }));
  const seen = new Set();
  for (let seed = 0; seed < 4000; seed++) {
    const pick = chooseToken(distribution, { mode: 'sample', seed });
    assert.deepEqual(pick, chooseToken(distribution, { mode: 'sample', seed }));
    seen.add(pick.aggregate ? 'other' : pick.piece);
  }
  assert.deepEqual(seen, new Set([...CANDIDATES.map(candidate => candidate.piece), 'other']));
  assert.equal(chooseToken(distribution).piece, ' bed');
});

test('a chosen aggregate stops honestly without inventing or decoding a token', () => {
  const start = createState({ covered: [0, 1, 2, 3] });
  const state = advance(start, 4);
  assert.equal(state.selection.aggregate, true);
  assert.equal(state.stopReason, 'outside-fixture');
  assert.equal(state.generated, 0);
  assert.equal(state.text, 'The cat sat on the');
  assert.deepEqual(state.ids, FIXTURE.examples[0].ids);
  assert.deepEqual(distributionFor(state), []);
  assert.deepEqual(advance(reset(state), 4), state);
});

test('earlier context changes a row; later pieces and appends cannot leak backward', () => {
  const state = createState();
  assert.notDeepEqual(traceFor(state).output, traceFor(createState({ subject: 'king' })).output);
  const longer = advance(state, 4);
  assert.equal(longer.pieces.length, 6);
  for (let row = 0; row < state.pieces.length; row++) assert.deepEqual(traceFor(longer, row), traceFor(state, row));
  const before = structuredClone(state);
  assert.deepEqual(traceFor(state), traceFor(state));
  assert.deepEqual(state, before);
});

test('invalid controls and probability arrays reject', () => {
  for (const seed of [-1, 1.5, NaN, 4294967296]) assert.throws(() => createState({ seed }), RangeError);
  assert.throws(() => createState({ mode: 'unknown' }), RangeError);
  assert.throws(() => createState({ subject: 'dog' }), RangeError);
  assert.throws(() => createState({ verb: 'ran' }), RangeError);
  for (const covered of [[4], [-1], [1.5], 'cat']) assert.throws(() => createState({ covered }), RangeError);
  assert.throws(() => traceFor(createState(), 9), RangeError);
  for (const probabilities of [[], [0.2], [-1, 2], [NaN], [Infinity]]) {
    assert.throws(() => chooseToken(probabilities.map(probability => ({ probability }))), RangeError);
  }
});
