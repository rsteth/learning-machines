// Teaching mechanics only. No model inference or measured GPU estimates.
export const CONFIG = Object.freeze({layers: 64, interval: 4, referenceTokens: 1024, maxTokens: 262144});
// A short story read piece by piece, using the pieces from study 01. Each event fills one
// slot of a two-slot summary: who the story is about, and where they are.
export const SLOTS = Object.freeze(['who', 'where']);
export const EVENTS = Object.freeze([
  Object.freeze({key: 'who', value: 'cat'}), Object.freeze({key: 'where', value: 'mat'}),
  Object.freeze({key: 'who', value: 'king'}), Object.freeze({key: 'where', value: 'throne'}),
  Object.freeze({key: 'who', value: 'cat'}), Object.freeze({key: 'where', value: 'bed'})
]);
/** The recall question both representations are asked: the first value a slot ever held. */
export const RECALL = Object.freeze({key: 'where', question: 'Where did the cat sit first?'});
// Equal abstract state units: not actual memory sizes or a Qwen memory ratio.
// One attention layer keeps n/referenceTokens units; one recurrent layer keeps 1.
export function growth(tokens) {
  if (!Number.isInteger(tokens) || tokens < 0 || tokens > CONFIG.maxTokens) throw new RangeError('tokens outside configured range');
  const n = tokens / CONFIG.referenceTokens;
  return {full: 64*n, hybrid: 16*n + 48, recurrent: 64};
}
export function architecture(converted = false) {
  return Array.from({length: 64}, (_, index) => ({index,
    kind: index % 4 === 3 ? (converted ? 'converted' : 'attention') : (converted ? 'adapted' : 'gdn')}));
}
// Deliberately lossy summary: the latest value for each slot replaces its history.
// It is an analogy for updating state, not a simulation of GatedDeltaNet.
export function illustrate(count) {
  if (!Number.isInteger(count) || count < 0 || count > EVENTS.length) throw new RangeError('invalid event count');
  const history = EVENTS.slice(0, count).map(event => ({...event}));
  return {history, state: Object.fromEntries(history.map(({key,value}) => [key,value]))};
}
