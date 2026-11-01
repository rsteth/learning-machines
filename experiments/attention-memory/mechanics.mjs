// Teaching mechanics only. No model inference or measured GPU estimates.
export const CONFIG = Object.freeze({layers: 64, interval: 4, referenceTokens: 1024, maxTokens: 262144});
export const EVENTS = Object.freeze([
  Object.freeze({key: 'door', value: 'blue'}), Object.freeze({key: 'cup', value: 'red'}),
  Object.freeze({key: 'door', value: 'green'}), Object.freeze({key: 'cup', value: 'white'}),
  Object.freeze({key: 'door', value: 'yellow'}), Object.freeze({key: 'cup', value: 'black'})
]);
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
// Deliberately lossy dictionary: the latest value for each key replaces its history.
// It is an analogy for updating state, not a simulation of GatedDeltaNet.
export function illustrate(count) {
  if (!Number.isInteger(count) || count < 0 || count > EVENTS.length) throw new RangeError('invalid event count');
  const history = EVENTS.slice(0, count).map(event => ({...event}));
  return {history, state: Object.fromEntries(history.map(({key,value}) => [key,value]))};
}
