import { createState, step } from './mechanics.mjs';
let state = createState({ mode: process.argv.includes('--sample') ? 'sample' : 'greedy', seed: 42 });
console.log('Teaching fixture only: synthetic probabilities, no model inference.');
while (!state.stopped) {
  console.log(JSON.stringify(state));
  state = step(state);
}
console.log(JSON.stringify(state));
