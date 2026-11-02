import { createState, step, distributionFor, chooseToken } from './mechanics.mjs';
// Teaching fixture only. Flags:
//   --sample       seeded sampling instead of greedy
//   --seed N       uint32 seed (default 42); the companion page's playground shows the same draws
//   --draws N      skip the walk-through: draw N times in a row from the first distribution and count
const value = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const seed = value('--seed') === undefined ? 42 : Number(value('--seed'));
const draws = value('--draws') === undefined ? undefined : Number(value('--draws'));
const label = item => item.aggregate ? 'all other tokens' : JSON.stringify(item.piece);
const draw = selection => (selection.seed / 4294967296).toFixed(3);
console.log('Teaching fixture only: synthetic probabilities, no model inference.');

if (draws !== undefined) {
  if (!Number.isInteger(draws) || draws < 1 || draws > 1e6) throw new RangeError('--draws must be a whole number from 1 to 1000000');
  createState({ mode: 'sample', seed }); // validates the seed
  const distribution = distributionFor(createState());
  const counts = distribution.map(() => 0);
  let s = seed;
  for (let i = 0; i < draws; i++) {
    const pick = chooseToken(distribution, { mode: 'sample', seed: s });
    if (i === 0) console.log(`# seed ${seed} draws ${draw(pick)} -> ${label(pick)}`);
    counts[distribution.findIndex(d => d.aggregate ? pick.aggregate : d.id === pick.id)]++;
    s = pick.seed;
  }
  distribution.forEach((item, i) => console.log(`# ${label(item).padEnd(18)} ${String(counts[i]).padStart(7)}  expected ~${Math.round(item.probability * draws)}`));
} else {
  let state = createState({ mode: process.argv.includes('--sample') ? 'sample' : 'greedy', seed });
  while (!state.stopped) {
    console.log(JSON.stringify(state));
    const before = state.phase;
    state = step(state);
    if (before !== 'append' && state.phase === 'append' && state.selection && state.mode === 'sample') console.log(`# draw ${draw(state.selection)} -> ${label(state.selection)}`);
  }
  console.log(JSON.stringify(state));
}
