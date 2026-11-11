import { createState, step, distributionFor, chooseToken, traceFor, SUBJECTS, VERBS } from './mechanics.mjs';
// Teaching fixture only. Flags:
//   --sample        seeded sampling instead of greedy
//   --seed N        uint32 seed (default 42); the companion page's sampler shows the same draws
//   --draws N       skip the walk-through: draw N times in a row from the first distribution and count
//   --subject NAME  cat (default) or king
//   --verb NAME     sat (default) or slept
//   --trace         print the last piece's legs: what it heard, what it worked out, where it ended
const value = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const seed = value('--seed') === undefined ? 42 : Number(value('--seed'));
const draws = value('--draws') === undefined ? undefined : Number(value('--draws'));
const subject = value('--subject') ?? SUBJECTS[0];
const verb = value('--verb') ?? VERBS[0];
const label = item => item.aggregate ? 'all other tokens' : JSON.stringify(item.piece);
const draw = selection => (selection.seed / 4294967296).toFixed(3);
console.log('Teaching fixture only: hand-authored notes and scores, no model inference.');

if (process.argv.includes('--trace')) {
  const trace = traceFor(createState({ subject, verb }));
  for (const leg of trace.legs) {
    const source = leg.kind === 'worked' ? `worked out from ${leg.from.join(' + ')}` : `${leg.kind} ${JSON.stringify(leg.piece)}`;
    console.log(`# ${source.padEnd(40)} wrote ${JSON.stringify(leg.wrote)}  view ${JSON.stringify(leg.viewTo)}`);
  }
  console.log(`# stored numbers ${JSON.stringify(trace.output)}`);
  for (const item of distributionFor(createState({ subject, verb }))) {
    console.log(`# ${label(item).padEnd(18)} score ${item.score.toFixed(3)}  probability ${(item.probability * 100).toFixed(1)}%`);
  }
} else if (draws !== undefined) {
  if (!Number.isInteger(draws) || draws < 1 || draws > 1e6) throw new RangeError('--draws must be a whole number from 1 to 1000000');
  createState({ mode: 'sample', seed }); // validates the seed
  const distribution = distributionFor(createState({ subject, verb }));
  const counts = distribution.map(() => 0);
  let s = seed;
  for (let i = 0; i < draws; i++) {
    const pick = chooseToken(distribution, { mode: 'sample', seed: s });
    if (i === 0) console.log(`# seed ${seed} draws ${draw(pick)} -> ${label(pick)}`);
    counts[distribution.findIndex(d => d.aggregate ? pick.aggregate : d.piece === pick.piece)]++;
    s = pick.seed;
  }
  distribution.forEach((item, i) => console.log(`# ${label(item).padEnd(18)} ${String(counts[i]).padStart(7)}  expected ~${Math.round(item.probability * draws)}`));
} else {
  let state = createState({ mode: process.argv.includes('--sample') ? 'sample' : 'greedy', seed, subject, verb });
  while (!state.stopped) {
    console.log(JSON.stringify(state));
    const before = state.phase;
    state = step(state);
    if (before !== 'append' && state.phase === 'append' && state.selection && state.mode === 'sample') console.log(`# draw ${draw(state.selection)} -> ${label(state.selection)}`);
  }
  console.log(JSON.stringify(state));
}
