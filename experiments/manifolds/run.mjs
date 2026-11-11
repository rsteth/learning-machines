import {FIXTURES as f} from './fixtures.mjs';
import {relativeGraspError,numericalConstraintRank,costAt,reconstructAndScore} from './mechanics.mjs';
const cell=f.landscape.cells[f.landscape.start],example=f.reconstructionExample;
console.log(JSON.stringify({study:'Learning Machines / 03 — manifolds and motion',seed:f.seed,rigVersion:f.rigVersion,generatorVersion:f.generatorVersion,
  canonical:{q:f.canonical.q,ballPose:f.canonical.ballPose,residual:relativeGraspError(f.canonical.q),rank:numericalConstraintRank(f.canonical.q).rank},
  landscape:{shape:[f.landscape.width,f.landscape.height],validCells:f.landscape.cells.filter(c=>c.valid).length,sampledCell:cell.index,ballPose:cell.ballPose,costs:costAt(cell.q),routes:f.landscape.exampleRoutes},
  pca:{trainingSamples:f.demonstrations.train.length,holdout:reconstructAndScore(f.demonstrations.holdout.map(s=>s.q),f.pca),holdoutProjected:f.scores.holdoutProjected,example:{coordinates:example.encoded,rawRmsRad:example.beforeRmsRad,projectedRmsRad:example.afterRmsRad,rawGrasp:example.projection.before,projectedGrasp:example.projection.after},outsideFamilyRmsRad:f.outsideSkill.reconstructionRmsRad}},null,2));
