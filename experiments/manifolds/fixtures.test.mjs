import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {FIXTURES as f} from './fixtures.mjs';
import {fixtureSource,SEED,orientPCAForDisplay} from './generate-fixtures.mjs';
import {RIG,forwardKinematics,composePose,inversePose,rotationLog,relativeGraspError,withinJointLimits,costAt,fitPCA,reconstructAndScore,planGridPath,projectDecodedConfiguration,encodePCA,decodePCA} from './mechanics.mjs';
test('committed fixture is byte-for-byte regenerated, no network or random clock',()=>{assert.equal(f.seed,SEED);assert.equal(fixtureSource(),readFileSync(new URL('./fixtures.mjs',import.meta.url),'utf8'));});
test('every accepted grid pose satisfies hand targets, grasp, limits and disclosed finite costs',()=>{
  assert.equal(f.landscape.cells.length,441);assert.ok(f.landscape.cells.some(c=>!c.valid));
  for(const c of f.landscape.cells){if(!c.valid){assert.equal(c.costs,null);assert.ok(c.invalidReason);continue;}assert.ok(c.solver.ok);assert.ok(withinJointLimits(c.q));assert.deepEqual(c.costs,costAt(c.q));assert.ok(c.costs.clearance>0);
    const g=relativeGraspError(c.q);assert.ok(g.positionResidualM<.001);assert.ok(g.angularResidualRad<.5*Math.PI/180);
    const fk=forwardKinematics(c.q);for(const side of ['left','right']){const target=composePose(c.ballPose,RIG.contactFrames[side]);const error=composePose(inversePose(target),fk[side]);assert.ok(Math.hypot(...error.position)<.000021);assert.ok(Math.hypot(...rotationLog(error.rotation))<.000041);}
  }
});
test('routes use only valid graph edges and planner beats shortest route under weighted objective',()=>{
  const g=f.landscape;for(const field of ['torque','jointLimit','clearanceCost']){const route=planGridPath(g,g.start,g.goal,{field}),short=planGridPath(g,g.start,g.goal,{field,shortest:true});assert.ok(route.ok);assert.ok(short.ok);
    const score=path=>path.slice(1).reduce((sum,v,i)=>{const u=path[i],a=g.cells[u],b=g.cells[v],distance=Math.hypot(...a.ballPose.position.map((x,j)=>x-b.ballPose.position[j]));return sum+distance*(1+4*(a.costs[field]+b.costs[field])/(2*route.normalizationMax));},0);
    assert.ok(route.cost<=score(short.path)+1e-10);for(let i=0;i<route.path.length;i++){assert.ok(g.cells[route.path[i]].valid);if(i){const a=g.cells[route.path[i-1]],b=g.cells[route.path[i]];assert.equal(Math.abs(a.row-b.row)+Math.abs(a.column-b.column),1);}}
  }
});
test('training/holdout trajectories are disjoint and PCA derives only from training samples',()=>{
  const trainIds=new Set(f.demonstrations.train.map(s=>s.trajectory));assert.ok(f.demonstrations.holdout.every(s=>!trainIds.has(s.trajectory)));
  const pca=orientPCAForDisplay(fitPCA(f.demonstrations.train.map(s=>s.q)),f.demonstrations.train);assert.deepEqual(pca.mean,f.pca.mean);assert.deepEqual(pca.eigenvectors,f.pca.eigenvectors);assert.deepEqual(pca.eigenvalues,f.pca.eigenvalues);
  assert.deepEqual(reconstructAndScore(f.demonstrations.holdout.map(s=>s.q),f.pca),f.scores.holdout);
  for(const s of [...f.demonstrations.train,...f.demonstrations.holdout]){assert.ok(s.solver.ok);assert.ok(relativeGraspError(s.q).positionResidualM<.001);}
  assert.ok(f.scores.holdout.rmsDeg>0);assert.ok(f.scores.holdout.explainedVariance<1);assert.ok(f.outsideSkill.reconstructionRmsRad>f.scores.holdout.rmsRad*10);
});
test('all held-out reconstructions report genuine raw/projected errors and projection statuses',()=>{
  let squared=0,count=0,failed=0;for(const s of f.demonstrations.holdout){const r=projectDecodedConfiguration(decodePCA(encodePCA(s.q,f.pca),f.pca));if(!r.ok){failed++;continue;}squared+=r.q.reduce((sum,v,i)=>sum+(v-s.q[i])**2,0);count++;assert.ok(r.after.positionResidualM<.00002);}
  assert.equal(count,f.scores.holdoutProjected.count);assert.equal(failed,f.scores.holdoutProjected.failed);assert.equal(Math.sqrt(squared/(count*14)),f.scores.holdoutProjected.rmsRad);
});
test('display signs use training-only references deterministically and retain PCA mathematics',()=>{
  const train=f.demonstrations.train,raw=fitPCA(train.map(s=>s.q)),oriented=orientPCAForDisplay(raw,train);
  assert.deepEqual(oriented,orientPCAForDisplay(raw,train));assert.deepEqual(oriented.orientation,f.pca.orientation);
  assert.deepEqual(oriented.mean,raw.mean);assert.deepEqual(oriented.eigenvalues,raw.eigenvalues);
  assert.equal(oriented.orientation[0].signApplied,-1);assert.equal(oriented.orientation[1].signApplied,1);
  assert.ok(oriented.orientation[0].trainingCorrelation>.99);assert.ok(oriented.orientation[1].trainingCorrelation>.99);
  assert.ok(oriented.orientation.every(o=>o.trainingCount===270&&!o.fallback&&o.trainingCovariance>0));
  for(let k=0;k<14;k++){const sign=k<2?oriented.orientation[k].signApplied:1;assert.deepEqual(oriented.eigenvectors[k],raw.eigenvectors[k].map(v=>sign*v));}
  // Every outer product, hence covariance and retained subspace projectors, is identical.
  for(let k=0;k<14;k++)for(let i=0;i<14;i++)for(let j=0;j<14;j++)assert.equal(oriented.eigenvectors[k][i]*oriented.eigenvectors[k][j],raw.eigenvectors[k][i]*raw.eigenvectors[k][j]);
  for(const s of [...train,...f.demonstrations.holdout])for(const rank of [2,14])assert.deepEqual(decodePCA(encodePCA(s.q,oriented,rank),oriented),decodePCA(encodePCA(s.q,raw,rank),raw));
  assert.deepEqual(reconstructAndScore(f.demonstrations.holdout.map(s=>s.q),oriented),reconstructAndScore(f.demonstrations.holdout.map(s=>s.q),raw));
  // The API receives training samples only. No holdout values enter orientation.
  const alteredHoldout=f.demonstrations.holdout.map(s=>({...s,ballPose:{...s.ballPose,position:[1e6,-1e6,-1e6]}}));
  const split={train,holdout:alteredHoldout};assert.deepEqual(orientPCAForDisplay(raw,split.train),oriented);
  const constantReference=train.map(s=>({...s,ballPose:{...s.ballPose,position:[0,0,0]}}));
  const fallback=orientPCAForDisplay(raw,constantReference);assert.deepEqual(fallback.eigenvectors,raw.eigenvectors);assert.ok(fallback.orientation.every(o=>o.fallback&&o.signApplied===1&&o.trainingCorrelation===0));
});
test('display-sign revision leaves physical canonical state, cost grid and demonstrations byte-equivalent',()=>{
  // JSON SHA-256 baselines from mechanics revision 17c6bf02, before display sign orientation.
  const baseline={canonical:'b368f5ae909a5420dd0876f88f7b417d7efe6d423bc27baa8405eeb8582cfd8b',landscape:'27dd32dd7656bd83525d657a7e64d8b386a53a59214d9857bdeb7597d13eb392',demonstrations:'ca37ed99c20f72fe7b3777802500773b2af88603604fca5d8940779f014aa5c8'};
  for(const [key,hash] of Object.entries(baseline))assert.equal(createHash('sha256').update(JSON.stringify(f[key])).digest('hex'),hash);
});
