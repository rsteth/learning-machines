import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';
import {RIG,solveGrasp,relativeGraspError,numericalConstraintRank,poseFromCoordinates,encodeTaskCoordinates,costAt,fitPCA,reconstructAndScore,encodePCA,decodePCA,projectDecodedConfiguration,planGridPath} from './mechanics.mjs';
export const SEED=308141;
function random(seed){let state=seed>>>0;return ()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};}
const compactSolver=r=>({ok:r.ok,reason:r.reason,iterations:r.iterations,limitHit:r.limitHit,positionResidualM:r.positionResidualM,angularResidualRad:r.angularResidualRad});
// A PCA eigenvector has no intrinsic sign. Choose a dataset-specific display gauge
// using training physical coordinates only; do not alter the generic eigensolver.
export function orientPCAForDisplay(fitted,trainingSamples){
  const pca={...fitted,eigenvectors:fitted.eigenvectors.map(v=>[...v])};
  const mean=a=>a.reduce((sum,v)=>sum+v,0)/a.length;
  pca.orientation=[{component:1,axis:'y',axisIndex:1,reference:'forward distance'},
    {component:2,axis:'z',axisIndex:2,reference:'height'}].map(({axisIndex,...description},k)=>{
    const scores=trainingSamples.map(sample=>encodePCA(sample.q,pca)[k]);
    const physical=trainingSamples.map(sample=>sample.ballPose.position[axisIndex]);
    const scoreMean=mean(scores),physicalMean=mean(physical);
    let cross=0,scoreSquared=0,physicalSquared=0;
    for(let i=0;i<scores.length;i++){const a=scores[i]-scoreMean,b=physical[i]-physicalMean;cross+=a*b;scoreSquared+=a*a;physicalSquared+=b*b;}
    const denominator=Math.sqrt(scoreSquared*physicalSquared);
    const originalCorrelation=denominator>0?cross/denominator:0;
    const correlationThreshold=1e-10,fallback=Math.abs(originalCorrelation)<=correlationThreshold;
    const signApplied=!fallback&&cross<0?-1:1;
    if(signApplied<0)pca.eigenvectors[k]=pca.eigenvectors[k].map(v=>-v);
    return {...description,signApplied,trainingCount:trainingSamples.length,
      originalCovariance:cross/(trainingSamples.length-1),trainingCovariance:signApplied*cross/(trainingSamples.length-1),
      originalCorrelation,trainingCorrelation:signApplied*originalCorrelation,correlationThreshold,fallback,
      rule:'Positive training score/physical-coordinate covariance; preserve eigensolver sign when correlation magnitude ≤ threshold.',
      interpretation:'Display sign convention only, not a semantic or disentangled coordinate.'};
  });
  return pca;
}
export function generateFixtures(seed=SEED){
  const rng=random(seed),canonical=solveGrasp();if(!canonical.ok)throw new Error('Canonical grasp failed');
  const y=Array.from({length:21},(_,i)=>.25+i*.021),z=Array.from({length:21},(_,i)=>.92+i*.028),cells=[];
  for(let j=0;j<z.length;j++)for(let i=0;i<y.length;i++){
    const ballPose=poseFromCoordinates(0,y[i],z[j]),solver=solveGrasp({ballPose,initialQ:canonical.q}),costs=solver.ok?costAt(solver.q):null,valid=solver.ok&&costs.clearance>0;
    cells.push({index:cells.length,row:j,column:i,valid,ballPose,q:solver.ok?solver.q:null,costs:valid?costs:null,solver:compactSolver(solver),invalidReason:valid?null:solver.ok?'capsule-overlap':solver.reason});
  }
  const landscape={width:y.length,height:z.length,axes:{y,z},slice:{x:0,roll:0,pitch:0,yaw:0,leftPosture:0,rightPosture:0},cells,
    validity:'IK converged, within joint limits, and capsule clearance > 0; not dynamic feasibility',fields:{torque:{label:'Ball-load torque estimate',unit:'N·m'},jointLimit:{label:'Joint-limit proximity',unit:'normalized index'},clearanceCost:{label:'Clearance proxy cost',unit:'normalized index'}}};
  const nearest=(yy,zz)=>cells.filter(c=>c.valid).reduce((best,c)=>(c.ballPose.position[1]-yy)**2+(c.ballPose.position[2]-zz)**2<(best.ballPose.position[1]-yy)**2+(best.ballPose.position[2]-zz)**2?c:best).index;
  landscape.start=nearest(.43,1.2);landscape.goal=nearest(.55,1.36);landscape.exampleRoutes={shortest:planGridPath(landscape,landscape.start,landscape.goal,{shortest:true}),lowCost:planGridPath(landscape,landscape.start,landscape.goal)};
  const train=[],holdout=[];let maximumPositionResidualM=0,maximumAngularResidualRad=0;
  for(let trajectory=0;trajectory<24;trajectory++){
    const offset=(rng()-.5)*.8,amplitude=.2+rng()*.4,extent=.7+rng()*.3,lateral=(rng()-.5)*.012;let initialQ=canonical.q;
    for(let step=0;step<15;step++){
      const t=step/14,reach=extent*(2*t-1),present=offset+amplitude*Math.sin(Math.PI*t),target=encodeTaskCoordinates(reach,present);target.ballPose.position[0]+=lateral*Math.sin(Math.PI*t);
      const solver=solveGrasp({...target,initialQ});if(!solver.ok)throw new Error(`Dataset IK failed ${trajectory}/${step}: ${solver.reason}`);
      const residual=relativeGraspError(solver.q);maximumPositionResidualM=Math.max(maximumPositionResidualM,residual.positionResidualM);maximumAngularResidualRad=Math.max(maximumAngularResidualRad,residual.angularResidualRad);
      const sample={trajectory,step,reach,present,lateral,ballPose:target.ballPose,q:solver.q,solver:compactSolver(solver)};(trajectory%4===3?holdout:train).push(sample);initialQ=solver.q;
    }
  }
  const pca=orientPCAForDisplay(fitPCA(train.map(s=>s.q)),train);pca.coordinateRanges=[0,1].map(k=>{const scores=train.map(s=>encodePCA(s.q,pca)[k]);return [Math.min(...scores),Math.max(...scores)];});
  const scores={train:reconstructAndScore(train.map(s=>s.q),pca),holdout:reconstructAndScore(holdout.map(s=>s.q),pca)};
  let projectedSquared=0,projectedCount=0,failed=0;
  for(const item of holdout){const projected=projectDecodedConfiguration(decodePCA(encodePCA(item.q,pca),pca));if(!projected.ok){failed++;continue;}projectedSquared+=projected.q.reduce((sum,v,i)=>sum+(v-item.q[i])**2,0);projectedCount++;}
  scores.holdoutProjected={rmsRad:Math.sqrt(projectedSquared/(14*projectedCount)),rmsDeg:Math.sqrt(projectedSquared/(14*projectedCount))*180/Math.PI,count:projectedCount,failed};
  const sample=holdout[Math.floor(holdout.length/2)],encoded=encodePCA(sample.q,pca),decoded=decodePCA(encoded,pca),projection=projectDecodedConfiguration(decoded);
  const rms=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]))/Math.sqrt(14);
  const reconstructionExample={sample,encoded,decoded,projection,beforeRmsRad:rms(decoded,sample.q),afterRmsRad:projection.ok?rms(projection.q,sample.q):null};
  const outsideTarget={ballPose:poseFromCoordinates(.13,.43,1.33),leftPosture:.1,rightPosture:-.1,initialQ:canonical.q},outside=solveGrasp(outsideTarget);
  if(!outside.ok)throw new Error('Outside-family example not reachable');
  const outsideDecoded=decodePCA(encodePCA(outside.q,pca),pca),outsideProjection=projectDecodedConfiguration(outsideDecoded);
  const outsideSkill={label:'Sideward presentation: outside the skill dataset',ballPose:outsideTarget.ballPose,q:outside.q,solver:compactSolver(outside),encoded:encodePCA(outside.q,pca),decoded:outsideDecoded,projection:outsideProjection,reconstructionRmsRad:rms(outside.q,outsideDecoded),projectedRmsRad:outsideProjection.ok?rms(outside.q,outsideProjection.q):null};
  return {seed,generatorVersion:'manifolds-fixtures-2',rigVersion:RIG.version,unitDefinitions:RIG.units,provenance:'Original seeded synthetic kinematic demonstrations, not observations of a physical robot. MIT.',
    canonical:{q:canonical.q,ballPose:RIG.canonicalBallPose,solver:compactSolver(canonical),grasp:relativeGraspError(canonical.q),rank:numericalConstraintRank(canonical.q)},graspAttachmentFrames:RIG.contactFrames,
    landscape,demonstrations:{train,holdout,sampleMetadata:{trajectories:24,samplesPerTrajectory:15,split:'Whole trajectories: index modulo four equals three is holdout; no holdout enters PCA fit.',maximumPositionResidualM,maximumAngularResidualRad}},pca,scores,reconstructionExample,outsideSkill};
}
export function fixtureSource(seed=SEED){return `// Generated by generate-fixtures.mjs. Seed ${seed}. Original synthetic data; MIT.\nexport const FIXTURES = ${JSON.stringify(generateFixtures(seed),null,2)};\n`;}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const destination=process.argv[2]||fileURLToPath(new URL('./fixtures.mjs',import.meta.url));writeFileSync(destination,fixtureSource());console.log(`Wrote deterministic fixture to ${destination}`);
}
