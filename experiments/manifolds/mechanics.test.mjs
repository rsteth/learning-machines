import test from 'node:test';
import assert from 'node:assert/strict';
import {RIG,forwardKinematics,solveGrasp,poseFromCoordinates,composePose,inversePose,rotationAround,rotationLog,relativeGraspError,numericalConstraintRank,withinJointLimits,ballLoadTorqueEstimate,collisionClearance,costAt,encodeTaskCoordinates,fitPCA,encodePCA,decodePCA,reconstructAndScore,projectDecodedConfiguration,planGridPath,symmetricEigen} from './mechanics.mjs';
const norm=x=>Math.hypot(...x),close=(a,b,t=1e-7)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
const solved=solveGrasp();
test('canonical 7R chains have real 3D position and orientation; all 14 joints act independently',()=>{
  assert.equal(solved.ok,true);const fk=forwardKinematics(solved.q);assert.equal(fk.left.joints.length,7);assert.equal(fk.right.joints.length,7);
  for(let i=0;i<14;i++){const q=[...solved.q];q[i]+=.02;const changed=forwardKinematics(q),side=i<7?'left':'right',other=i<7?'right':'left';assert.deepEqual(changed[other],fk[other]);assert.notDeepEqual(changed[side].rotation,fk[side].rotation);}
  for(const side of ['left','right']){close(norm(fk[side].points[1].map((v,i)=>v-fk[side].points[0][i])),RIG.upperArm);close(norm(fk[side].points[2].map((v,i)=>v-fk[side].points[1][i])),RIG.forearm);}
  assert.throws(()=>forwardKinematics(Array(14).fill(NaN)),TypeError);
});
test('six-component rigid grasp and numerical local dimension are substantiated',()=>{
  const residual=relativeGraspError(solved.q);assert.equal(residual.residual.length,6);assert.ok(residual.positionResidualM<.001);assert.ok(residual.angularResidualRad<.5*Math.PI/180);
  const rank=numericalConstraintRank(solved.q);assert.equal(rank.rank,6);assert.equal(rank.localDimension,8);assert.ok(rank.singularValues[5]>.3);
  // Extended arms collinear with the shoulder baseline, outside the elbow range.
  const singular=Array(14).fill(0);singular[0]=-Math.PI/2;singular[4]=Math.PI/2;singular[7]=Math.PI/2;singular[11]=-Math.PI/2;
  assert.ok(numericalConstraintRank(singular).rank<6);
});
test('SO(3) residual is stable near zero and near a half turn',()=>{
  for(const theta of [0,1e-9,.6,Math.PI-1e-5,Math.PI-1e-8,Math.PI]){const axis=[1/Math.sqrt(3),1/Math.sqrt(3),1/Math.sqrt(3)],R=rotationAround(axis,theta);close(norm(rotationLog(R)),theta,1e-7);}
  const p=poseFromCoordinates(.1,.2,.3,.4,-.5,.6),identity=composePose(p,inversePose(p));assert.ok(norm(identity.position)<1e-12);assert.ok(norm(rotationLog(identity.rotation))<1e-12);
});
test('bounded IK respects full hand frames, both redundant posture preferences, warm starts and failures',()=>{
  for(const reach of [-1,0,1])for(const present of [-1,0,1]){const args=encodeTaskCoordinates(reach,present),r=solveGrasp({...args,initialQ:solved.q});assert.equal(r.ok,true);assert.ok(withinJointLimits(r.q));assert.ok(r.positionResidualM<.00002);assert.ok(r.angularResidualRad<.00004);assert.ok(r.iterations<=100);}
  for(const v of [-.35,.35]){const r=solveGrasp({leftPosture:v,rightPosture:-v,initialQ:solved.q});assert.ok(r.ok);close(r.q[2],v,.009);close(r.q[9],-v,.009);assert.ok(r.positionResidualM<.00002);}
  assert.equal(solveGrasp({ballPose:poseFromCoordinates(0,2,1)}).reason,'unreachable');
  assert.equal(solveGrasp({ballPose:{position:[0,0,0],rotation:[[1,0,0],[0,1,0],[0,0,-1]]}}).reason,'invalid-input');
  assert.equal(solveGrasp({initialQ:[NaN]}).ok,false);assert.equal(solveGrasp({maxIterations:0}).reason,'iteration-limit');
  assert.equal(solveGrasp({leftPosture:10}).ok,false);assert.throws(()=>encodeTaskCoordinates(2,0),RangeError);
});
test('analytic JᵀF torque matches finite-difference hand height potential; scales with mass',()=>{
  const estimate=ballLoadTorqueEstimate(solved.q),h=1e-6,force=RIG.ballMass*RIG.gravity/2;
  for(let i=0;i<14;i++){const a=[...solved.q],b=[...solved.q];a[i]+=h;b[i]-=h;const side=i<7?'left':'right',numeric=(forwardKinematics(a)[side].position[2]-forwardKinematics(b)[side].position[2])*force/(2*h);close(estimate.tau[i],numeric,1e-8);}
  close(ballLoadTorqueEstimate(solved.q,{mass:RIG.ballMass*2}).normNm,2*estimate.normNm);assert.equal(ballLoadTorqueEstimate(solved.q,{mass:0}).normNm,0);assert.throws(()=>ballLoadTorqueEstimate(solved.q,{mass:-1}));
  assert.ok(collisionClearance(solved.q).clearanceM>0);const c=costAt(solved.q);assert.ok(c.jointLimit>=0&&c.jointLimit<=1);assert.ok(c.clearanceCost>=0&&c.clearanceCost<=1);
});
test('Jacobi PCA fits actual covariance, orthonormal vectors and monotone reconstruction',()=>{
  const samples=Array.from({length:30},(_,i)=>Array.from({length:14},(_,j)=>Math.sin(i*.2)*(j+1)*.02+Math.cos(i*.3)*(j%3)*.04));
  const pca=fitPCA(samples);assert.ok(pca.fitDiagnostics.converged);for(let i=0;i<14;i++)for(let j=0;j<14;j++)close(pca.eigenvectors[i].reduce((s,x,k)=>s+x*pca.eigenvectors[j][k],0),i===j?1:0,1e-9);
  let previous=Infinity;for(let rank=0;rank<=14;rank++){const score=reconstructAndScore(samples,pca,rank);assert.ok(score.rmsRad<=previous+1e-12);previous=score.rmsRad;}assert.ok(previous<1e-10);
  assert.ok(reconstructAndScore(samples,pca,2).rmsRad<1e-9);assert.deepEqual(decodePCA([],pca),pca.mean);assert.equal(encodePCA(samples[0],pca).length,2);
  const e=symmetricEigen([[2,1],[1,2]]);close(e.eigenvalues[0],3);close(e.eigenvalues[1],1);
});
test('projection exposes raw nonlinear violation and corrects or fails explicitly',()=>{
  const candidate=[...solved.q];candidate[0]+=.025;candidate[8]-=.02;
  const r=projectDecodedConfiguration(candidate);assert.ok(r.before.positionResidualM>.001);assert.ok(r.ok);assert.ok(r.after.positionResidualM<.00002);assert.ok(r.after.angularResidualRad<.00004);assert.ok(r.correctionRmsRad>0);assert.ok(withinJointLimits(r.q));
  assert.equal(projectDecodedConfiguration([NaN]).ok,false);
});
test('Dijkstra follows valid cells, optimizes declared objective, rejects blocked endpoints and routes',()=>{
  const cells=Array.from({length:9},(_,i)=>({valid:true,ballPose:poseFromCoordinates(i%3,Math.floor(i/3),0),costs:{torque:i===4?100:1,jointLimit:.2,clearanceCost:.3}}));
  const grid={width:3,height:3,cells},short=planGridPath(grid,3,5,{shortest:true}),cheap=planGridPath(grid,3,5,{weight:100});assert.deepEqual(short.path,[3,4,5]);assert.equal(short.distanceM,2);assert.ok(!cheap.path.includes(4));close(cheap.cost,8);assert.equal(cheap.distanceM,4);
  cells[1].valid=cells[4].valid=cells[7].valid=false;assert.equal(planGridPath(grid,3,5).reason,'no-route');assert.equal(planGridPath(grid,3,4).reason,'invalid-endpoint');assert.equal(planGridPath(grid,3,5,{weight:-1}).reason,'invalid-options');
});
