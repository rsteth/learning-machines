// Original dependency-free teaching mechanics. MIT. Metres, radians, seconds.
const I = () => [[1,0,0],[0,1,0],[0,0,1]];
const add=(a,b)=>a.map((x,i)=>x+b[i]), sub=(a,b)=>a.map((x,i)=>x-b[i]);
const scale=(a,s)=>a.map(x=>x*s), dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const norm=a=>Math.hypot(...a), cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const transpose=A=>A[0].map((_,j)=>A.map(row=>row[j]));
const mv=(A,v)=>A.map(row=>dot(row,v));
const mm=(A,B)=>A.map(row=>transpose(B).map(col=>dot(row,col)));
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=a=>Array.isArray(a)&&a.every(Number.isFinite);
const zero=(n,m)=>Array.from({length:n},()=>Array(m).fill(0));
const axes=[[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0]];
const limits=[[-2.6,2.6],[-1.8,1.8],[-2.4,2.4],[0.08,2.7],[-2.9,2.9],[-2.3,2.3],[-2.9,2.9]];
const names=['shoulder yaw','shoulder lift','shoulder swivel','elbow flex','wrist yaw','wrist pitch','wrist roll'];
export const RIG = Object.freeze({version:'manifolds-rig-1',units:{length:'m',angle:'rad',torque:'N·m'},
  axes,limits,jointNames:names,shoulders:{left:[-.23,0,1.25],right:[.23,0,1.25]},
  upperArm:.32,forearm:.29,linkRadius:.025,ballRadius:.11,ballMass:.27,gravity:9.81,
  canonicalBallPose:{position:[0,.43,1.2],rotation:I()},
  initialQ:[-.3,-.6,0,1.3,.3,-.7,0,.3,-.6,0,1.3,-.3,-.7,0],
  contactFrames:{left:{position:[-.11,0,0],rotation:I()},right:{position:[.11,0,0],rotation:I()}},
  tolerances:{positionM:0.00002,angleRad:0.00004,maxIterations:100}});
export function rotationAround(axis,angle) {
  const [x,y,z]=axis,c=Math.cos(angle),s=Math.sin(angle),v=1-c;
  return [[x*x*v+c,x*y*v-z*s,x*z*v+y*s],[y*x*v+z*s,y*y*v+c,y*z*v-x*s],[z*x*v-y*s,z*y*v+x*s,z*z*v+c]];
}
export function poseFromCoordinates(x,y,z,roll=0,pitch=0,yaw=0) {
  return {position:[x,y,z],rotation:mm(mm(rotationAround([0,0,1],yaw),rotationAround([0,1,0],pitch)),rotationAround([1,0,0],roll))};
}
export function composePose(a,b) {return {position:add(a.position,mv(a.rotation,b.position)),rotation:mm(a.rotation,b.rotation)};}
export function inversePose(a) {const rotation=transpose(a.rotation);return {position:scale(mv(rotation,a.position),-1),rotation};}
// SO(3) logarithm: axis times principal angle, including a stable half-turn branch.
export function rotationLog(R) {
  const theta=Math.acos(clamp((R[0][0]+R[1][1]+R[2][2]-1)/2,-1,1));
  const v=[R[2][1]-R[1][2],R[0][2]-R[2][0],R[1][0]-R[0][1]];
  if(theta<1e-7) return scale(v,.5);
  if(Math.PI-theta<1e-4) {
    const d=[R[0][0],R[1][1],R[2][2]],k=d.indexOf(Math.max(...d));
    const axis=[0,0,0];axis[k]=Math.sqrt(Math.max(0,(d[k]+1)/2));
    for(let j=0;j<3;j++) if(j!==k) axis[j]=(R[k][j]+R[j][k])/(4*Math.max(axis[k],1e-12));
    if(dot(axis,v)<0) return scale(axis,-theta);return scale(axis,theta);
  }
  return scale(v,theta/(2*Math.sin(theta)));
}
function validPose(p) {
  if(!p||!finite(p.position)||p.position.length!==3||!Array.isArray(p.rotation)||p.rotation.length!==3||!p.rotation.every(r=>finite(r)&&r.length===3))return false;
  const e=mm(p.rotation,transpose(p.rotation));
  return e.every((r,i)=>r.every((v,j)=>Math.abs(v-(i===j?1:0))<1e-6))&&dot(p.rotation[0],cross(p.rotation[1],p.rotation[2]))>.999999;
}
function checkQ(q,n=14) {if(!finite(q)||q.length!==n)throw new TypeError(`Expected ${n} finite joint angles`);}
export function withinJointLimits(q) {return finite(q)&&q.length===14&&q.every((x,i)=>x>=limits[i%7][0]-1e-10&&x<=limits[i%7][1]+1e-10);}
function armFK(q,side) {
  let position=[...RIG.shoulders[side]],rotation=I();const points=[[...position]],joints=[];
  for(let i=0;i<7;i++) {
    joints.push({position:[...position],axis:mv(rotation,axes[i])});
    rotation=mm(rotation,rotationAround(axes[i],q[i]));
    if(i===2||i===3){position=add(position,mv(rotation,[0,i===2?RIG.upperArm:RIG.forearm,0]));points.push([...position]);}
  }
  return {position,rotation,points,joints};
}
export function forwardKinematics(q) {checkQ(q);return {left:armFK(q.slice(0,7),'left'),right:armFK(q.slice(7),'right')};}
function poseResidual(actual,target) {return [...sub(actual.position,target.position),...rotationLog(mm(actual.rotation,transpose(target.rotation)))];}
export const GRASP_REFERENCE={position:[.22,0,0],rotation:I()};
export function relativeGraspError(q,reference=GRASP_REFERENCE) {
  const {left,right}=forwardKinematics(q),relative=composePose(inversePose(left),right),residual=poseResidual(relative,reference);
  return {residual,positionResidualM:norm(residual.slice(0,3)),angularResidualRad:norm(residual.slice(3)),relative};
}
function jacobian(fn,q,eps=1e-6) {
  const n=fn(q).length,J=zero(n,q.length);
  for(let j=0;j<q.length;j++){const a=[...q],b=[...q];a[j]+=eps;b[j]-=eps;const x=fn(a),y=fn(b);for(let i=0;i<n;i++)J[i][j]=(x[i]-y[i])/(2*eps);}
  return J;
}
function linearSolve(A,b) {
  const n=b.length,M=A.map((r,i)=>[...r,b[i]]);
  for(let k=0;k<n;k++){
    let p=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[p][k]))p=i;
    if(Math.abs(M[p][k])<1e-15)return null;[M[p],M[k]]=[M[k],M[p]];
    const v=M[k][k];for(let j=k;j<=n;j++)M[k][j]/=v;
    for(let i=0;i<n;i++)if(i!==k){const f=M[i][k];for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}
  }
  return M.map(r=>r[n]);
}
// Symmetric Jacobi eigensolver. Columns of V are eigenvectors; sorted output stores vectors as rows.
export function symmetricEigen(matrix,tolerance=1e-13,maxSweeps=100) {
  const n=matrix.length,A=matrix.map(r=>[...r]),V=zero(n,n);for(let i=0;i<n;i++)V[i][i]=1;
  let sweeps=0,maxOffDiagonal=0;
  for(;sweeps<maxSweeps;sweeps++){
    maxOffDiagonal=0;
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
      maxOffDiagonal=Math.max(maxOffDiagonal,Math.abs(A[p][q]));if(Math.abs(A[p][q])<tolerance)continue;
      const angle=.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(angle),s=Math.sin(angle),ap=A[p][p],aq=A[q][q],off=A[p][q];
      for(let k=0;k<n;k++)if(k!==p&&k!==q){const a=A[k][p],b=A[k][q];A[k][p]=A[p][k]=c*a-s*b;A[k][q]=A[q][k]=s*a+c*b;}
      A[p][p]=c*c*ap-2*s*c*off+s*s*aq;A[q][q]=s*s*ap+2*s*c*off+c*c*aq;A[p][q]=A[q][p]=0;
      for(let k=0;k<n;k++){const a=V[k][p],b=V[k][q];V[k][p]=c*a-s*b;V[k][q]=s*a+c*b;}
    }
    if(maxOffDiagonal<tolerance)break;
  }
  const order=Array.from({length:n},(_,i)=>i).sort((a,b)=>A[b][b]-A[a][a]);
  const eigenvectors=order.map(i=>{const v=V.map(r=>r[i]),k=v.reduce((best,x,j)=>Math.abs(x)>Math.abs(v[best])?j:best,0);return scale(v,v[k]<0?-1:1);});
  return {eigenvalues:order.map(i=>A[i][i]),eigenvectors,sweeps,converged:maxOffDiagonal<tolerance,maxOffDiagonal};
}
export function numericalConstraintRank(q,reference=GRASP_REFERENCE,epsilon=1e-6) {
  const J=jacobian(x=>relativeGraspError(x,reference).residual,q,epsilon),values=symmetricEigen(mm(J,transpose(J))).eigenvalues;
  const singularValues=values.map(v=>Math.sqrt(Math.max(0,v))),threshold=Math.max(1e-8,singularValues[0]*1e-6),rank=singularValues.filter(s=>s>threshold).length;
  const reliable=relativeGraspError(q,reference).angularResidualRad<Math.PI-.001;
  return {rank:reliable?rank:null,localDimension:reliable?14-rank:null,singularValues,threshold,epsilon,jacobian:J,reliable,reason:reliable?'regular-log-chart':'orientation-log-branch-cut',rowConvention:'translation in metres; rotation in radians (unscaled)'};
}
function boundedSolve(initial,fn,accepted,maxIterations,localLimits) {
  let q=initial.map((v,i)=>clamp(v,...localLimits[i])),limitHit=q.some((v,i)=>v!==initial[i]),iterations=0,reason='iteration-limit';
  for(;iterations<maxIterations;iterations++){
    const e=fn(q);if(accepted(e)){reason='converged';break;}
    const J=jacobian(fn,q),JT=transpose(J),A=mm(J,JT);for(let i=0;i<A.length;i++)A[i][i]+=1e-7;
    const y=linearSolve(A,scale(e,-1));if(!y){reason='singular';break;}
    let step=mv(JT,y),largest=Math.max(...step.map(Math.abs));if(largest>.18)step=scale(step,.18/largest);
    let improved=false;
    for(let back=0;back<9;back++){
      const raw=add(q,scale(step,2**-back)),candidate=raw.map((v,i)=>clamp(v,...localLimits[i]));
      limitHit ||= raw.some((v,i)=>v!==candidate[i]);
      if(norm(fn(candidate))<norm(e)-1e-12){q=candidate;improved=true;break;}
    }
    if(!improved){reason=limitHit?'joint-limit':'stalled';break;}
  }
  const residual=fn(q),ok=accepted(residual);if(ok)reason='converged';
  return {ok,q,iterations,limitHit,reason,residual};
}
function armSolve(side,target,initial,posture,maxIterations) {
  // The seventh equation selects one local elbow-swivel preference, not another grasp constraint.
  const fn=q=>[...poseResidual(armFK(q,side),target),.12*(q[2]-posture)];
  const accept=e=>norm(e.slice(0,3))<RIG.tolerances.positionM&&norm(e.slice(3,6))<RIG.tolerances.angleRad&&Math.abs(e[6])<.001;
  return boundedSolve(initial,fn,accept,maxIterations,limits);
}
export function solveGrasp({ballPose=RIG.canonicalBallPose,leftPosture=0,rightPosture=0,initialQ=RIG.initialQ,maxIterations=RIG.tolerances.maxIterations}={}) {
  const failure=reason=>({ok:false,q:finite(initialQ)&&initialQ.length===14?[...initialQ]:[...RIG.initialQ],iterations:0,limitHit:false,reason,positionResidualM:null,angularResidualRad:null});
  if(!validPose(ballPose)||!finite(initialQ)||initialQ.length!==14||!Number.isFinite(leftPosture)||!Number.isFinite(rightPosture)||!Number.isInteger(maxIterations)||maxIterations<0||maxIterations>500)return failure('invalid-input');
  const targets={left:composePose(ballPose,RIG.contactFrames.left),right:composePose(ballPose,RIG.contactFrames.right)};
  if(['left','right'].some(side=>norm(sub(targets[side].position,RIG.shoulders[side]))>RIG.upperArm+RIG.forearm+1e-8))return failure('unreachable');
  const left=armSolve('left',targets.left,initialQ.slice(0,7),leftPosture,maxIterations),right=armSolve('right',targets.right,initialQ.slice(7),rightPosture,maxIterations),q=[...left.q,...right.q],fk=forwardKinematics(q);
  const errors=['left','right'].map(s=>poseResidual(fk[s],targets[s]));
  const positionResidualM=Math.max(...errors.map(e=>norm(e.slice(0,3)))),angularResidualRad=Math.max(...errors.map(e=>norm(e.slice(3))));
  const grasp=relativeGraspError(q),jumpRmsRad=norm(sub(q,initialQ))/Math.sqrt(14);
  const ok=left.ok&&right.ok&&withinJointLimits(q)&&grasp.positionResidualM<.0001&&grasp.angularResidualRad<.0002;
  return {ok,q,iterations:Math.max(left.iterations,right.iterations),limitHit:left.limitHit||right.limitHit,reason:ok?'converged':(!left.ok?left.reason:right.reason),positionResidualM,angularResidualRad,grasp,jumpRmsRad,discontinuous:jumpRmsRad>.65};
}
export function ballPoseFromConfiguration(q) {return composePose(forwardKinematics(q).left,inversePose(RIG.contactFrames.left));}
export function projectDecodedConfiguration(qCandidate,reference=GRASP_REFERENCE) {
  if(!finite(qCandidate)||qCandidate.length!==14)return {ok:false,reason:'invalid-input',q:null,iterations:0};
  const before=relativeGraspError(qCandidate,reference);
  const result=boundedSolve(qCandidate,q=>relativeGraspError(q,reference).residual,e=>norm(e.slice(0,3))<.00002&&norm(e.slice(3))<.00004,100,[...limits,...limits]);
  const after=relativeGraspError(result.q,reference);
  return {...result,positionResidualM:after.positionResidualM,angularResidualRad:after.angularResidualRad,before,after,correctionRmsRad:norm(sub(result.q,qCandidate))/Math.sqrt(14),ballPose:ballPoseFromConfiguration(result.q)};
}
export function ballLoadTorqueEstimate(q,{mass=RIG.ballMass,gravity=RIG.gravity}={}) {
  if(!Number.isFinite(mass)||mass<0||!Number.isFinite(gravity)||gravity<0)throw new RangeError('Nonnegative mass and gravity required');
  const fk=forwardKinematics(q),force=[0,0,mass*gravity/2],tau=[];
  for(const side of ['left','right'])for(const j of fk[side].joints)tau.push(dot(cross(j.axis,sub(fk[side].position,j.position)),force));
  return {tau,normNm:norm(tau),mass,gravity,forcePerHand:force,units:'N·m'};
}
// Minimum distance between two finite centerline segments (not infinite lines).
function segmentDistance(p1,q1,p2,q2){
  const d1=sub(q1,p1),d2=sub(q2,p2),r=sub(p1,p2),a=dot(d1,d1),e=dot(d2,d2),f=dot(d2,r);let s=0,t=0;
  if(a<1e-15&&e<1e-15)return norm(r);
  if(a<1e-15)t=clamp(f/e,0,1);else{const c=dot(d1,r);if(e<1e-15)s=clamp(-c/a,0,1);else{const b=dot(d1,d2),den=a*e-b*b;s=den>1e-15?clamp((b*f-c*e)/den,0,1):0;t=(b*s+f)/e;if(t<0){t=0;s=clamp(-c/a,0,1);}else if(t>1){t=1;s=clamp((b-c)/a,0,1);}}}
  return norm(sub(add(p1,scale(d1,s)),add(p2,scale(d2,t))));
}
export function collisionClearance(q) {
  const fk=forwardKinematics(q),L=fk.left.points,R=fk.right.points;let d=Infinity;
  for(let i=0;i<2;i++)for(let j=0;j<2;j++)d=Math.min(d,segmentDistance(L[i],L[i+1],R[j],R[j+1])-2*RIG.linkRadius);
  // Fixed torso proxy is a vertical capsule. Shoulder-adjacent upper links excluded intentionally.
  const torso=[[0,0,.8],[0,0,1.28]],radius=.13;
  for(const arm of [L,R])d=Math.min(d,segmentDistance(arm[1],arm[2],...torso)-RIG.linkRadius-radius);
  return {clearanceM:d,proxy:'arm-arm capsules and forearm-torso capsule; excludes shoulder mounts, paws and ball'};
}
export function costAt(q,params={}) {
  const torque=ballLoadTorqueEstimate(q,params),clearance=collisionClearance(q).clearanceM;
  const jointLimit=Math.max(...q.map((v,i)=>{const [lo,hi]=limits[i%7];return Math.abs(2*(v-lo)/(hi-lo)-1);}));
  return {torque:torque.normNm,jointLimit,clearance,clearanceCost:1/(1+Math.max(0,clearance)/.05),units:{torque:'N·m',jointLimit:'normalized index',clearance:'m',clearanceCost:'normalized index'}};
}
export function encodeTaskCoordinates(reach,present) {
  if(!Number.isFinite(reach)||!Number.isFinite(present)||Math.abs(reach)>1||Math.abs(present)>1)throw new RangeError('Task coordinates in [-1,1]');
  return {ballPose:poseFromCoordinates(.015*present,.43+.065*reach,1.2+.055*present,.06*present,.07*present,.08*present),leftPosture:.12*present,rightPosture:-.12*present};
}
export function fitPCA(samples) {
  if(!Array.isArray(samples)||samples.length<2)throw new RangeError('At least two training samples required');samples.forEach(q=>checkQ(q));
  const mean=Array(14).fill(0);for(const q of samples)q.forEach((v,i)=>mean[i]+=v/samples.length);
  const covariance=zero(14,14);for(const q of samples){const d=sub(q,mean);for(let i=0;i<14;i++)for(let j=0;j<14;j++)covariance[i][j]+=d[i]*d[j]/(samples.length-1);}
  const eig=symmetricEigen(covariance);return {mean,eigenvalues:eig.eigenvalues.map(x=>Math.max(0,x)),eigenvectors:eig.eigenvectors,trainingCount:samples.length,fitDiagnostics:{method:'sample covariance + symmetric Jacobi rotations',...eig,eigenvectors:undefined,eigenvalues:undefined}};
}
export function encodePCA(q,pca,rank=2){checkQ(q);return pca.eigenvectors.slice(0,rank).map(v=>dot(v,sub(q,pca.mean)));}
export function decodePCA(z,pca){if(!finite(z)||z.length>14)throw new RangeError('Finite PCA coordinates required');return pca.mean.map((v,j)=>v+z.reduce((s,x,i)=>s+x*pca.eigenvectors[i][j],0));}
export function reconstructAndScore(samples,pca,rank=2) {
  if(!Number.isInteger(rank)||rank<0||rank>14||samples.length===0)throw new RangeError('Invalid rank or empty samples');
  let squared=0;for(const q of samples){const d=sub(q,decodePCA(encodePCA(q,pca,rank),pca));squared+=dot(d,d);}
  const total=pca.eigenvalues.reduce((a,b)=>a+b,0);return {rmsRad:Math.sqrt(squared/(14*samples.length)),rmsDeg:Math.sqrt(squared/(14*samples.length))*180/Math.PI,explainedVariance:total?pca.eigenvalues.slice(0,rank).reduce((a,b)=>a+b,0)/total:1,count:samples.length,rank};
}
// Four-neighbour Dijkstra. Objective = edge length * (1 + weight * mean normalized field).
export function planGridPath(grid,start,goal,{field='torque',weight=4,shortest=false}={}) {
  const {width,height,cells}=grid;
  const failure=reason=>({ok:false,reason,path:[],cost:null,distanceM:null});
  if(!Number.isInteger(start)||!Number.isInteger(goal)||start<0||goal<0||start>=cells.length||goal>=cells.length||!cells[start].valid||!cells[goal].valid)return failure('invalid-endpoint');
  if(!['torque','jointLimit','clearanceCost'].includes(field)||!Number.isFinite(weight)||weight<0)return failure('invalid-options');
  const max=Math.max(...cells.filter(c=>c.valid).map(c=>c.costs[field]),1e-12),dist=Array(cells.length).fill(Infinity),prev=Array(cells.length).fill(-1),done=new Set();dist[start]=0;
  while(true){let u=-1;for(let i=0;i<cells.length;i++)if(!done.has(i)&&(u<0||dist[i]<dist[u]))u=i;if(u<0||!Number.isFinite(dist[u]))break;if(u===goal)break;done.add(u);
    const x=u%width,y=Math.floor(u/width),neighbors=[];if(x>0)neighbors.push(u-1);if(x<width-1)neighbors.push(u+1);if(y>0)neighbors.push(u-width);if(y<height-1)neighbors.push(u+width);
    for(const v of neighbors){if(!cells[v].valid||done.has(v))continue;const length=norm(sub(cells[u].ballPose.position,cells[v].ballPose.position)),cost=length*(1+(shortest?0:weight)*(cells[u].costs[field]+cells[v].costs[field])/(2*max));if(dist[u]+cost<dist[v]-1e-14){dist[v]=dist[u]+cost;prev[v]=u;}}
  }
  if(!Number.isFinite(dist[goal]))return failure('no-route');const path=[];for(let u=goal;u!==-1;u=prev[u])path.push(u);path.reverse();let distanceM=0;for(let i=1;i<path.length;i++)distanceM+=norm(sub(cells[path[i]].ballPose.position,cells[path[i-1]].ballPose.position));
  return {ok:true,reason:'converged',path,cost:dist[goal],distanceM,field,weight:shortest?0:weight,normalizationMax:max,objective:'four-neighbour edge metres × (1 + weight × mean field / maximum valid field)'};
}
