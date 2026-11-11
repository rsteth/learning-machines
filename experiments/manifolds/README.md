# 03 / manifolds and motion

**How many dimensions does a movement need?** A deterministic two-arm volleyball teaching model. Mechanics and reproducible evidence live here; the companion website owns the essay, visual renderer and controls. This is neither a physical-robot benchmark nor a trained VLA, dynamics engine, safety assessment or learned neural controller.

## Reproduce

Requires Node.js 18+, built-in modules only, no network, models or dependencies.

```sh
npm test
npm run experiment:manifolds
npm run fixtures:manifolds
# Independently check byte identity without overwriting the committed fixture:
node experiments/manifolds/generate-fixtures.mjs /tmp/manifolds-fixtures.mjs
cmp experiments/manifolds/fixtures.mjs /tmp/manifolds-fixtures.mjs
```

`generateFixtures(308141)` uses a fixed 32-bit linear-congruential seed. It generates the grid, trajectory-level train/holdout split, actual sample covariance, full Jacobi eigendecomposition, diagnostics and examples. `fixtures.mjs` is generated evidence, not manually curated outputs. The tests compare complete regenerated source bytes, recompute every valid grid cell's costs, verify every hand target and check the training split. Floating-point bytes were verified on Node v24.22.1/Linux; the algebra is deterministic within that runtime, not a promise of byte-identical transcendental functions across every JS engine. Fixture runtime reads do not recompute the training or grid.

## Rig and units

The world is right-handed: x left/right, y forward, z up. Lengths are metres, angles radians (UI degrees), torque N·m. Torso fixed, shoulder anchors `[-.23,0,1.25]` and `[.23,0,1.25]`. Upper arm .32m, forearm .29m, ball radius .11m. Canonical ball centre `[0,.43,1.2]`, rotation identity. The two fixed contact transforms are `[-.11,0,0]` and `[.11,0,0]`, both with identity rotation. Opposing paw contact normals are mounted along local +x for the left paw and −x for the right: identical frame rotations do not mean identical facing surfaces.

Each arm has **seven revolute coordinates** with local axes, in order: z, x, y (shoulder); x (elbow); z, x, y (wrist). Apply each rotation in the current local frame; translate .32m along local +y after shoulder joint 3, and .29m after elbow joint 4. Wrist axes intersect at the hand origin and determine full 3D paw orientation. This is a genuine spatial serial chain, not a planar arm carrying decorative coordinates. `forwardKinematics` returns each joint's world-space origin/axis, the shoulder/elbow/wrist polyline, and full hand pose. It intentionally accepts finite angles outside limits for mathematical diagnostics; accepted IK states enforce limits.

Per-arm joint limits in radians:

| Joint | Minimum | Maximum |
|---|---:|---:|
| Shoulder yaw | −2.6 | 2.6 |
| Shoulder lift | −1.8 | 1.8 |
| Shoulder swivel | −2.4 | 2.4 |
| Elbow flex | .08 | 2.7 |
| Wrist yaw | −2.9 | 2.9 |
| Wrist pitch | −2.3 | 2.3 |
| Wrist roll | −2.9 | 2.9 |

`RIG.initialQ` is an IK seed, **not** the exact canonical grasp. Use `FIXTURES.canonical.q` for an already solved reset state. The robot's left/right labels refer to named serial chains at negative/positive world x, respectively.

## Six constraints and local dimension

`relativeGraspError(q)` computes `T_L^-1 T_R` versus a fixed translation `[.22,0,0]` and identity orientation. The six residual entries are three local translation differences (m), then the principal SO(3) logarithm of the orientation error (rad). It is not a position-only closure. The rotation logarithm has small-angle and stable half-turn branches; its sign at exactly π is nonunique, as it must be.

`numericalConstraintRank` differentiates these six residuals by central finite differences, ε=1e−6 rad. Singular values are square roots of eigenvalues of `J Jᵀ`; count singular values above `max(1e−8, 1e−6 σmax)`. Translation rows remain in metres and rotation rows in radians, without rescaling. This convention affects conditioning/tolerance, not the exact algebraic rank; it is explicit in the returned diagnostics. Within .001rad of the log's π branch cut, report `reliable:false`, `rank:null` rather than pretending a nonsmooth chart has a reliable rank.

At the canonical regular state the six singular values are approximately 2.529, 2.466, 1.541, .566, .432, .373 under that convention. Thus the local implicit-function count is **14 − 6 = 8**. This does not establish global coordinates or universal rank six. The tests include an extended collinear singular configuration outside the permitted elbow range with rank five. An IK target fixes the ball's six pose variables and solves twelve hand-pose equations plus two optional posture preferences; do not subtract twelve again when explaining the movable-ball family.

## Bounded IK, projection and honest failure

`solveGrasp({ballPose,leftPosture,rightPosture,initialQ,maxIterations})` independently solves each hand target `T_ball T_contact` with damped least squares and finite-difference derivatives. A seventh weakly scaled residual `.12*(shoulderSwivel − preference)` selects a redundant posture. Solve the damped residual-space normal equations (`1e−7 I`), cap each step to .18rad, clamp real joint bounds, and backtrack up to nine times for actual residual decrease. Default cap 100 iterations (caller cap ≤500).

Each accepted hand has translation error <.00002m and angular error <.00004rad. Re-evaluate FK and full relative closure after solving; accept only if both arms converge, all joints remain bounded and grasp residual <.0001m / .0002rad. Returned `ok`, `reason`, `iterations`, `limitHit`, target residuals and actual `q` distinguish converged, unreachable, invalid-input, singular, joint-limit, stalled and iteration-limit outcomes. A rejected result can contain a finite diagnostic attempted `q`, **not a valid pose**. A UI must retain its last valid pose and expose failure. A geometric reach precheck rejects targets beyond combined link length but does not claim every closer point is reachable.

Warm-start using the last accepted q. `jumpRmsRad` and `discontinuous` flag jumps above .65rad RMS, even if geometrically valid: the caller should require an explicit reset/transition instead of animating as though this were a continuous trajectory. Solver convergence does not check swept collision, dynamics or time continuity. The live controls offer a bounded teaching region; they are not a globally reliable manipulation planner.

`projectDecodedConfiguration` uses the same bounded procedure on the **six relative-grasp residuals**, leaving object pose free. Its damped minimum-norm corrections locally reduce violation but are not certified global Euclidean nearest-point projections. It returns before/after residuals, correction RMS, ball pose inferred from the left contact, and explicit failure. Projection can alter reconstruction error and object pose. PCA decoding outside joint bounds is allowed as a raw candidate; only a successful bounded projection should be displayed as an accepted grasp.

## Cost fields and graph routes

`ballLoadTorqueEstimate` assumes a .27kg volleyball, g=9.81m/s² and equal upward support `[0,0,mg/2]` at each hand. For each joint `Jv_i = axis_i × (hand − jointOrigin)`, calculate `tau_i = Jv_i · F`. The displayed scalar is the norm over fourteen joints, in N·m. This is the signed actuator contribution needed to oppose the ball's downward load, **not full robot torque or energy**. Excludes arm weight, contact moments, squeezing, friction, inertia, acceleration and balance. Independent finite-difference potential-energy tests verify the Jacobian/sign convention and mass scaling.

`jointLimit` is the maximum across joints of `abs(2*(q−min)/(max−min)−1)`: zero at all midranges, one at a boundary. `clearance` is the minimum signed surface gap (m) among all four cross-arm upper/forearm capsule pairs, plus each forearm versus a torso capsule. Arm radius .025m; torso centreline `[0,0,.8]` to `[0,0,1.28]`, radius .13m. The upper-arm/shoulder mount against torso, same-arm neighbouring links, paws and ball are intentionally excluded. `clearanceCost = 1 / (1 + max(0,clearance)/.05)`. It is a geometric proxy, not collision certification.

The deterministic 21×21 grid samples y=.25… .67m and z=.92…1.48m with x=0, orientation identity, posture preferences zero. A cell is valid only if IK converges and the capsule gap is positive. Invalid cells expose reasons and have **null costs**. The fixture has 331 valid cells out of 441; the count is generated and tested, not asserted as an exact reachability boundary. The grid can miss narrow feasible regions and IK can fail at a geometrically reachable point.

`planGridPath(grid,startIndex,goalIndex,{field,weight,shortest})` runs deterministic Dijkstra on valid four-neighbour cells. Edge objective: geometric edge length × `(1 + weight * mean(field)/maxValidField)`. Fields: torque, jointLimit, clearanceCost; weight default4. `shortest:true` sets weight0. This optimizes exactly that discrete nonnegative objective, not a continuous trajectory or accumulated physical energy. Compare the shortest graph route to the least-cost graph route; a straight connection is not implied. Vertices are validated; continuous interpolation between vertices is **not** certified. Each displayed pose should use its actual grid q rather than interpolate joints and imply closure. Tests compare objective values, blocked graphs and graph adjacency.

## Designed task versus fitted representation

`encodeTaskCoordinates(reach,present)` accepts each in [−1,1]. It maps to ball x=.015 present, y=.43+.065 reach, z=1.2+.055 present; roll=.06 present, pitch=.07 present, yaw=.08 present; left/right swivel preferences ±.12 present. These are **hand-designed controls**, not learned semantic PCA axes.

Twenty-four synthetic trajectories each contain fifteen poses. Seeded amplitude, extent, presentation offset and a small lateral perturbation vary the family. Every fourth whole trajectory is held out: **270 training and 90 holdout poses**, with no holdout data in mean/covariance/eigensolver. The sample covariance uses divisor n−1. Symmetric Jacobi rotations fit all fourteen eigenvectors; deterministic sorting and sign selection stabilize outputs. PCA retains raw joint radians, with no arbitrary per-joint standardization or wrapping. This bounded family stays away from angle wrap boundaries; a broad cyclic joint dataset would need a different treatment.

PCA eigenvectors have arbitrary signs. The generic `fitPCA`/Jacobi rule remains unchanged (largest-magnitude joint loading positive). Fixture generator version 2 adds a **display sign convention**: PC1 is oriented to positive covariance between its training scores and physical ball y (forward distance), and PC2 to positive covariance with physical ball z (height). Only training samples enter this choice. If absolute Pearson correlation is ≤1e−10, including a constant reference, preserve the eigensolver sign deterministically. `pca.orientation` records each physical reference, before/after covariance and correlation, applied sign, threshold, fallback and training count.

For this dataset PC1 is flipped (forward-distance correlation becomes +.999385); PC2 is unchanged (height correlation +.996272). Positive PC1 therefore tends away from the torso and positive PC2 tends higher in this displayed family. These are sign conventions and observed correlations, **not semantic identities or disentangled axes**: do not rename PC1 “Reach” or PC2 “Present,” and do not promise these directions for another dataset or arbitrary extrapolation. Regenerating scores, ranges and examples after the vector flip leaves every decoded training/holdout reconstruction bit-identical when paired with its newly signed encoding, and leaves covariance, eigenvalues, subspace, physical samples, torque/clearance fields and reconstruction metrics unchanged. The displayed signs make no change to the world coordinate frame or camera.

`encodePCA` projects centred q onto retained eigenvectors; `decodePCA` adds their weighted sum to the training mean. Coordinate ranges are the observed training score extrema, not a validity promise for the rectangle between them. `reconstructAndScore` reports joint-coordinate RMS over samples × fourteen dimensions. `explainedVariance` always refers to the **training covariance**, including when paired with holdout error; it is not holdout R².

Generated evidence: two PCs explain approximately 99.674% training variance; holdout RMS ≈.209° before projection and ≈.129° after projection, with 90/90 projections converged. The illustrated outside-family sideward presentation is reachable but has ≈8.91° reconstruction RMS. Exact values and failures are in `FIXTURES.scores`, not embedded constants to copy into the site. A good two-dimensional linear fit to this deliberately narrow family does not imply a universal two-dimensional robot, or valid unconstrained decoded poses. Projection preserves closure, not the intended outside-family handoff. No knowledge distillation, neural training, human demonstrations or dynamics are represented.

## Interface and integration

Pure browser-safe mechanics use no DOM or Node modules. See exported names in `mechanics.mjs`; build-time generator/CLI alone import Node APIs. `FIXTURES` contains seed/version/units, canonical solved q, relative attachment frames, full grid cells with solver diagnostics, train/holdout samples, fitted PCA, scores and in-family/outside-family reconstructions. Pose rotations are nested 3×3 row-major arrays, not Euler vectors; use `poseFromCoordinates` for UI ball inputs.

The site should pin **committed** `mechanics.mjs` and `fixtures.mjs` through its existing `scripts/sync-learning-machines.mjs` integrity workflow, preserving previous study pins. The final source commit must be a real revision and any public source link must not claim an unpushed local revision is available on GitHub. Mechanics commits and sync verification do not themselves authorize publishing the article.

## Sources and license

Original rig, code, tests, deterministic synthetic data and `preview.svg` are [MIT licensed](../../LICENSE). No third-party code or robot datasets are copied.

- Lynch & Park, [Configuration Space Representation](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-3-2-configuration-space-representation/) and [Configuration and Velocity Constraints](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-4-configuration-and-velocity-constraints/) for the geometric distinction.
- [Learning latent actions to control assistive robots](https://pmc.ncbi.nlm.nih.gov/articles/PMC8335729/) is related research, not this fixture's source.
- [π₀](https://www.pi.website/blog/pi0) is a VLA example, not an implemented component here.
- [LoRA](https://arxiv.org/abs/2106.09685) concerns low-rank weight updates; it does not establish intrinsic robot or activation dimension.
