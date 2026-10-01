import {
  targetVariance,
  validateModel,
} from "./covariance.js";
import { checkObservations } from "./dataset.js";
import { solveLinear, type SolveResult } from "./matrix.js";
import { effectiveNoiseVariances, prepareObservations } from "./prepare.js";
import { LIMITS, type CovarianceModel, type Observation, type Point } from "./types.js";

export interface PointPrediction {
  target: Point;
  predictWhat: "latent" | "observation";
  weights: { index: number; label: string; weight: number }[];
  /** Weights on effective (de-duplicated / merged) observation nodes. */
  effectiveWeights: { index: number; label: string; weight: number }[];
  weightSum: number;
  /** Lagrange multiplier mu with convention C w + mu*1 = k. */
  lagrangeMultiplier: number;
  prediction: number;
  /** Raw variance straight from the algebra (may be slightly negative). */
  varianceRaw: number;
  /** Reported variance; tiny negative round-off is clamped to 0. */
  variance: number;
  varianceStdDev: number;
  varianceNote: "ok" | "clamped_tiny_negative" | "negative_refused";
  health: "ok" | "ill_conditioned" | "singular";
  equationResidualInf: number;
  equationResidualRel: number;
  /** |sum_i w_i - 1|, the unbiasedness constraint violation. */
  constraintResidual: number;
  condition1Estimate: number;
  rank: number;
  messages: string[];
  preparationNotes: string[];
}

export interface KrigingInput {
  observations: Observation[];
  model: CovarianceModel;
  /**
   * Optional per-observation measurement noise variances (length n), used for
   * merged duplicate locations. Defaults to model.measurementNoiseVariance.
   */
  noiseVariances?: number[];
}

function validateInput(input: KrigingInput): void {
  const errs = validateModel(input.model);
  const check = checkObservations(
    input.observations,
    input.model.measurementNoiseVariance,
  );
  errs.push(...check.errors);
  if (errs.length) throw new Error(`输入不合法：\n - ${errs.join("\n - ")}`);
}

/** Build the observation block C_ij = Cov(z_i, z_j) (nugget on diagonal). */
function buildObsBlock(
  obs: Observation[],
  model: CovarianceModel,
  noiseVariances: number[],
): number[][] {
  const n = obs.length;
  const C = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      // Field covariance (no nugget) + effective noise only on the diagonal.
      let c = covariancePair(obs[i]!, obs[j]!, model);
      if (i === j) c += noiseVariances[i]!;
      C[i]![j] = c;
      C[j]![i] = c;
    }
  }
  return C;
}

function covariancePair(a: Point, b: Point, model: CovarianceModel): number {
  // local import-free distance helper
  return model.sill * Math.exp(-Math.hypot(a.x - b.x, a.y - b.y) / model.range);
}

/** Assemble the augmented ordinary-kriging KKT system for one target. */
function buildSystem(
  obs: Observation[],
  target: Point,
  model: CovarianceModel,
  noiseVariances: number[],
): { A: number[][]; b: number[] } {
  const n = obs.length;
  const C = buildObsBlock(obs, model, noiseVariances);
  const k = obs.map((o) => covariancePair(target, o, model));

  const A = Array.from({ length: n + 1 }, () => new Array<number>(n + 1).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i]![j] = C[i]![j]!;
    // Standard symmetric ordinary-kriging KKT block:
    //   [[C,   1], [1^T, 0]] [w; mu] = [k; 1]
    // i.e. C w + mu*1 = k, 1^T w = 1. Here mu equals MINUS the conventional
    // non-negative Lagrange multiplier, so mu <= 0 in interior solutions.
    A[i]![n] = 1;
    A[n]![i] = 1;
  }
  A[n]![n] = 0;
  const b = [...k, 1];
  return { A, b };
}

function classifyHealth(sol: SolveResult, n: number): PointPrediction["health"] {
  if (sol.singular || sol.rank < n) return "singular";
  if (sol.cond1Estimate > LIMITS.illConditionedCond) return "ill_conditioned";
  return "ok";
}

export function predictPoint(
  input: KrigingInput,
  target: Point,
  predictWhat: "latent" | "observation" = "latent",
): PointPrediction {
  validateInput(input);
  if (!Number.isFinite(target.x) || !Number.isFinite(target.y)) {
    throw new Error("预测目标坐标必须为有限数");
  }
  const { model } = input;
  const prepared = prepareObservations(input.observations, model);
  const obs = prepared.effective;
  const noiseVariances =
    input.noiseVariances ??
    effectiveNoiseVariances(prepared.members, model.measurementNoiseVariance);
  const n = obs.length;
  const { A, b } = buildSystem(obs, target, model, noiseVariances);
  const sol = solveLinear(A, b);

  const w = sol.x.slice(0, n);
  const mu = sol.x[n]!;
  const effectiveWeights = w.map((weight, index) => ({
    index,
    label: obs[index]!.label ?? `#${index}`,
    weight,
  }));

  // Expand effective-node weights back onto the ORIGINAL observations:
  // a merged effective reading represents its members; the effective reading
  // is an inverse-variance weighted average, so each member's contribution to
  // the predictor is w_eff * (v_i contribution fraction). We report each
  // member's share of the effective weight using equal split for identical
  // zero-noise duplicates and inverse-variance fractions for noisy merges.
  const origWeightMap = new Map<number, number>();
  prepared.members.forEach((memberIndices, effIdx) => {
    const weff = w[effIdx]!;
    if (memberIndices.length === 1) {
      origWeightMap.set(memberIndices[0]!, weff);
      return;
    }
    if (model.measurementNoiseVariance === 0) {
      for (const oi of memberIndices)
        origWeightMap.set(oi, weff / memberIndices.length);
    } else {
      const t2 = model.measurementNoiseVariance;
      const fracDen = memberIndices.reduce(
        (s, oi) => s + 1 / t2,
        0,
      );
      for (const oi of memberIndices)
        origWeightMap.set(oi, (weff * 1 / t2) / fracDen);
    }
  });
  const weights = input.observations.map((o, index) => ({
    index,
    label: o.label ?? `#${index}`,
    weight: origWeightMap.get(index) ?? 0,
  }));

  const prediction = w.reduce((s, wi, i) => s + wi * obs[i]!.value, 0);
  // k occupies the first n RHS entries; b[n] = 1 is the constraint and must
  // never enter w^T k.
  const kVec = b.slice(0, n);
  const wk = w.reduce((s, wi, i) => s + wi * kVec[i]!, 0);
  const c00 = targetVariance(model, predictWhat);
  // Ordinary kriging variance: Var = c00 - w^T k - mu.
  // Checks (mu <= 0 typically):
  //  - single noiseless point, distance h: mu = C(h) - sill, Var = sill-C(h)
  //  - single noisy point predicted at the SAME spot: w = 1, mu = -nugget,
  //    latent Var = nugget; predicting a NEW observation adds its own nugget
  //    so the measured-vs-measured MSE = 2*nugget.
  const varianceRaw = c00 - wk - mu;

  let variance = varianceRaw;
  let varianceNote: PointPrediction["varianceNote"] = "ok";
  if (varianceRaw < 0 && varianceRaw > -LIMITS.tinyNegativeVarianceTol) {
    variance = 0;
    varianceNote = "clamped_tiny_negative";
  } else if (varianceRaw <= -LIMITS.tinyNegativeVarianceTol) {
    varianceNote = "negative_refused";
  }

  const weightSum = w.reduce((a, z) => a + z, 0);
  const health = classifyHealth(sol, n + 1);
  const messages: string[] = [];
  if (health === "singular")
    messages.push(
      "克里金方程组奇异（秩不足）：结果不可靠，已如实报告，未添加任何未声明对角项",
    );
  if (health === "ill_conditioned")
    messages.push(
      `方程组严重病态（条件数约 ${sol.cond1Estimate.toExponential(2)}），权重与方差可能不稳定`,
    );
  if (varianceNote === "negative_refused")
    messages.push(
      `方差为实质性负值 ${varianceRaw.toExponential(3)}（超过舍入容差），不应作为有效不确定性使用`,
    );
  if (varianceNote === "clamped_tiny_negative")
    messages.push("方差出现舍入量级微小负值，已夹到 0");

  return {
    target,
    predictWhat,
    weights,
    effectiveWeights,
    weightSum,
    lagrangeMultiplier: mu,
    prediction,
    varianceRaw,
    variance: varianceNote === "negative_refused" ? NaN : variance,
    varianceStdDev:
      varianceNote === "negative_refused" || variance < 0 ? NaN : Math.sqrt(variance),
    varianceNote,
    health,
    equationResidualInf: sol.residualInf,
    equationResidualRel: sol.residualRel,
    constraintResidual: Math.abs(weightSum - 1),
    condition1Estimate: sol.cond1Estimate,
    rank: sol.rank,
    messages,
    preparationNotes: prepared.notes,
  };
}

export interface GridSpec {
  xMin: number; xMax: number;
  yMin: number; yMax: number;
  nx: number;
  ny: number;
}

export function gridPoints(spec: GridSpec): Point[] {
  for (const [name, v] of Object.entries(spec)) {
    if (!Number.isFinite(v)) throw new Error(`网格参数 ${name} 必须为有限数`);
  }
  if (spec.nx < 1 || spec.ny < 1 || !Number.isInteger(spec.nx) || !Number.isInteger(spec.ny))
    throw new Error("nx/ny 必须为 >= 1 的整数");
  if (spec.xMax < spec.xMin || spec.yMax < spec.yMin)
    throw new Error("网格范围非法：max 不能小于 min");
  const total = spec.nx * spec.ny;
  if (total > LIMITS.maxGridNodes)
    throw new Error(`网格节点数 ${total} 超过上限 ${LIMITS.maxGridNodes}`);
  const pts: Point[] = [];
  const dx = spec.nx === 1 ? 0 : (spec.xMax - spec.xMin) / (spec.nx - 1);
  const dy = spec.ny === 1 ? 0 : (spec.yMax - spec.yMin) / (spec.ny - 1);
  for (let iy = 0; iy < spec.ny; iy++)
    for (let ix = 0; ix < spec.nx; ix++)
      pts.push({ x: spec.xMin + ix * dx, y: spec.yMin + iy * dy });
  return pts;
}

export function predictGrid(
  input: KrigingInput,
  spec: GridSpec,
  predictWhat: "latent" | "observation" = "latent",
): { nodes: PointPrediction[]; illCount: number; singularCount: number } {
  const pts = gridPoints(spec);
  const nodes = pts.map((p) => predictPoint(input, p, predictWhat));
  return {
    nodes,
    illCount: nodes.filter((r) => r.health !== "ok").length,
    singularCount: nodes.filter((r) => r.health === "singular").length,
  };
}

export interface LeaveOneOutResult {
  index: number;
  label: string;
  heldOut: Observation;
  /** Predict what the held-out MEASUREMENT should be (includes noise variance). */
  prediction: number;
  variance: number;
  stdError: number;
  /** observed - predicted */
  error: number;
  standardizedResidual: number;
  health: "ok" | "ill_conditioned" | "singular";
  constraintResidual: number;
  equationResidualRel: number;
}

export interface LeaveOneOutSummary {
  count: number;
  meanError: number;
  meanAbsoluteError: number;
  rootMeanSquareError: number;
  /** Mean squared standardized residual; ideally ~1 when model is consistent. */
  meanSquaredStandardizedResidual: number;
  /** Mean standardized residual; ideally ~0. */
  meanStandardizedResidual: number;
  /** Fraction of held-out values within prediction +/- 1.96 std. */
  coverage95: number;
  anyUnreliable: boolean;
}

export interface LeaveOneOutOutput {
  feasible: boolean;
  reason?: string;
  folds: LeaveOneOutResult[];
  summary?: LeaveOneOutSummary;
}

export function leaveOneOut(input: KrigingInput): LeaveOneOutOutput {
  validateInput(input);
  const { observations: obs, model } = input;
  if (obs.length < 2) {
    return {
      feasible: false,
      reason: "至少需要 2 个测点才能进行逐点留一（只剩 0 个训练点时无法估计）",
      folds: [],
    };
  }

  const folds: LeaveOneOutResult[] = [];
  for (let i = 0; i < obs.length; i++) {
    const training = obs.filter((_, j) => j !== i);
    const sub: KrigingInput = { observations: training, model };
    // Compare against the actual noisy measurement -> "observation" target,
    // so the predictive variance includes the new-observation noise.
    const r = predictPoint(sub, { x: obs[i]!.x, y: obs[i]!.y }, "observation");
    const error = obs[i]!.value - r.prediction;
    folds.push({
      index: i,
      label: obs[i]!.label ?? `#${i}`,
      heldOut: obs[i]!,
      prediction: r.prediction,
      variance: r.variance,
      stdError: r.varianceStdDev,
      error,
      standardizedResidual: Number.isFinite(r.varianceStdDev)
        ? error / r.varianceStdDev
        : NaN,
      health: r.health,
      constraintResidual: r.constraintResidual,
      equationResidualRel: r.equationResidualRel,
    });
  }

  const n = folds.length;
  const err = folds.map((f) => f.error);
  const sr = folds.map((f) => f.standardizedResidual).filter(Number.isFinite);
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const mse = mean(err.map((e) => e * e));
  const cover =
    folds.filter((f) => Number.isFinite(f.stdError) && Math.abs(f.error) <= 1.96 * f.stdError)
      .length / n;

  return {
    feasible: true,
    folds,
    summary: {
      count: n,
      meanError: mean(err),
      meanAbsoluteError: mean(err.map(Math.abs)),
      rootMeanSquareError: Math.sqrt(mse),
      meanSquaredStandardizedResidual:
        sr.length === n ? mean(sr.map((v) => v * v)) : NaN,
      meanStandardizedResidual: sr.length === n ? mean(sr) : NaN,
      coverage95: cover,
      anyUnreliable: folds.some((f) => f.health !== "ok"),
    },
  };
}
