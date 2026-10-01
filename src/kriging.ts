import type {
  GridCell,
  GridResult,
  GridSpec,
  KrigingPointResult,
  LooFold,
  LooResult,
  LooSummary,
  ModelParams,
  Point,
  SolveDiagnostics,
  TargetKind,
} from "./types.js";
import { KrigingError } from "./types.js";
import {
  buildRhs,
  covariance,
  covarianceBetween,
  validateModel,
} from "./covariance.js";
import { solveLinear } from "./solver.js";
import { prepareData, validateRaw, type PreparedData } from "./data.js";

/** 实质负方差阈值：相对基台 */
const SUBSTANTIAL_NEG_VAR = -1e-6;
/** 小量负方差钳制阈值（相对基台）：在此之内视为数值误差钳为 0 */
const TINY_NEG_VAR = -1e-9;

type TrainPoint = Point & { sourceIndices: number[]; localNugget: number };

function buildMatrixLocal(points: TrainPoint[], m: ModelParams): number[][] {
  const n = points.length;
  const size = n + 1;
  const A: number[][] = Array.from({ length: size }, () =>
    new Array<number>(size).fill(0),
  );
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i][j] =
        i === j
          ? m.psill + points[i].localNugget
          : covarianceBetween(points[i], points[j], m);
    }
    A[i][n] = 1;
    A[n][i] = 1;
  }
  return A;
}

function toDiagnostics(
  n: number,
  r: ReturnType<typeof solveLinear>,
  weightsSum: number,
): SolveDiagnostics {
  return {
    n,
    reciprocalConditionNumber: r.reciprocalConditionNumber,
    conditionNumberEstimate: r.conditionNumberEstimate,
    rhsResidualNorm: r.residualNormInf,
    rhsResidualMaxAbs: r.residualMaxAbs,
    constraintResidual: weightsSum - 1,
    singular: r.singular,
    illConditioned: r.illConditioned,
    message: r.singular
      ? "克里金系数矩阵奇异，无法得到唯一解（未加入任何未声明对角项）"
      : r.illConditioned
        ? "系统严重病态，结果不可靠"
        : undefined,
  };
}

/** 核心单点求解。target=观测坐标时 b 对角仍不含 nugget（目标点不是训练观测自身）。 */
export function predictPoint(
  rawPoints: Point[] | PreparedData,
  model: ModelParams,
  target: { x: number; y: number },
  targetKind: TargetKind = "latent",
): KrigingPointResult {
  validateModel(model);
  const prepared: PreparedData =
    "points" in rawPoints && (rawPoints as PreparedData).groups
      ? (rawPoints as PreparedData)
      : prepareData(validateRaw(rawPoints as Point[]), model);
  const points = prepared.points as TrainPoint[];
  const n = points.length;

  const A = buildMatrixLocal(points, model);
  const b = buildRhs(points, target, model);
  const solved = solveLinear(A, b);
  if (solved.singular) {
    throw new KrigingError(
      "SINGULAR",
      "克里金方程组奇异：无法唯一求解，且未做未声明正则化",
    );
  }
  const sol = solved.x;
  const weights = sol.slice(0, n);
  const lagrange = sol[n];
  const weightsSum = weights.reduce((s, w) => s + w, 0);

  let predicted = 0;
  for (let i = 0; i < n; i++) predicted += weights[i] * points[i].v;

  // 标准普通克里金：sigma2 = C(0) - w'b - mu。
  // 本系统增广变量直接是该 mu（方程 K w + 1 mu = k）。
  let wb = 0;
  for (let i = 0; i < n; i++) wb += weights[i] * b[i];
  const rawVariance = model.psill - wb - lagrange;

  let krigingVariance = rawVariance;
  let clamped = false;
  const relTiny = TINY_NEG_VAR * model.psill;
  const relSub = SUBSTANTIAL_NEG_VAR * model.psill;
  let substantial = false;
  if (rawVariance < 0) {
    if (rawVariance < relSub) {
      substantial = true;
    } else if (rawVariance >= relTiny) {
      krigingVariance = 0;
      clamped = true;
    } else {
      // 介于 tiny 与 substantial 之间：保留为负并标记，提示数值/病态
      substantial = rawVariance < relSub;
    }
  }

  const diagnostics = toDiagnostics(n + 1, solved, weightsSum);

  return {
    target: { x: target.x, y: target.y },
    weights: points.map((p, i) => ({
      index: p.sourceIndices[0],
      x: p.x,
      y: p.y,
      weight: weights[i],
    })),
    lagrangeMultiplier: lagrange,
    predicted,
    krigingVariance,
    rawVariance,
    observationVariance:
      targetKind === "observation"
        ? Math.max(0, krigingVariance) + model.nugget
        : null,
    weightsSum,
    negativeWeightCount: weights.filter((w) => w < 0).length,
    clampedNegativeVariance: clamped,
    substantialNegativeVariance: substantial,
    diagnostics,
  };
}

export function validateGrid(g: GridSpec): void {
  const nums = [g.xmin, g.ymin, g.xmax, g.ymax, g.nx, g.ny];
  if (nums.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
    throw new KrigingError("INVALID_INPUT", "网格参数必须为有限数值");
  }
  if (g.xmax <= g.xmin || g.ymax <= g.ymin) {
    throw new KrigingError("INVALID_INPUT", "网格要求 xmax>xmin 且 ymax>ymin");
  }
  if (!Number.isInteger(g.nx) || !Number.isInteger(g.ny) || g.nx < 1 || g.ny < 1) {
    throw new KrigingError("INVALID_INPUT", "nx、ny 必须为正整数");
  }
  if (g.nx * g.ny > 40000) {
    throw new KrigingError("INVALID_INPUT", "网格单元过多（上限 40000，例如 200x200）");
  }
}

export function predictGrid(
  rawPoints: Point[],
  model: ModelParams,
  grid: GridSpec,
  targetKind: TargetKind = "latent",
): GridResult {
  validateModel(model);
  validateGrid(grid);
  const prepared = prepareData(validateRaw(rawPoints), model);
  const cells: GridCell[] = [];
  let worst = 0;
  let illCount = 0;
  for (let iy = 0; iy < grid.ny; iy++) {
    const y =
      grid.ny === 1
        ? grid.ymin
        : grid.ymin + ((grid.ymax - grid.ymin) * iy) / (grid.ny - 1);
    for (let ix = 0; ix < grid.nx; ix++) {
      const x =
        grid.nx === 1
          ? grid.xmin
          : grid.xmin + ((grid.xmax - grid.xmin) * ix) / (grid.nx - 1);
      const r = predictPoint(prepared, model, { x, y }, targetKind);
      if (r.diagnostics.conditionNumberEstimate > worst) {
        worst = Number.isFinite(r.diagnostics.conditionNumberEstimate)
          ? r.diagnostics.conditionNumberEstimate
          : worst;
      }
      if (r.diagnostics.illConditioned) illCount++;
      cells.push({ ...r, ix, iy });
    }
  }
  return {
    grid,
    target: targetKind,
    cells,
    worstConditionNumber: worst,
    illConditionedCount: illCount,
  };
}

/**
 * 逐点留一交叉验证：每次真正删除被预测测点后用其余点重训重算。
 * 标准化残差使用与“新观测”同口径方差 = 潜在场方差 + nugget。
 */
export function leaveOneOut(rawPoints: Point[], model: ModelParams): LooResult {
  validateModel(model);
  const valid = validateRaw(rawPoints);
  if (valid.length < 2) {
    throw new KrigingError(
      "INVALID_INPUT",
      "留一交叉验证至少需要 2 个测点（单测点删除后无训练数据）",
    );
  }
  // 对重合点先检测冲突（零噪声）；LOO 时重合点需成组删除，否则其它重合副本会“泄漏”答案。
  const fullPrepared = prepareData(valid, model);
  const folds: LooFold[] = [];

  for (let k = 0; k < valid.length; k++) {
    const held = valid[k];
    // 删除与 held 同位置的全部副本，避免重合泄漏
    const train = valid.filter(
      (p) =>
        Math.hypot(p.x - held.x, p.y - held.y) >
        // 使用与数据检查一致的阈值（局部导入避免循环）
        1e-10,
    );
    if (train.length === 0) {
      throw new KrigingError(
        "INVALID_INPUT",
        `删除下标 ${k}（及其重合副本）后无训练点，无法做该折`,
      );
    }
    const r = predictPoint(train, model, { x: held.x, y: held.y }, "observation");
    const predVar = (r.observationVariance ?? r.krigingVariance) || 0;
    const error = r.predicted - held.v;
    folds.push({
      index: k,
      x: held.x,
      y: held.y,
      observed: held.v,
      predicted: r.predicted,
      error,
      krigingVariance: r.krigingVariance,
      predictionVariance: predVar,
      standardizedResidual: predVar > 0 ? error / Math.sqrt(predVar) : NaN,
      illConditioned: r.diagnostics.illConditioned,
    });
  }

  const n = folds.length;
  const me = folds.reduce((s, f) => s + f.error, 0) / n;
  const mae = folds.reduce((s, f) => s + Math.abs(f.error), 0) / n;
  const mse = folds.reduce((s, f) => s + f.error * f.error, 0) / n;
  const stdVals = folds.filter((f) => Number.isFinite(f.standardizedResidual));
  const meanStandardized =
    stdVals.reduce((s, f) => s + f.standardizedResidual, 0) / (stdVals.length || 1);
  const rmseStandardized = Math.sqrt(
    stdVals.reduce((s, f) => s + f.standardizedResidual ** 2, 0) /
      (stdVals.length || 1),
  );
  const meanKrigingVariance =
    folds.reduce((s, f) => s + f.krigingVariance, 0) / n;
  let covered = 0;
  for (const f of folds) {
    const sd = Math.sqrt(f.predictionVariance);
    if (sd > 0 && Math.abs(f.error) <= 1.96 * sd) covered++;
  }
  const summary: LooSummary = {
    folds: n,
    me,
    mae,
    rmse: Math.sqrt(mse),
    mse,
    meanStandardized,
    rmseStandardized,
    meanKrigingVariance,
    coverage95: covered / n,
  };
  void fullPrepared;
  return { folds, summary };
}

export { covariance, covarianceBetween };
