// 共享类型定义

/** 二维测点 */
export interface Point {
  x: number;
  y: number;
  v: number; // 标量观测值
}

/**
 * 各向同性指数协方差模型参数。
 * 坐标单位必须一致（例如全部为米）；相关长度 length 与坐标同单位。
 *
 * C(h) = psill * exp(-h / length)
 * 观测协方差矩阵对角项额外加独立测量噪声方差 nugget（仅观测侧）。
 */
export interface ModelParams {
  psill: number; // 基台（偏基台）：> 0
  length: number; // 指数相关长度：> 0
  nugget: number; // 独立测量噪声方差：>= 0（不进入目标点协方差）
}

/** 预测目标类型：latent=潜在场（无噪声），observation=新观测（含噪声方差） */
export type TargetKind = "latent" | "observation";

/** 线性方程组求解诊断 */
export interface SolveDiagnostics {
  n: number; // 方程阶数（含无偏约束）
  reciprocalConditionNumber: number; // 1/rcond 越大越病态；rcond 估计值
  conditionNumberEstimate: number; // 条件数估计（可能为 Infinity）
  rhsResidualNorm: number; // A*w - b 的残差范数（组装后实际校验）
  rhsResidualMaxAbs: number;
  constraintResidual: number; // sum(w)-1 的偏差
  singular: boolean;
  illConditioned: boolean;
  message?: string;
}

/** 单点克里金结果 */
export interface KrigingPointResult {
  target: { x: number; y: number };
  weights: { index: number; x: number; y: number; weight: number }[];
  lagrangeMultiplier: number;
  predicted: number;
  /** 预测方差（潜在场口径）。小量负方差被钳制并在 clampedNegativeVariance 标记 */
  krigingVariance: number;
  rawVariance: number; // 求解得到的未钳制方差
  observationVariance: number | null; // 新观测口径方差（= 潜在方差 + nugget）
  weightsSum: number;
  negativeWeightCount: number;
  clampedNegativeVariance: boolean;
  substantialNegativeVariance: boolean;
  diagnostics: SolveDiagnostics;
}

/** 留一交叉验证单测点结果 */
export interface LooFold {
  index: number;
  x: number;
  y: number;
  observed: number;
  predicted: number;
  error: number; // predicted - observed
  krigingVariance: number; // 该点潜在场预测方差（训练集不含自身）
  predictionVariance: number; // 与观测同口径方差 = krigingVariance + nugget
  standardizedResidual: number; // error / sqrt(predictionVariance)
  illConditioned: boolean;
}

export interface LooSummary {
  folds: number;
  me: number; // 平均误差（偏差）
  mae: number; // 平均绝对误差
  rmse: number; // 均方根误差
  mse: number;
  // 标准化残差（与观测同口径）汇总
  meanStandardized: number;
  rmseStandardized: number;
  // 平均克里金方差与预测区间覆盖率（±1.96σ）
  meanKrigingVariance: number;
  coverage95: number;
}

export interface LooResult {
  folds: LooFold[];
  summary: LooSummary;
}

/** 网格定义 */
export interface GridSpec {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  nx: number;
  ny: number;
}

export interface GridCell extends KrigingPointResult {
  ix: number;
  iy: number;
}

export interface GridResult {
  grid: GridSpec;
  target: TargetKind;
  cells: GridCell[];
  worstConditionNumber: number;
  illConditionedCount: number;
}

export class KrigingError extends Error {
  constructor(
    public code:
      | "INVALID_INPUT"
      | "DUPLICATE_CONFLICT"
      | "SINGULAR"
      | "ILL_CONDITIONED",
    message: string,
  ) {
    super(message);
    this.name = "KrigingError";
  }
}
