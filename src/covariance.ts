import type { ModelParams, Point } from "./types.js";
import { KrigingError } from "./types.js";

export function validateModel(m: ModelParams): void {
  if (typeof m.psill !== "number" || !Number.isFinite(m.psill) || m.psill <= 0) {
    throw new KrigingError("INVALID_INPUT", "基台 psill 必须为正数");
  }
  if (typeof m.length !== "number" || !Number.isFinite(m.length) || m.length <= 0) {
    throw new KrigingError("INVALID_INPUT", "相关长度 length 必须为正数");
  }
  if (
    typeof m.nugget !== "number" ||
    !Number.isFinite(m.nugget) ||
    m.nugget < 0
  ) {
    throw new KrigingError(
      "INVALID_INPUT",
      "独立测量噪声方差 nugget 必须为非负数",
    );
  }
}

export function euclid(a: { x: number; y: number }, b: { x: number; y: number }): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

/** 各向同性指数协方差（不含测量噪声项）。 */
export function covariance(h: number, m: ModelParams): number {
  return m.psill * Math.exp(-h / m.length);
}

export function covarianceBetween(
  a: { x: number; y: number },
  b: { x: number; y: number },
  m: ModelParams,
): number {
  return covariance(euclid(a, b), m);
}

/**
 * 构建 n 个观测之间的克里金系数矩阵（含无偏约束），阶数 n+1。
 * 对角项为 C(0)+nugget = psill+nugget：噪声只加在观测协方差对角。
 */
export function buildKrigingMatrix(points: Point[], m: ModelParams): number[][] {
  const n = points.length;
  const size = n + 1;
  const A: number[][] = Array.from({ length: size }, () =>
    new Array<number>(size).fill(0),
  );
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let c: number;
      if (i === j) {
        c = m.psill + m.nugget; // C(0) + 独立噪声
      } else {
        c = covarianceBetween(points[i], points[j], m);
      }
      A[i][j] = c;
    }
    A[i][n] = 1;
    A[n][i] = 1;
  }
  A[n][n] = 0;
  return A;
}

/**
 * 目标点与各观测的协方差右端项（不含 nugget：目标不是带噪观测本身）。
 * 末尾为无偏约束常数 1。
 */
export function buildRhs(
  points: Point[],
  target: { x: number; y: number },
  m: ModelParams,
): number[] {
  const n = points.length;
  const b = new Array<number>(n + 1);
  for (let i = 0; i < n; i++) {
    b[i] = covarianceBetween(points[i], target, m);
  }
  b[n] = 1;
  return b;
}
