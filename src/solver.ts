// 带部分选主元的高斯消元（LU）求解器，含条件数估计与残差校验。
// 不做任何未声明的对角正则化；奇异/病态直接报告。

export interface LuFactor {
  lu: number[][]; // 合并的 LU
  piv: number[]; // 行置换
  n: number;
  singular: boolean;
  smallestPivotRatio: number;
  normInf: number;
  normOne: number;
}

function matCopy(A: number[][]): number[][] {
  return A.map((row) => row.slice());
}

export function luDecompose(A: number[][]): LuFactor {
  const n = A.length;
  const lu = matCopy(A);
  const piv = Array.from({ length: n }, (_, i) => i);
  let normInf = 0;
  let normOne = 0;
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let j = 0; j < n; j++) r += Math.abs(lu[i][j]);
    if (r > normInf) normInf = r;
  }
  for (let j = 0; j < n; j++) {
    let c = 0;
    for (let i = 0; i < n; i++) c += Math.abs(lu[i][j]);
    if (c > normOne) normOne = c;
  }

  let smallestPivotRatio = Infinity;
  let singular = false;
  const scale = normInf || 1;

  for (let k = 0; k < n; k++) {
    let maxRow = k;
    let maxVal = Math.abs(lu[k][k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i][k]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }
    if (maxRow !== k) {
      const tmp = lu[k];
      lu[k] = lu[maxRow];
      lu[maxRow] = tmp;
      const t = piv[k];
      piv[k] = piv[maxRow];
      piv[maxRow] = t;
    }
    const pivot = lu[k][k];
    const ratio = Math.abs(pivot) / scale;
    if (ratio < smallestPivotRatio) smallestPivotRatio = ratio;
    if (pivot === 0 || !Number.isFinite(pivot)) {
      singular = true;
      continue;
    }
    for (let i = k + 1; i < n; i++) {
      lu[i][k] /= pivot;
      const lik = lu[i][k];
      for (let j = k + 1; j < n; j++) {
        lu[i][j] -= lik * lu[k][j];
      }
    }
  }
  return { lu, piv, n, singular, smallestPivotRatio, normInf, normOne };
}

function solveLu(f: LuFactor, b: number[]): number[] {
  const { lu, piv, n } = f;
  const x = new Array<number>(n);
  // 前向代入 Ly = Pb
  for (let i = 0; i < n; i++) {
    let s = b[piv[i]];
    for (let j = 0; j < i; j++) s -= lu[i][j] * x[j];
    x[i] = s;
  }
  // 回代 Ux = y
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= lu[i][j] * x[j];
    x[i] = s / lu[i][i];
  }
  return x;
}

// 求解 A^T x = b（利用同一 LU）
function solveLuTranspose(f: LuFactor, bIn: number[]): number[] {
  const { lu, piv, n } = f;
  // 求 (PA=LU)^T => A^T P^T = U^T L^T；先解 U^T y = P b，再解 L^T x = y
  const b = new Array<number>(n);
  for (let i = 0; i < n; i++) b[piv[i]] = bIn[i];
  const y = new Array<number>(n);
  for (let j = 0; j < n; j++) {
    let s = b[j];
    for (let i = 0; i < j; i++) s -= lu[i][j] * y[i]; // U^T 列元素 lu[j][i]? 元素 U[i][j]=lu[i][j], i<=j；U^T[j][i]=lu[i][j]
    y[j] = s / lu[j][j];
  }
  const x = new Array<number>(n);
  for (let j = n - 1; j >= 0; j--) {
    let s = y[j];
    // L^T[j][i] = L[i][j]=lu[i][j] for i>j；L 对角为 1
    for (let i = j + 1; i < n; i++) s -= lu[i][j] * x[i];
    x[j] = s;
  }
  return x;
}

function dotInfNorm(v: number[]): number {
  let m = 0;
  for (const x of v) {
    const a = Math.abs(x);
    if (a > m) m = a;
  }
  return m;
}
function oneNorm(v: number[]): number {
  let s = 0;
  for (const x of v) s += Math.abs(x);
  return s;
}

/**
 * 估计 ||A^-1||_1（简化的 1-范数逆估计，Hager/Higham 思路，固定少量迭代）。
 * 返回估计值。
 */
export function inverseOneNormEstimate(f: LuFactor, iterations = 3): number {
  const n = f.n;
  if (f.singular) return Infinity;
  let x = new Array<number>(n).fill(1 / n);
  let gamma = 0;
  for (let k = 0; k < iterations; k++) {
    // y = A^-1 x  => A y = x
    const y = solveLu(f, x);
    const gammaNew = oneNorm(y);
    const xi = y.map((val) => (val >= 0 ? 1 : -1));
    // z = A^-T xi => A^T z = xi
    const z = solveLuTranspose(f, xi);
    let zMaxAbs = 0;
    let zMaxIdx = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(z[i]);
      if (a > zMaxAbs) {
        zMaxAbs = a;
        zMaxIdx = i;
      }
    }
    if (k > 0 && zMaxAbs <= z.reduce((s, zv, i) => s + (x[i] >= 0 ? zv : -zv), 0)) {
      gamma = gammaNew;
      break;
    }
    gamma = gammaNew;
    x = new Array<number>(n).fill(0);
    x[zMaxIdx] = 1;
    if (zMaxAbs <= gammaNew) {
      gamma = gammaNew;
      break;
    }
  }
  return gamma;
}

export interface SolveResult {
  x: number[];
  reciprocalConditionNumber: number;
  conditionNumberEstimate: number;
  residualNormInf: number;
  residualMaxAbs: number;
  singular: boolean;
  illConditioned: boolean;
}

/** 条件数告警阈值：rcond 低于该值视为严重病态 */
export const RCOND_WARN = 1e-8;
export const RCOND_FATAL = 1e-12;

export function solveLinear(A: number[][], b: number[]): SolveResult {
  const f = luDecompose(A);
  const n = f.n;
  let x: number[] = new Array<number>(n).fill(NaN);
  let conditionNumberEstimate = Infinity;
  let reciprocalConditionNumber = 0;
  let singular = f.singular;

  if (!singular) {
    x = solveLu(f, b);
    const invNorm = inverseOneNormEstimate(f);
    conditionNumberEstimate = f.normOne * invNorm;
    reciprocalConditionNumber =
      conditionNumberEstimate > 0 && Number.isFinite(conditionNumberEstimate)
        ? 1 / conditionNumberEstimate
        : 0;
  } else {
    // 退化情形：仍尝试标记奇异，不声称得到解
  }

  // 残差校验 A x - b（无穷范数与最大绝对分量）
  let residualNormInf = 0;
  let residualMaxAbs = 0;
  if (!singular) {
    for (let i = 0; i < n; i++) {
      let s = -b[i];
      for (let j = 0; j < n; j++) s += A[i][j] * x[j];
      const a = Math.abs(s);
      residualMaxAbs = Math.max(residualMaxAbs, a);
      const rowScale = (f.normInf || 1);
      const rel = a / (rowScale * (dotInfNorm(x) || 1) + Math.abs(b[i]) + 1);
      if (rel > residualNormInf) residualNormInf = rel;
    }
  }

  // 以最小主元比与 rcond 双重判定
  const illConditioned =
    !singular &&
    (reciprocalConditionNumber < RCOND_WARN ||
      f.smallestPivotRatio < 1e-10 ||
      residualNormInf > 1e-6);
  if (f.smallestPivotRatio === 0 || Number.isNaN(f.smallestPivotRatio)) {
    singular = true;
  }

  return {
    x,
    reciprocalConditionNumber,
    conditionNumberEstimate,
    residualNormInf,
    residualMaxAbs,
    singular,
    illConditioned,
  };
}
