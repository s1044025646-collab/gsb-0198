// Dense linear algebra for small Kriging systems (n typically <= 400).
// Gaussian elimination with partial pivoting, plus condition diagnostics.

export interface SolveResult {
  /** Solution vector x of A x = b. */
  x: number[];
  /** Residual infinity norm max|A x - b| (raw matrix units). */
  residualInf: number;
  /** Relative residual ||Ax-b||inf / (||A||inf ||x||inf + ||b||inf). */
  residualRel: number;
  /** Cheap 1-norm condition-number estimate; Infinity when singular. */
  cond1Estimate: number;
  /** True when a pivot at (near) machine zero was encountered. */
  singular: boolean;
  /** Number of non-negligible pivots during elimination. */
  rank: number;
  /** Smallest accepted pivot divided by matrix infinity norm. */
  minPivotRatio: number;
}

function normInf(A: number[][]): number {
  let m = 0;
  for (const row of A) {
    let s = 0;
    for (const v of row) s += Math.abs(v);
    if (s > m) m = s;
  }
  return m;
}

function matVec(A: number[][], x: number[]): number[] {
  return A.map((row) => {
    let s = 0;
    for (let j = 0; j < x.length; j++) s += row[j]! * x[j]!;
    return s;
  });
}

// Independent Gaussian-elimination solver (partial pivot) used for condition
// probing and inversion; throws on exact singularity.
function solveSafe(AIn: number[][], bIn: number[]): number[] {
  const n = AIn.length;
  const A = AIn.map((r) => r.slice());
  const b = bIn.slice();
  for (let k = 0; k < n; k++) {
    let p = k;
    let pv = Math.abs(A[k]![k]!);
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(A[i]![k]!) > pv) {
        pv = Math.abs(A[i]![k]!);
        p = i;
      }
    }
    if (pv === 0) throw new Error("singular matrix");
    if (p !== k) {
      const t = A[k]!; A[k] = A[p]!; A[p] = t;
      const tb = b[k]!; b[k] = b[p]!; b[p] = tb;
    }
    for (let i = k + 1; i < n; i++) {
      const f = A[i]![k]! / A[k]![k]!;
      for (let j = k; j < n; j++) A[i]![j] = A[i]![j]! - f * A[k]![j]!;
      b[i] = b[i]! - f * b[k]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]!;
    for (let j = i + 1; j < n; j++) s -= A[i]![j]! * x[j]!;
    x[i] = s / A[i]![i]!;
  }
  return x;
}

/** Solve A x = b for a square matrix. Never mutates inputs. */
function solveSafeTranspose(A: number[][], rhs: number[]): number[] {
  const n = A.length;
  const T = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) T[i]![j] = A[j]![i]!;
  return solveSafe(T, rhs);
}

export function solveLinear(AIn: number[][], bIn: number[]): SolveResult {
  const n = AIn.length;
  if (n === 0 || bIn.length !== n) {
    throw new Error("solveLinear: dimension mismatch");
  }
  const A = AIn.map((r) => r.slice());
  const b = bIn.slice();
  const scale = Math.max(normInf(A), Number.EPSILON);
  const pivotTol = n * Number.EPSILON * scale;

  let rank = 0;
  let minPivotRatio = Infinity;
  let singular = false;

  for (let k = 0; k < n; k++) {
    let p = k;
    let pv = Math.abs(A[k]![k]!);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i]![k]!);
      if (v > pv) {
        pv = v;
        p = i;
      }
    }
    if (pv <= pivotTol) {
      singular = true;
      continue;
    }
    minPivotRatio = Math.min(minPivotRatio, pv / scale);
    rank++;
    if (p !== k) {
      const tmp = A[k]!;
      A[k] = A[p]!;
      A[p] = tmp;
      const tb = b[k]!;
      b[k] = b[p]!;
      b[p] = tb;
    }
    for (let i = k + 1; i < n; i++) {
      const factor = A[i]![k]! / A[k]![k]!;
      A[i]![k] = 0;
      for (let j = k + 1; j < n; j++) {
        A[i]![j] = A[i]![j]! - factor * A[k]![j]!;
      }
      b[i] = b[i]! - factor * b[k]!;
    }
  }

  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i]!;
    for (let j = i + 1; j < n; j++) sum -= A[i]![j]! * x[j]!;
    const piv = A[i]![i]!;
    x[i] = Math.abs(piv) <= pivotTol ? 0 : sum / piv;
  }

  const Ax = matVec(AIn, x);
  let resInf = 0;
  for (let i = 0; i < n; i++) {
    resInf = Math.max(resInf, Math.abs(Ax[i]! - bIn[i]!));
  }
  const xInf = Math.max(1, ...x.map((v) => Math.abs(v)));
  const bInf = Math.max(1, ...bIn.map((v) => Math.abs(v)));
  const residualRel = resInf / (normInf(AIn) * xInf + bInf);

  let cond1Estimate: number;
  try {
    let a1 = 0;
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.abs(AIn[i]![j]!);
      a1 = Math.max(a1, s);
    }
    // Hager/Higham-style 1-norm estimate of ||A^{-1}|| with a few iterations.
    let x = new Array<number>(n).fill(1 / n);
    let invNorm = 0;
    for (let iter = 0; iter < 3; iter++) {
      const v = solveSafe(AIn, x);
      const gamma = v.reduce((acc, z) => acc + Math.abs(z), 0);
      invNorm = Math.max(invNorm, gamma);
      const xi = v.map((z) => (z >= 0 ? 1 : -1));
      // x ~= A^{-T} xi
      x = solveSafeTranspose(AIn, xi);
      const maxAbs = Math.max(1, ...x.map((z) => Math.abs(z)));
      x = x.map((z) => z / maxAbs);
    }
    cond1Estimate = a1 * invNorm;
  } catch {
    cond1Estimate = Infinity;
  }

  return {
    x,
    residualInf: resInf,
    residualRel,
    cond1Estimate,
    singular,
    rank,
    minPivotRatio: singular ? 0 : minPivotRatio,
  };
}

export function invertMatrix(A: number[][]): number[][] {
  const n = A.length;
  const cols: number[][] = [];
  for (let j = 0; j < n; j++) {
    const e = new Array<number>(n).fill(0);
    e[j] = 1;
    cols.push(solveSafe(A, e));
  }
  const rows: number[][] = Array.from(
    { length: n },
    () => new Array<number>(n).fill(0),
  );
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) rows[i]![j] = cols[j]![i]!;
  return rows;
}
