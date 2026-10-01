import { euclid } from "./covariance.js";
import { LIMITS, type Observation } from "./types.js";

export interface DuplicateInfo {
  /** 0-based observation indices sharing the exact same location. */
  indices: number[];
  x: number;
  y: number;
  values: number[];
  /** Range of values at this location; 0 means exact repeats. */
  valueSpread: number;
  /**
   * Conflict = duplicate with zero noise allowed AND unequal values.
   * Such data cannot be interpolated exactly and must be rejected.
   */
  conflictWithZeroNoise: boolean;
}

export interface DatasetCheck {
  count: number;
  xRange: [number, number];
  yRange: [number, number];
  valueRange: [number, number];
  duplicates: DuplicateInfo[];
  errors: string[];
  warnings: string[];
}

export function checkObservations(
  obs: Observation[],
  measurementNoiseVariance: number,
): DatasetCheck {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (obs.length === 0) {
    errors.push("观测点数量必须 >= 1");
  }
  if (obs.length > LIMITS.maxObservations) {
    errors.push(
      `观测点数量 ${obs.length} 超过上限 ${LIMITS.maxObservations}（稠密求解器）`,
    );
  }
  for (let i = 0; i < obs.length; i++) {
    const o = obs[i]!;
    if (!Number.isFinite(o.x) || !Number.isFinite(o.y) || !Number.isFinite(o.value)) {
      errors.push(`第 ${i} 个观测含非有限数值（x/y/value 必须均为有限数）`);
    }
  }

  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let vMin = Infinity, vMax = -Infinity;
  for (const o of obs) {
    xMin = Math.min(xMin, o.x); xMax = Math.max(xMax, o.x);
    yMin = Math.min(yMin, o.y); yMax = Math.max(yMax, o.y);
    vMin = Math.min(vMin, o.value); vMax = Math.max(vMax, o.value);
  }

  // Group exact duplicate locations.
  const groups = new Map<string, number[]>();
  for (let i = 0; i < obs.length; i++) {
    const o = obs[i]!;
    const key = `${o.x}|${o.y}`;
    const g = groups.get(key);
    if (g) g.push(i);
    else groups.set(key, [i]);
  }

  const duplicates: DuplicateInfo[] = [];
  for (const indices of groups.values()) {
    if (indices.length < 2) continue;
    const values = indices.map((i) => obs[i]!.value);
    const spread = Math.max(...values) - Math.min(...values);
    const conflict = measurementNoiseVariance === 0 && spread !== 0;
    duplicates.push({
      indices,
      x: obs[indices[0]!]!.x,
      y: obs[indices[0]!]!.y,
      values,
      valueSpread: spread,
      conflictWithZeroNoise: conflict,
    });
    if (conflict) {
      errors.push(
        `位置 (${obs[indices[0]!]!.x}, ${obs[indices[0]!]!.y}) 存在 ${indices.length} 个值不同的重合观测（${values.join(", ")}），且噪声方差为 0：无法无偏精确穿过，必须拒绝或开启噪声`,
      );
    } else {
      warnings.push(
        `位置 (${obs[indices[0]!]!.x}, ${obs[indices[0]!]!.y}) 有 ${indices.length} 个重合观测；策略：保留全部，通过各自噪声项区分，不做强制相等`,
      );
    }
  }

  // Near-duplicate / ill-layout hint: very small minimum pair distance vs span.
  if (obs.length >= 2 && errors.length === 0) {
    let minD = Infinity;
    for (let i = 0; i < obs.length; i++)
      for (let j = i + 1; j < obs.length; j++) {
        const d = euclid(obs[i]!, obs[j]!);
        if (d > 0 && d < minD) minD = d;
      }
    const span = Math.max(xMax - xMin, yMax - yMin, 1e-12);
    if (minD < span * 1e-6) {
      warnings.push(
        `存在非常接近但不完全重合的测点（最小间距 ${minD.toExponential(2)}），可能导致克里金方程组病态`,
      );
    }
  }

  return {
    count: obs.length,
    xRange: obs.length ? [xMin, xMax] : [NaN, NaN],
    yRange: obs.length ? [yMin, yMax] : [NaN, NaN],
    valueRange: obs.length ? [vMin, vMax] : [NaN, NaN],
    duplicates,
    errors,
    warnings,
  };
}
