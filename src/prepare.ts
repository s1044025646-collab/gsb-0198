import { checkObservations } from "./dataset.js";
import type { CovarianceModel, Observation } from "./types.js";

export interface PreparedData {
  /** Effective observations fed to the Kriging system. */
  effective: Observation[];
  /**
   * effectiveIndex -> original indices merged into that effective point.
   */
  members: number[][];
  notes: string[];
}

/**
 * Duplicate-location policy (explicit, no silent dropping):
 *
 * - Zero noise + different values at the same location: rejected upstream by
 *   checkObservations (the system cannot honour conflicting exact values).
 * - Zero noise + identical values: merged into one effective observation,
 *   because identical rows would make the KKT matrix singular while adding no
 *   information.
 * - Positive measurement noise: all readings at a location are combined into
 *   one BLUE effective reading
 *       value* = sum(v_i/tau2) / sum(1/tau2)
 *   with effective noise variance tau2* = 1 / sum(1/tau2)
 *   (equal tau2 => simple average with variance tau2/k). This is equivalent
 *   the full singular-free model and yields identical predictions.
 */
export function prepareObservations(
  observations: Observation[],
  model: CovarianceModel,
): PreparedData {
  const check = checkObservations(
    observations,
    model.measurementNoiseVariance,
  );
  if (check.errors.length) {
    throw new Error(`输入不合法：\n - ${check.errors.join("\n - ")}`);
  }

  const groups = new Map<string, number[]>();
  for (let i = 0; i < observations.length; i++) {
    const o = observations[i]!;
    const key = `${o.x}|${o.y}`;
    const g = groups.get(key);
    if (g) g.push(i);
    else groups.set(key, [i]);
  }

  const effective: Observation[] = [];
  const members: number[][] = [];
  const notes: string[] = [];

  for (const indices of groups.values()) {
    const first = observations[indices[0]!]!;
    if (indices.length === 1) {
      effective.push({ ...first });
      members.push(indices);
      continue;
    }
    if (model.measurementNoiseVariance === 0) {
      // Identical values guaranteed by the conflict check.
      effective.push({
        x: first.x,
        y: first.y,
        value: first.value,
        label: first.label ?? `merged(${indices.join(",")})`,
      });
      notes.push(
        `位置 (${first.x}, ${first.y}) 的 ${indices.length} 个零噪同值观测已去重为 1 个有效测点`,
      );
    } else {
      const t2 = model.measurementNoiseVariance;
      const wSum = indices.length / t2;
      const vStar =
        indices.reduce((s, i) => s + observations[i]!.value / t2, 0) / wSum;
      effective.push({
        x: first.x,
        y: first.y,
        value: vStar,
        label: first.label ?? `merged(${indices.join(",")})`,
      });
      notes.push(
        `位置 (${first.x}, ${first.y}) 的 ${indices.length} 个有噪观测按逆噪声方差合并：值=${vStar}, 有效噪声方差=${t2 / indices.length}`,
      );
    }
    members.push(indices);
  }

  return { effective, members, notes };
}

/**
 * Effective per-observation noise variances aligned with PreparedData.
 * A merged group of k noisy readings has variance tau2/k; singletons tau2.
 */
export function effectiveNoiseVariances(
  members: number[][],
  baseNoiseVariance: number,
): number[] {
  return members.map((g) =>
    baseNoiseVariance === 0 ? 0 : baseNoiseVariance / g.length,
  );
}
