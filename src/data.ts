import type { ModelParams, Point } from "./types.js";
import { KrigingError } from "./types.js";
import { euclid } from "./covariance.js";

export const MIN_POINTS = 1;
export const MAX_POINTS = 2000;
/** 判定位置重合的距离阈值（相对坐标尺度，很小） */
export const COINCIDENT_EPS = 1e-10;

export interface DataIssue {
  level: "error" | "warning";
  code: string;
  message: string;
  indices?: number[];
}

export interface PreparedData {
  /** 去重聚合后的有效测点（保留首个出现的原始下标用于权重解释） */
  points: (Point & { sourceIndices: number[] })[];
  issues: DataIssue[];
  groups: number[][]; // 每个有效测点对应的原始下标集合
}

function validNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

export function validateRaw(points: unknown): Point[] {
  if (!Array.isArray(points)) {
    throw new KrigingError("INVALID_INPUT", "points 必须为数组");
  }
  if (points.length < MIN_POINTS || points.length > MAX_POINTS) {
    throw new KrigingError(
      "INVALID_INPUT",
      `测点数量须在 ${MIN_POINTS}..${MAX_POINTS} 之间，收到 ${points.length}`,
    );
  }
  return points.map((p, i) => {
    if (!p || typeof p !== "object") {
      throw new KrigingError("INVALID_INPUT", `第 ${i} 个测点不是对象`);
    }
    const r = p as Record<string, unknown>;
    if (!validNumber(r.x) || !validNumber(r.y) || !validNumber(r.v)) {
      throw new KrigingError(
        "INVALID_INPUT",
        `第 ${i} 个测点的 x/y/v 必须为有限数值`,
      );
    }
    return { x: r.x, y: r.y, v: r.v };
  });
}

/**
 * 数据检查与重复位置处理。
 * 策略（明确、确定）：
 *  - 位置重合且 nugget=0：观测值必须一致；冲突 => 拒绝（零噪声无法解释差异）。
 *    值一致则视为同一物理测点，保留一份。
 *  - 位置重合且 nugget>0：按相同噪声方差做等权平均（逆方差加权在此退化为等权），
 *    聚合后方差为 nugget/k（k 个重复）。为保持统一对角噪声，聚合点单独记录其噪声折减，
 *    本实现要求模型噪声逐点指定时才折减；当前模型为单一 nugget，故采用等权平均并把该组
 *    噪声视为 nugget/k，通过给克里金矩阵该观测行使用折减噪声实现。
 */
export function prepareData(
  rawPoints: Point[],
  m: ModelParams,
): PreparedData {
  const issues: DataIssue[] = [];
  const used = new Array<boolean>(rawPoints.length).fill(false);
  const points: (Point & { sourceIndices: number[]; localNugget: number })[] = [];
  const groups: number[][] = [];

  for (let i = 0; i < rawPoints.length; i++) {
    if (used[i]) continue;
    const cluster = [i];
    used[i] = true;
    for (let j = i + 1; j < rawPoints.length; j++) {
      if (used[j]) continue;
      if (euclid(rawPoints[i], rawPoints[j]) <= COINCIDENT_EPS) {
        cluster.push(j);
        used[j] = true;
      }
    }
    const vals = cluster.map((idx) => rawPoints[idx].v);
    if (cluster.length > 1) {
      if (m.nugget === 0) {
        const first = vals[0];
        const conflict = vals.some((v) => Math.abs(v - first) > COINCIDENT_EPS);
        if (conflict) {
          throw new KrigingError(
            "DUPLICATE_CONFLICT",
            `零噪声下位置重合的测点观测值冲突（下标 ${cluster.join(", ")}）；无法把含噪测量当成必须穿过的真值之外的确定真值`,
          );
        }
        issues.push({
          level: "warning",
          code: "DUPLICATE_ZERO_NOISE",
          message: `下标 ${cluster.join(", ")} 位置重合且值一致，合并为一个测点`,
          indices: cluster,
        });
        points.push({
          x: rawPoints[i].x,
          y: rawPoints[i].y,
          v: first,
          sourceIndices: cluster,
          localNugget: 0,
        });
      } else {
        const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
        issues.push({
          level: "warning",
          code: "DUPLICATE_AVERAGED",
          message: `下标 ${cluster.join(", ")} 位置重合，按逆噪声方差（此处等权）平均，独立噪声方差折减为 nugget/${cluster.length}`,
          indices: cluster,
        });
        points.push({
          x: rawPoints[i].x,
          y: rawPoints[i].y,
          v: mean,
          sourceIndices: cluster,
          localNugget: m.nugget / cluster.length,
        });
      }
    } else {
      points.push({
        x: rawPoints[i].x,
        y: rawPoints[i].y,
        v: rawPoints[i].v,
        sourceIndices: cluster,
        localNugget: m.nugget,
      });
    }
    groups.push(cluster);
  }
  return { points, issues, groups };
}
