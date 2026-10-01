// End-to-end demonstration using ONLY fabricated data.
// 1) builds a dataset, 2) compares two correlation lengths (grid + LOOCV),
// 3) shows an unreliable (ill-conditioned) layout, 4) persists to SQLite and
// verifies report reproducibility after reopening the database.

import { rmSync } from "node:fs";
import { resolve } from "node:path";
import {
  leaveOneOut,
  predictGrid,
  predictPoint,
  type Observation,
} from "./index.js";
import { Store } from "./store.js";
import type { CovarianceModel } from "./types.js";

function line(t: string): void {
  console.log("\n" + "=".repeat(72) + "\n" + t + "\n" + "=".repeat(72));
}

const observations: Observation[] = [
  { x: 0, y: 0, value: 1.0, label: "p0" },
  { x: 2, y: 0, value: 1.8, label: "p1" },
  { x: 4, y: 0.5, value: 2.6, label: "p2" },
  { x: 0.5, y: 2.5, value: 1.4, label: "p3" },
  { x: 2.5, y: 3, value: 2.1, label: "p4" },
  { x: 4.5, y: 2.5, value: 3.0, label: "p5" },
  { x: 1, y: 5, value: 1.1, label: "p6" },
  { x: 3.5, y: 5.5, value: 2.0, label: "p7" },
  { x: 6, y: 5, value: 2.8, label: "p8" },
];

const baseModel: CovarianceModel = {
  sill: 1,
  range: 1.5,
  measurementNoiseVariance: 0.04,
};

function f(v: number): string {
  return Number.isFinite(v) ? v.toFixed(4) : "NaN";
}

line("1) 单点预测（潜在场）— 权重/乘子/方程与约束残差");
{
  const r = predictPoint(
    { observations, model: baseModel },
    { x: 3, y: 2.5 },
    "latent",
  );
  console.log("目标 (3, 2.5)  预测 =", f(r.prediction), " 标准差 =", f(r.varianceStdDev));
  console.log("权重和 =", f(r.weightSum), " 拉格朗日乘子 mu =", f(r.lagrangeMultiplier));
  console.log("方程残差(相对) =", r.equationResidualRel.toExponential(2),
    " 无偏约束残差 =", r.constraintResidual.toExponential(2),
    " 条件数 ≈", isFinite(r.condition1Estimate) ? r.condition1Estimate.toExponential(2) : "Inf");
  console.log("各观测权重:");
  for (const w of r.weights)
    console.log(`  ${w.label.padEnd(4)} w = ${f(w.weight)}`);
}

line("2) 同一数据集在两种相关长度下的网格预测（7x5）对比");
const grid = { xMin: 0, xMax: 6, yMin: 0, yMax: 5, nx: 7, ny: 5 };
for (const range of [0.8, 4.0]) {
  const model = { ...baseModel, range };
  const out = predictGrid({ observations, model }, grid, "latent");
  const preds = out.nodes.map((n) => n.prediction);
  const sds = out.nodes.map((n) => n.varianceStdDev);
  console.log(`range = ${range}: min=${f(Math.min(...preds))} max=${f(Math.max(...preds))} ` +
    `平均标准差=${f(sds.reduce((a, b) => a + b, 0) / sds.length)} 非可靠节点=${out.illCount}`);
}

line("3) 两种相关长度下的逐点留一交叉验证（每次真正剔除该点）");
for (const range of [0.8, 4.0]) {
  const model = { ...baseModel, range };
  const loo = leaveOneOut({ observations, model });
  const s = loo.summary!;
  console.log(`range = ${range}: RMSE=${f(s.rootMeanSquareError)} MAE=${f(s.meanAbsoluteError)} ` +
    `偏差=${f(s.meanError)} 标准化残差均方=${f(s.meanSquaredStandardizedResidual)} 95%覆盖=${f(s.coverage95)}`);
  console.log("  各折误差(观测-预测)与标准化残差:");
  for (const fold of loo.folds)
    console.log(`  ${fold.label.padEnd(4)} err=${f(fold.error).padStart(9)} z=${f(fold.standardizedResidual).padStart(9)}`);
}

line("4) 不可可靠求解的输入示例（测点几乎完全重合 -> 严重病态）");
{
  const illModel: CovarianceModel = { sill: 1, range: 1, measurementNoiseVariance: 0 };
  const bad: Observation[] = [
    { x: 0, y: 0, value: 1 },
    { x: 1e-10, y: 0, value: 2 },
    { x: 2e-10, y: 0, value: 1.5 },
    { x: 3e-10, y: 0, value: 2.5 },
  ];
  const r = predictPoint({ observations: bad, model: illModel }, { x: 1, y: 1 });
  console.log("health =", r.health, " 条件数 ≈",
    isFinite(r.condition1Estimate) ? r.condition1Estimate.toExponential(2) : "Inf");
  if (r.messages.length === 0)
    console.log("  （求解器报告健康；该布局的数值风险见 check 的 warning）");
  for (const msg of r.messages) console.log("  -", msg);
  console.log("（不裁剪负权重，也不添加未声明对角项）");
}

line("5) 零噪声冲突重复测点（同位置不同值）被拒绝");
{
  const conflict: Observation[] = [
    { x: 0, y: 0, value: 1 },
    { x: 0, y: 0, value: 2 },
  ];
  try {
    predictPoint(
      { observations: conflict, model: { ...baseModel, measurementNoiseVariance: 0 } },
      { x: 1, y: 1 },
    );
    console.log("意外通过（不应发生）");
  } catch (e) {
    console.log("已拒绝:", (e as Error).message.split("\n").slice(0, 2).join(" "));
  }
}

line("6) SQLite 持久化与重启后报告复现");
{
  const dbPath = resolve("data", "demo.sqlite");
  rmSync(dbPath, { force: true });
  rmSync(dbPath + "-wal", { force: true });
  rmSync(dbPath + "-shm", { force: true });

  let store = new Store(dbPath);
  const ds = store.createDataset(
    "demo-fabricated",
    observations,
    "consistent-length-unit",
    "scalar-unit",
  );
  const analysis = store.createAnalysis(ds.id, "range-1.5", baseModel);
  const before = predictPoint(
    { observations: analysis.observations, model: analysis.model },
    { x: 3, y: 2.5 },
  );
  const looBefore = leaveOneOut({
    observations: analysis.observations,
    model: analysis.model,
  });
  store.saveReport(analysis.id, "point", before);
  store.saveReport(analysis.id, "loocv", looBefore);
  console.log("已写入数据集/分析/报告 ->", dbPath);
  store.close();

  store = new Store(dbPath);
  const reloaded = store.getAnalysis(analysis.id)!;
  const after = predictPoint(
    { observations: reloaded.observations, model: reloaded.model },
    { x: 3, y: 2.5 },
  );
  const reports = store.listReports(analysis.id);
  console.log("重启后单点预测 =", f(after.prediction),
    " 与重启前一致:", Math.abs(after.prediction - before.prediction) < 1e-15);
  console.log("持久化报告数 =", reports.length,
    " 留一 RMSE =", f(looBefore.summary!.rootMeanSquareError));
  store.close();
}

line("演示完成（全部数据均为虚构，仅用于算法验证）");
