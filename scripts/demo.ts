// 端到端演示：虚构数据、两种相关长度网格预测与留一误差、
// 一个不能可靠求解的输入，以及“重启后”报告复现。
import { resolve } from "node:path";
import { rmSync } from "node:fs";
import { openDatabase, getDb } from "../src/db.ts";
import {
  createDataset,
  createModel,
  getReport,
  saveReport,
} from "../src/store.ts";
import { leaveOneOut, predictGrid, predictPoint } from "../src/kriging.ts";
import { KrigingError, type ModelParams, type Point } from "../src/types.ts";

function line(t: string): void {
  console.log("\n=== " + t + " ===");
}

const points: Point[] = [
  { x: 0, y: 0, v: 1 },
  { x: 4, y: 0, v: 3 },
  { x: 8, y: 0, v: 2 },
  { x: 0, y: 4, v: 2.5 },
  { x: 4, y: 4, v: 5 },
  { x: 8, y: 4, v: 3.5 },
  { x: 2, y: 8, v: 1.5 },
  { x: 6, y: 8, v: 4.2 },
];

const dbPath = resolve(process.cwd(), "data", "demo.db");
rmSync(dbPath, { force: true });
openDatabase(dbPath);

line("单点预测与权重解释（潜在场 vs 新观测）");
const m0: ModelParams = { psill: 1, length: 3, nugget: 0.1 };
const sp = predictPoint(points, m0, { x: 4, y: 4 }, "observation");
console.log("预测值:", sp.predicted.toFixed(4));
console.log("权重和:", sp.weightsSum.toFixed(10), "拉格朗日:", sp.lagrangeMultiplier.toFixed(4));
console.log("潜在场方差:", sp.krigingVariance.toFixed(4), "新观测方差:", sp.observationVariance?.toFixed(4));
console.log("方程右端相对残差:", sp.diagnostics.rhsResidualNorm.toExponential(2));

const ds = createDataset("演示-虚构测点", points, "坐标与数值单位一致（虚构，无地质含义）");

line("两种相关长度：网格预测 + 留一交叉验证");
const grid = { xmin: 0, ymin: 0, xmax: 8, ymax: 8, nx: 5, ny: 5 };
for (const length of [1.5, 8]) {
  const params: ModelParams = { psill: 1, length, nugget: 0.1 };
  const g = predictGrid(points, params, grid, "latent");
  const loo = leaveOneOut(points, params);
  const model = createModel(ds.id, params, `length=${length}`);
  saveReport("grid", ds.id, model.id, g);
  saveReport("loocv", ds.id, model.id, loo);
  const center = g.cells.find((c) => c.ix === 2 && c.iy === 2)!;
  console.log(
    `length=${length}  中心点预测=${center.predicted.toFixed(3)} 方差=${center.krigingVariance.toFixed(3)} | LOOCV RMSE=${loo.summary.rmse.toFixed(4)} MAE=${loo.summary.mae.toFixed(4)} 标准化RMSE=${loo.summary.rmseStandardized.toFixed(3)} 覆盖率95=${loo.summary.coverage95.toFixed(2)}`,
  );
}

line("不能可靠求解的输入：零噪声下重合位置冲突观测");
try {
  predictPoint(
    [
      { x: 0, y: 0, v: 0 },
      { x: 0, y: 0, v: 99 },
      { x: 5, y: 0, v: 1 },
    ],
    { psill: 1, length: 2, nugget: 0 },
    { x: 1, y: 1 },
  );
  console.log("未拒绝（异常）");
} catch (e) {
  if (e instanceof KrigingError) console.log(`已拒绝 [${e.code}]: ${e.message}`);
}

line("病态布局演示（近重复点，不外推时给出病态标记）");
const ill = predictPoint(
  [
    { x: 0, y: 0, v: 0 },
    { x: 1e-7, y: 0, v: 1 },
    { x: 2e-7, y: 0, v: 0 },
    { x: 3e-7, y: 0, v: 1 },
  ],
  { psill: 1, length: 1000, nugget: 0 },
  { x: 1.5e-7, y: 0 },
);
console.log(
  "条件数估计:",
  ill.diagnostics.conditionNumberEstimate,
  "病态标记:",
  ill.diagnostics.illConditioned,
);

line("重启复现：关闭并重新打开数据库，读取已保存报告");
const reportBefore = getReport(1) as any;
getDb().close();
openDatabase(dbPath);
const reportAfter = getReport(1) as any;
console.log(
 "report1 绑定测点指纹:",
  reportAfter.dataset.pointHash,
  " 同版测点:",
  reportAfter.consistency.samePointVersion,
);
console.log(
  "复现预测一致:",
  Math.abs(reportAfter.result.cells[12].predicted - reportBefore.result.cells[12].predicted) < 1e-12,
);
console.log("\n演示完成。数据库:", dbPath);
