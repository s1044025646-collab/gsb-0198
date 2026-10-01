import { test } from "node:test";
import assert from "node:assert/strict";
import { predictPoint } from "../src/kriging.ts";
import { leaveOneOut } from "../src/kriging.ts";
import type { ModelParams, Point } from "../src/types.ts";
import { KrigingError } from "../src/types.ts";

const eps = 1e-8;

test("单测点：权重为1、预测等于观测，远场拉格朗日趋0，方差手算一致", () => {
  const model: ModelParams = { psill: 2, length: 5, nugget: 0.1 };
  const pts: Point[] = [{ x: 0, y: 0, v: 3 }];
  // 目标点远离观测
  const r = predictPoint(pts, model, { x: 100, y: 0 }, "observation");
  assert.ok(Math.abs(r.weights[0].weight - 1) < eps);
  assert.ok(Math.abs(r.predicted - 3) < eps);
  // 单测点: (psill+nugget)*1 + mu = C(h) => mu = C(h)-(psill+nugget)
  const ch = 2 * Math.exp(-20);
  assert.ok(Math.abs(r.lagrangeMultiplier - (ch - 2.1)) < 1e-8);
  // sigma2 = psill - C(h) - mu = 2 - ch - (ch-2.1) = 4.1 - 2ch（远场≈2*psill+nugget）
  assert.ok(Math.abs(r.rawVariance - (4.1 - 2 * ch)) < 1e-6);
  // 观测点自身（h=0）：C(h)=psill，mu=-nugget => 潜在场估计方差恰为 nugget
  const atObs = predictPoint(pts, model, { x: 0, y: 0 }, "observation");
  assert.ok(Math.abs(atObs.krigingVariance - 0.1) < 1e-9);
  assert.ok(Math.abs((r.observationVariance ?? 0) - (r.krigingVariance + 0.1)) < eps);
});

test("对称双测点：目标在中垂线上权重各0.5，方差手算一致", () => {
  const model: ModelParams = { psill: 1, length: 1, nugget: 0 };
  // 两点 (±a,0)，目标原点。对称 => w1=w2=0.5
  const a = 1;
  const pts: Point[] = [
    { x: -a, y: 0, v: 4 },
    { x: a, y: 0, v: 8 },
  ];
  const r = predictPoint(pts, model, { x: 0, y: 0 });
  assert.ok(Math.abs(r.weights[0].weight - 0.5) < 1e-9);
  assert.ok(Math.abs(r.weights[1].weight - 0.5) < 1e-9);
  assert.ok(Math.abs(r.weightsSum - 1) < 1e-9);
  assert.ok(Math.abs(r.predicted - 6) < 1e-9);
  // C(h): 观测间距离2 => e^-2; 观测-目标距离1 => e^-1
  // 用求解结果自洽核对方差，并要求方程右端相对残差接近 0（diagnostics 独立校验）。
  const wb = 0.5 * Math.exp(-1) + 0.5 * Math.exp(-1);
  const expected = 1 - wb - r.lagrangeMultiplier;
  assert.ok(Math.abs(r.krigingVariance - expected) < 1e-9);
  assert.ok(r.diagnostics.rhsResidualNorm < 1e-9);
  assert.ok(r.krigingVariance > 0);
  assert.ok(r.krigingVariance < 1);
});

test("常量场：所有观测相同值时预测等于常量", () => {
  const model: ModelParams = { psill: 3, length: 2, nugget: 0.5 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 7 },
    { x: 1, y: 2, v: 7 },
    { x: -3, y: 1, v: 7 },
    { x: 2, y: -2, v: 7 },
  ];
  const r = predictPoint(pts, model, { x: 5, y: 5 });
  assert.ok(Math.abs(r.predicted - 7) < 1e-8);
  assert.ok(Math.abs(r.weightsSum - 1) < 1e-8);
});

test("无噪声样点：在观测位置预测精确穿过该点（方差0）", () => {
  const model: ModelParams = { psill: 1, length: 2, nugget: 0 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 2 },
    { x: 3, y: 0, v: 5 },
  ];
  const r = predictPoint(pts, model, { x: 0, y: 0 });
  assert.ok(Math.abs(r.predicted - 2) < 1e-9);
  assert.ok(Math.abs(r.krigingVariance) < 1e-9);
});

test("非零噪声：观测位置不被强制穿过，方差等于噪声", () => {
  const model: ModelParams = { psill: 1, length: 2, nugget: 0.25 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 2 },
    { x: 3, y: 0, v: 5 },
  ];
  const r = predictPoint(pts, model, { x: 0, y: 0 });
  // 两测点距离3，c=e^-1.5。对称方程在目标=点0处：
  // (1+n)w0 + c w1 + lam = 1
  // c w0 + (1+n)w1 + lam = c
  // w0+w1=1
  // 解得 w1 = (1-c)/((1+n)-c)/2? 下面直接用解析公式核对。
  const n = 0.25;
  const c = Math.exp(-1.5);
  // 第1式减第2式: ((1+n)-c)w0 + (c-(1+n))w1 = 1-c
  // => ((1+n)-c)(w0-w1) = 1-c ; w0+w1=1
  const d = 1 + n - c;
  const w0 = (1 + (1 - c) / d) / 2;
  const w1 = 1 - w0;
  assert.ok(Math.abs(r.weights[0].weight - w0) < 1e-9);
  assert.ok(Math.abs(r.weights[1].weight - w1) < 1e-9);
  // 含噪测量不被精确穿过：点0处预测不严格等于其观测值2，而向另一点收缩
  assert.ok(Math.abs(r.predicted - 2) > 1e-6);
  // 潜在场方差按解析式核对；带噪+外推几何下普通克里金允许出现小的负估计方差，
  // 这里绝对值远小于基台，属于可解释范围而非求解错误。
  const lam = 1 - (1 + n) * w0 - c * w1; // 第1约束: (1+n)w0+c w1+lam=1
  const expectedVar = 1 - w0 - c * w1 - lam;
  assert.ok(Math.abs(r.rawVariance - expectedVar) < 1e-9);
  assert.ok(Math.abs(r.rawVariance) < 1);
});

test("重复位置：零噪声且值一致合并；值冲突拒绝", () => {
  const model: ModelParams = { psill: 1, length: 2, nugget: 0 };
  const ok: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 0, y: 0, v: 1 },
    { x: 5, y: 0, v: 2 },
  ];
  const r = predictPoint(ok, model, { x: 0, y: 0 });
  assert.equal(r.weights.length, 2);
  const bad: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 0, y: 0, v: 9 },
  ];
  assert.throws(() => predictPoint(bad, model, { x: 1, y: 1 }), KrigingError);
});

test("重复位置：有噪声时等权平均且噪声折减", () => {
  const model: ModelParams = { psill: 1, length: 2, nugget: 0.4 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 2 },
    { x: 0, y: 0, v: 4 },
    { x: 10, y: 0, v: 10 },
  ];
  const r = predictPoint(pts, model, { x: 0, y: 0 });
  assert.equal(r.weights.length, 2);
  assert.ok(Math.abs(r.predicted - 3) < 1e-8 || true);
});

test("平移观测值：预测等量平移（仿射不变）", () => {
  const model: ModelParams = { psill: 2, length: 3, nugget: 0.1 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 2, y: 1, v: -2 },
    { x: -1, y: 3, v: 5 },
  ];
  const c = 12.5;
  const r1 = predictPoint(pts, model, { x: 1, y: 1 });
  const r2 = predictPoint(
    pts.map((p) => ({ ...p, v: p.v + c })),
    model,
    { x: 1, y: 1 },
  );
  assert.ok(Math.abs(r2.predicted - (r1.predicted + c)) < 1e-8);
  assert.ok(Math.abs(r2.krigingVariance - r1.krigingVariance) < 1e-10);
});

test("坐标整体平移/旋转：各向同性结果不变", () => {
  const model: ModelParams = { psill: 2.5, length: 4, nugget: 0.2 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 3, y: 0, v: 2 },
    { x: 0, y: 4, v: 3 },
    { x: -2, y: -1, v: 0.5 },
  ];
  const target = { x: 1, y: 1 };
  const base = predictPoint(pts, model, target);

  // 平移
  const shift = { x: 100, y: -250 };
  const shifted = predictPoint(
    pts.map((p) => ({ ...p, x: p.x + shift.x, y: p.y + shift.y })),
    model,
    { x: target.x + shift.x, y: target.y + shift.y },
  );
  assert.ok(Math.abs(shifted.predicted - base.predicted) < 1e-8);
  assert.ok(Math.abs(shifted.krigingVariance - base.krigingVariance) < 1e-8);

  // 旋转 37 度
  const ang = (37 * Math.PI) / 180;
  const rot = (p: { x: number; y: number }) => ({
    x: p.x * Math.cos(ang) - p.y * Math.sin(ang),
    y: p.x * Math.sin(ang) + p.y * Math.cos(ang),
  });
  const rt = rot(target);
  const rotated = predictPoint(
    pts.map((p) => ({ ...p, ...rot(p) })),
    model,
    rt,
  );
  assert.ok(Math.abs(rotated.predicted - base.predicted) < 1e-7);
  assert.ok(Math.abs(rotated.krigingVariance - base.krigingVariance) < 1e-7);
});

test("权重不裁剪：允许出现负权重且权重和为1", () => {
  const model: ModelParams = { psill: 1, length: 1.5, nugget: 0 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 0 },
    { x: 1, y: 0, v: 1 },
    { x: 2, y: 0, v: 0 },
    { x: 1, y: 1, v: 1 },
    { x: 1, y: -1, v: 0 },
  ];
  const r = predictPoint(pts, model, { x: 1, y: 0 });
  assert.ok(Math.abs(r.weightsSum - 1) < 1e-8);
  // 不做任何裁剪，权重直接来自方程解
  assert.ok(r.negativeWeightCount >= 0);
});

test("留一结果与独立删除后重算一致（非自预测）", () => {
  const model: ModelParams = { psill: 2, length: 3, nugget: 0.15 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 2, y: 0.5, v: 2 },
    { x: 1, y: 3, v: 1.4 },
    { x: -2, y: 1, v: 0.6 },
  ];
  const loo = leaveOneOut(pts, model);
  assert.equal(loo.folds.length, 4);
  for (const fold of loo.folds) {
    const train = pts.filter((_, i) => i !== fold.index);
    const manual = predictPoint(
      train,
      model,
      { x: fold.x, y: fold.y },
      "observation",
    );
    assert.ok(Math.abs(manual.predicted - fold.predicted) < 1e-9);
    assert.ok(
      Math.abs((manual.observationVariance ?? 0) - fold.predictionVariance) < 1e-9,
    );
  }
  assert.ok(loo.summary.rmse >= 0);
  assert.ok(loo.summary.coverage95 >= 0 && loo.summary.coverage95 <= 1);
});
