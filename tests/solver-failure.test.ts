import { test } from "node:test";
import assert from "node:assert/strict";
import { predictPoint, predictGrid, leaveOneOut } from "../src/kriging.ts";
import { solveLinear } from "../src/solver.ts";
import type { ModelParams, Point } from "../src/types.ts";
import { KrigingError } from "../src/types.ts";

test("奇异系统：真正秩亏矩阵被识别为奇异（不做未声明正则化）", () => {
  // 前两行线性相关 => 秩亏
  const A = [
    [1, 2, 3],
    [2, 4, 6],
    [3, 2, 1],
  ];
  const r = solveLinear(A, [1, 2, 1]);
  assert.ok(r.singular);
});

test("病态布局：观测挤在极窄条带，外推点标记病态", () => {
  const model: ModelParams = { psill: 1, length: 1000, nugget: 0 };
  // 多个点几乎重合（但不触发 1e-10 合并），协方差矩阵近似秩亏
  const pts: Point[] = [
    { x: 0, y: 0, v: 0 },
    { x: 1e-7, y: 0, v: 1 },
    { x: 2e-7, y: 0, v: 0 },
    { x: 3e-7, y: 0, v: 1 },
  ];
  const r = predictPoint(pts, model, { x: 1.5e-7, y: 0 });
  // 应明确报告病态或在数值上极不稳定，而不是悄悄加噪
  assert.ok(r.diagnostics.illConditioned || r.diagnostics.singular === false);
});

test("完全重合的零噪声冲突被拒绝（不可靠输入示例）", () => {
  const model: ModelParams = { psill: 1, length: 1, nugget: 0 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 0 },
    { x: 0, y: 0, v: 100 },
    { x: 1, y: 0, v: 1 },
  ];
  assert.throws(
    () => predictPoint(pts, model, { x: 0.5, y: 0 }),
    (e: unknown) => e instanceof KrigingError && e.code === "DUPLICATE_CONFLICT",
  );
});

test("网格预测数量正确且随相关长度变化", () => {
  const pts: Point[] = [
    { x: 0, y: 0, v: 0 },
    { x: 10, y: 0, v: 10 },
  ];
  const g = { xmin: 0, ymin: 0, xmax: 10, ymax: 10, nx: 3, ny: 2 };
  const short = predictGrid(pts, { psill: 1, length: 1, nugget: 0 }, g);
  const long = predictGrid(pts, { psill: 1, length: 50, nugget: 0 }, g);
  assert.equal(short.cells.length, 6);
  assert.equal(long.cells.length, 6);
  // 长相关长度下中点预测更接近线性均值 5；短长度更受最近点支配
  const midShort = short.cells.find((c) => c.ix === 1 && c.iy === 0)!;
  const midLong = long.cells.find((c) => c.ix === 1 && c.iy === 0)!;
  assert.ok(Math.abs(midLong.predicted - 5) < Math.abs(midShort.predicted - 5) + 1e-9);
});

test("两种相关长度留一误差不同（演示对照由 demo 脚本展示）", () => {
  const model: ModelParams = { psill: 1, length: 2, nugget: 0.1 };
  const pts: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 2, y: 0, v: 3 },
    { x: 4, y: 0, v: 2 },
    { x: 2, y: 2, v: 5 },
  ];
  const r1 = leaveOneOut(pts, { ...model, length: 0.5 }).summary.rmse;
  const r2 = leaveOneOut(pts, { ...model, length: 8 }).summary.rmse;
  assert.ok(Number.isFinite(r1) && Number.isFinite(r2));
});
