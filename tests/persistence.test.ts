import { test } from "node:test";
import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { openDatabase, getDb } from "../src/db.ts";
import {
  createDataset,
  createModel,
  getReport,
  saveReport,
} from "../src/store.ts";
import { predictPoint } from "../src/kriging.ts";
import type { ModelParams, Point } from "../src/types.ts";

const dbPath = resolve(tmpdir(), `okrig-test-${Date.now()}.db`);

test("重启后报告复现：重新打开库读取绑定快照结果一致", () => {
  rmSync(dbPath, { force: true });
  openDatabase(dbPath);
  const pts: Point[] = [
    { x: 0, y: 0, v: 1 },
    { x: 3, y: 0, v: 4 },
    { x: 1, y: 2, v: 2 },
  ];
  const model: ModelParams = { psill: 2, length: 2.5, nugget: 0.1 };
  const ds = createDataset("重启测试", pts, "米");
  const m = createModel(ds.id, model, "基准");
  const result = predictPoint(pts, model, { x: 1, y: 1 }, "observation");
  const reportId = saveReport("predict", ds.id, m.id, result);

  // 模拟重启：清空实例缓存后重新打开同一文件
  getDb().close();
  openDatabase(dbPath);
  const loaded = getReport(reportId) as any;
  assert.equal(loaded.dataset.pointCount, 3);
  assert.equal(loaded.consistency.samePointVersion, true);
  assert.ok(Math.abs(loaded.result.predicted - result.predicted) < 1e-12);
  assert.ok(
    Math.abs(loaded.result.krigingVariance - result.krigingVariance) < 1e-12,
  );
  getDb().close();
  rmSync(dbPath, { force: true });
  for (const suffix of ["-wal", "-shm"]) {
    rmSync(dbPath + suffix, { force: true });
  }
});
