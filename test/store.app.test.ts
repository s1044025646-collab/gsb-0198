import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AddressInfo } from "node:net";
import { Store } from "../src/store.js";
import { createApp } from "../src/app.js";
import type { Server } from "node:http";
import { leaveOneOut, predictPoint } from "../src/kriging.js";

const tmpDir = mkdtempSync(join(tmpdir(), "kriging-test-"));
const dbPath = join(tmpDir, "sub", "kriging.sqlite");

describe("SQLite 快照与重启复现", () => {
  const observations = [
    { x: 0, y: 0, value: 1, label: "a" },
    { x: 1, y: 0, value: 0.5, label: "b" },
    { x: 0, y: 1, value: 2, label: "c" },
    { x: 2, y: 2, value: -1, label: "d" },
  ];
  const model = { sill: 1, range: 1.2, measurementNoiseVariance: 0.05 };

  it("关闭后重开，冻结快照与报告逐字节复现", () => {
    let store = new Store(dbPath);
    const ds = store.createDataset("demo", observations, "m", "unit");
    const analysis = store.createAnalysis(ds.id, "m1", model);
    const point = predictPoint(
      { observations: analysis.observations, model: analysis.model },
      { x: 1, y: 1 },
    );
    const loo = leaveOneOut({
      observations: analysis.observations,
      model: analysis.model,
    });
    store.saveReport(analysis.id, "point", point);
    store.saveReport(analysis.id, "loocv", loo);
    const reportCount = store.listReports(analysis.id).length;
    const pointPrediction = point.prediction;
    const looRMSE = loo.summary!.rootMeanSquareError;
    store.close();

    // Reopen: snapshots must be immutable and results identical.
    store = new Store(dbPath);
    const reloaded = store.getAnalysis(analysis.id)!;
    assert.deepEqual(reloaded.observations, observations);
    assert.deepEqual(reloaded.model, model);
    const reports = store.listReports(analysis.id);
    assert.equal(reports.length, reportCount);
    const repPoint = reports.find((r) => r.kind === "point")!.payload as {
      prediction: number;
    };
    const repLoo = reports.find((r) => r.kind === "loocv")!.payload as {
      summary: { rootMeanSquareError: number };
    };
    assert.equal(repPoint.prediction, pointPrediction);
    assert.equal(repLoo.summary.rootMeanSquareError, looRMSE);
    // Recompute from snapshot: bitwise identical.
    const again = predictPoint(
      { observations: reloaded.observations, model: reloaded.model },
      { x: 1, y: 1 },
    );
    assert.equal(again.prediction, pointPrediction);
    store.close();
  });

  it("同一分析不会混入两版测点：数据集后改不影响已冻结分析", () => {
    const store = new Store(dbPath);
    const datasets = store.listDatasets();
    const dsId = datasets[0]!.id;
    const before = store.getAnalysis(1)!;
    // Create a *new* dataset/analysis rather than mutating; old analysis
    // snapshot stays identical.
    const ds2 = store.createDataset("demo-v2", [
      ...observations,
      { x: 5, y: 5, value: 99 },
    ]);
    assert.notEqual(ds2.id, dsId);
    const after = store.getAnalysis(1)!;
    assert.deepEqual(after.observations, before.observations);
    store.close();
  });
});

describe("HTTP API 端到端", () => {
  let server: Server;
  let base: string;
  let store: Store;

  before(async () => {
    store = new Store(join(tmpDir, "api.sqlite"));
    server = createApp({ store });
    await new Promise<void>((resolve) => server.listen(0, resolve));
    base = `http://localhost:${(server.address() as AddressInfo).port}`;
  });
  after(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    store.close();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  async function json<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(base + path, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
    const body = (await res.json()) as T;
    if (!res.ok) throw new Error(JSON.stringify(body));
    return body;
  }

  it("导入 -> 检查 -> 分析 -> 单点/网格 -> 留一 -> 导出 全链路", async () => {
    const health = await json<{ ok: boolean }>("/health");
    assert.equal(health.ok, true);

    const created = await json<{ dataset: { id: number } }>("/api/datasets", {
      method: "POST",
      body: JSON.stringify({
        name: "api-ds",
        observations: [
          { x: 0, y: 0, value: 1 },
          { x: 1, y: 0, value: 0 },
          { x: 0, y: 1, value: 2 },
        ],
      }),
    });
    const dsId = created.dataset.id;

    const check = await json<{ check: { count: number } }>(
      `/api/datasets/${dsId}/check`,
      { method: "POST", body: JSON.stringify({ measurementNoiseVariance: 0 }) },
    );
    assert.equal(check.check.count, 3);

    const analysis = await json<{ analysis: { id: number } }>("/api/analyses", {
      method: "POST",
      body: JSON.stringify({
        datasetId: dsId,
        model: { sill: 1, range: 1, measurementNoiseVariance: 0.1 },
      }),
    });
    const aid = analysis.analysis.id;

    const point = await json<{
      result: { prediction: number; weights: unknown[]; constraintResidual: number };
    }>(`/api/analyses/${aid}/predict-point`, {
      method: "POST",
      body: JSON.stringify({ target: { x: 0.5, y: 0.5 }, predictWhat: "latent" }),
    });
    assert.ok(Number.isFinite(point.result.prediction));
    assert.equal(point.result.weights.length, 3);
    assert.ok(point.result.constraintResidual < 1e-10);

    const grid = await json<{ output: { nodes: unknown[] } }>(
      `/api/analyses/${aid}/predict-grid`,
      {
        method: "POST",
        body: JSON.stringify({
          grid: { xMin: 0, xMax: 1, yMin: 0, yMax: 1, nx: 3, ny: 3 },
        }),
      },
    );
    assert.equal(grid.output.nodes.length, 9);

    const loo = await json<{
      feasible: boolean;
      summary: { count: number; rootMeanSquareError: number };
    }>(`/api/analyses/${aid}/cross-validation`, { method: "POST" });
    assert.equal(loo.feasible, true);
    assert.equal(loo.summary.count, 3);
    assert.ok(loo.summary.rootMeanSquareError >= 0);

    const exported = await json<{
      analysis: { id: number };
      reports: { kind: string }[];
    }>(`/api/analyses/${aid}/export`);
    assert.equal(exported.analysis.id, aid);
    assert.ok(exported.reports.some((r) => r.kind === "loocv"));
  });

  it("零噪冲突重复测点被 API 拒绝（400）", async () => {
    const res = await fetch(base + "/api/datasets", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        observations: [
          { x: 0, y: 0, value: 1 },
          { x: 0, y: 0, value: 2 },
        ],
      }),
    });
    assert.equal(res.status, 400);
  });
});
