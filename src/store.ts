import { getDb } from "./db.js";
import type { ModelParams, Point } from "./types.js";

export interface DatasetRecord {
  id: number;
  name: string;
  unitNote: string;
  createdAt: string;
  pointCount: number;
  pointHash: string;
  points: Point[];
}

function hashPoints(points: Point[]): string {
  let h = 2166136261;
  const s = JSON.stringify(points.map((p) => [p.x, p.y, p.v]));
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function createDataset(
  name: string,
  points: Point[],
  unitNote: string,
): DatasetRecord {
  const db = getDb();
  const pointHash = hashPoints(points);
  const info = db
    .prepare("INSERT INTO datasets (name, unit_note, point_count) VALUES (?, ?, ?)")
    .run(name, unitNote, points.length);
  const id = Number(info.lastInsertRowid);
  const ins = db.prepare(
    "INSERT INTO points (dataset_id, ord, x, y, v) VALUES (?, ?, ?, ?, ?)",
  );
  points.forEach((p, i) => ins.run(id, i, p.x, p.y, p.v));
  void pointHash;
  return getDataset(id)!;
}

interface DatasetRow {
  id: number;
  name: string;
  unit_note: string;
  created_at: string;
  point_count: number;
}

export function getDataset(id: number): DatasetRecord | null {
  const db = getDb();
  const row = db.prepare("SELECT * FROM datasets WHERE id = ?").get(id) as
    | DatasetRow
    | undefined;
  if (!row) return null;
  const prows = db
    .prepare("SELECT x, y, v FROM points WHERE dataset_id = ? ORDER BY ord")
    .all(id) as { x: number; y: number; v: number }[];
  const points = prows.map((r) => ({ x: r.x, y: r.y, v: r.v }));
  return {
    id: row.id,
    name: row.name,
    unitNote: row.unit_note,
    createdAt: row.created_at,
    pointCount: row.point_count,
    pointHash: hashPoints(points),
    points,
  };
}

export function listDatasets(): Omit<DatasetRecord, "points">[] {
  const db = getDb();
  const rows = db.prepare("SELECT id FROM datasets ORDER BY id").all() as {
    id: number;
  }[];
  return rows.map((r) => {
    const d = getDataset(r.id)!;
    const { points: _points, ...meta } = d;
    return meta;
  });
}

export interface ModelRecord {
  id: number;
  datasetId: number;
  params: ModelParams;
  label: string;
  createdAt: string;
  pointHash: string;
}

export function createModel(
  datasetId: number,
  params: ModelParams,
  label: string,
): ModelRecord {
  const db = getDb();
  const ds = getDataset(datasetId);
  if (!ds) throw new Error(`数据集 ${datasetId} 不存在`);
  const info = db
    .prepare(
      "INSERT INTO models (dataset_id, psill, length, nugget, label) VALUES (?, ?, ?, ?, ?)",
    )
    .run(datasetId, params.psill, params.length, params.nugget, label);
  const id = Number(info.lastInsertRowid);
  return getModel(id)!;
}

interface ModelRow {
  id: number;
  dataset_id: number;
  psill: number;
  length: number;
  nugget: number;
  label: string;
  created_at: string;
}

export function getModel(id: number): ModelRecord | null {
  const db = getDb();
  const row = db.prepare("SELECT * FROM models WHERE id = ?").get(id) as
    | ModelRow
    | undefined;
  if (!row) return null;
  const ds = getDataset(row.dataset_id)!;
  return {
    id: row.id,
    datasetId: row.dataset_id,
    params: { psill: row.psill, length: row.length, nugget: row.nugget },
    label: row.label,
    createdAt: row.created_at,
    pointHash: ds.pointHash,
  };
}

export function listModels(datasetId: number): ModelRecord[] {
  const db = getDb();
  const rows = db
    .prepare("SELECT id FROM models WHERE dataset_id = ? ORDER BY id")
    .all(datasetId) as { id: number }[];
  return rows.map((r) => getModel(r.id)!);
}

/**
 * 保存报告。payload 外包一层绑定信息：数据集/模型 id、参数、测点指纹与数量，
 * 保证同一分析绑定同一版测点，重启后可原样复现。
 */
export function saveReport(
  kind: string,
  datasetId: number,
  modelId: number,
  payload: unknown,
): number {
  const db = getDb();
  const ds = getDataset(datasetId);
  const model = getModel(modelId);
  if (!ds || !model) throw new Error("数据集或模型不存在");
  if (model.datasetId !== datasetId) {
    throw new Error("模型与数据集不匹配，禁止跨分析混用");
  }
  const wrapped = {
    kind,
    dataset: {
      id: ds.id,
      name: ds.name,
      pointCount: ds.pointCount,
      pointHash: ds.pointHash,
    },
    model: { id: model.id, ...model.params, pointHash: model.pointHash },
    consistency: {
      samePointVersion: ds.pointHash === model.pointHash,
    },
    result: payload,
    savedAt: new Date().toISOString(),
  };
  const info = db
    .prepare(
      "INSERT INTO reports (dataset_id, model_id, kind, payload) VALUES (?, ?, ?, ?)",
    )
    .run(datasetId, modelId, kind, JSON.stringify(wrapped, null, 2));
  return Number(info.lastInsertRowid);
}

export function getReport(id: number): unknown | null {
  const db = getDb();
  const row = db.prepare("SELECT payload FROM reports WHERE id = ?").get(id) as
    | { payload: string }
    | undefined;
  return row ? (JSON.parse(row.payload) as unknown) : null;
}

interface ReportRow {
  id: number;
  dataset_id: number;
  model_id: number;
  kind: string;
  created_at: string;
}

export function listReports(datasetId?: number) {
  const db = getDb();
  const rows = datasetId
    ? (db
        .prepare(
          "SELECT id, dataset_id, model_id, kind, created_at FROM reports WHERE dataset_id = ? ORDER BY id",
        )
        .all(datasetId) as unknown as ReportRow[])
    : (db
        .prepare(
          "SELECT id, dataset_id, model_id, kind, created_at FROM reports ORDER BY id",
        )
        .all() as unknown as ReportRow[]);
  return rows.map((r) => ({
    id: r.id,
    datasetId: r.dataset_id,
    modelId: r.model_id,
    kind: r.kind,
    createdAt: r.created_at,
  }));
}
