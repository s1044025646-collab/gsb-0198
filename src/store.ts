import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { checkObservations } from "./dataset.js";
import type { CovarianceModel, Observation } from "./types.js";

export interface DatasetRow {
  id: number;
  name: string;
  observations: Observation[];
  coordinateUnit: string;
  valueUnit: string;
  createdAt: string;
}

export interface AnalysisRow {
  id: number;
  datasetId: number;
  name: string;
  // Frozen copies: an analysis never mixes two versions of the observations.
  observations: Observation[];
  model: CovarianceModel;
  createdAt: string;
}

export interface ReportRow {
  id: number;
  analysisId: number;
  kind: "point" | "grid" | "loocv" | "check";
  payload: unknown;
  createdAt: string;
}

export class Store {
  readonly db: DatabaseSync;

  constructor(dbPath: string) {
    const abs = resolve(dbPath);
    mkdirSync(dirname(abs), { recursive: true });
    this.db = new DatabaseSync(abs);
    this.db.exec("PRAGMA journal_mode = WAL;");
    this.db.exec("PRAGMA foreign_keys = ON;");
    this.migrate();
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        observations TEXT NOT NULL,
        coordinate_unit TEXT NOT NULL DEFAULT 'same-unit (unspecified)',
        value_unit TEXT NOT NULL DEFAULT 'same-unit (unspecified)',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
      CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL REFERENCES datasets(id),
        name TEXT NOT NULL,
        observations_snapshot TEXT NOT NULL,
        model_snapshot TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
      CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL REFERENCES analyses(id),
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
    `);
  }

  createDataset(
    name: string,
    observations: Observation[],
    coordinateUnit = "same-unit (unspecified)",
    valueUnit = "same-unit (unspecified)",
  ): DatasetRow {
    const check = checkObservations(observations, 0);
    if (check.errors.length) {
      throw new Error(`数据检查失败：\n - ${check.errors.join("\n - ")}`);
    }
    const stmt = this.db.prepare(
      `INSERT INTO datasets (name, observations, coordinate_unit, value_unit)
       VALUES (?, ?, ?, ?)`,
    );
    const info = stmt.run(
      name,
      JSON.stringify(observations),
      coordinateUnit,
      valueUnit,
    );
    return this.getDataset(Number(info.lastInsertRowid))!;
  }

  getDataset(id: number): DatasetRow | null {
    const row = this.db
      .prepare("SELECT * FROM datasets WHERE id = ?")
      .get(id) as
      | {
          id: number; name: string; observations: string;
          coordinate_unit: string; value_unit: string; created_at: string;
        }
      | undefined;
    if (!row) return null;
    return {
      id: row.id,
      name: row.name,
      observations: JSON.parse(row.observations) as Observation[],
      coordinateUnit: row.coordinate_unit,
      valueUnit: row.value_unit,
      createdAt: row.created_at,
    };
  }

  listDatasets(): DatasetRow[] {
    const rows = this.db
      .prepare("SELECT id FROM datasets ORDER BY id")
      .all() as { id: number }[];
    return rows
      .map((r) => this.getDataset(r.id)!)
      .filter((r): r is DatasetRow => r !== null);
  }

  /** Freeze the dataset observations + model into an immutable analysis. */
  createAnalysis(
    datasetId: number,
    name: string,
    model: CovarianceModel,
  ): AnalysisRow {
    const ds = this.getDataset(datasetId);
    if (!ds) throw new Error(`数据集 ${datasetId} 不存在`);
    const info = this.db
      .prepare(
        `INSERT INTO analyses (dataset_id, name, observations_snapshot, model_snapshot)
         VALUES (?, ?, ?, ?)`,
      )
      .run(datasetId, name, JSON.stringify(ds.observations), JSON.stringify(model));
    return this.getAnalysis(Number(info.lastInsertRowid))!;
  }

  getAnalysis(id: number): AnalysisRow | null {
    const row = this.db
      .prepare("SELECT * FROM analyses WHERE id = ?")
      .get(id) as
      | {
          id: number; dataset_id: number; name: string;
          observations_snapshot: string; model_snapshot: string;
          created_at: string;
        }
      | undefined;
    if (!row) return null;
    return {
      id: row.id,
      datasetId: row.dataset_id,
      name: row.name,
      observations: JSON.parse(row.observations_snapshot) as Observation[],
      model: JSON.parse(row.model_snapshot) as CovarianceModel,
      createdAt: row.created_at,
    };
  }

 listAnalyses(): AnalysisRow[] {
    const rows = this.db
      .prepare("SELECT id FROM analyses ORDER BY id")
      .all() as { id: number }[];
    return rows
      .map((r) => this.getAnalysis(r.id)!)
      .filter((r): r is AnalysisRow => r !== null);
  }

  saveReport(
    analysisId: number,
    kind: ReportRow["kind"],
    payload: unknown,
  ): ReportRow {
    if (!this.getAnalysis(analysisId)) throw new Error(`分析 ${analysisId} 不存在`);
    const info = this.db
      .prepare("INSERT INTO reports (analysis_id, kind, payload) VALUES (?, ?, ?)")
      .run(analysisId, kind, JSON.stringify(payload));
    return {
      id: Number(info.lastInsertRowid),
      analysisId,
      kind,
      payload,
      createdAt: new Date().toISOString(),
    };
  }

  listReports(analysisId?: number): ReportRow[] {
    const rows = (analysisId === undefined
      ? (this.db.prepare("SELECT * FROM reports ORDER BY id").all() as {
          id: number; analysis_id: number; kind: ReportRow["kind"];
          payload: string; created_at: string;
        }[])
      : (this.db
          .prepare("SELECT * FROM reports WHERE analysis_id = ? ORDER BY id")
          .all(analysisId) as {
          id: number; analysis_id: number; kind: ReportRow["kind"];
          payload: string; created_at: string;
        }[]));
    return rows.map((r) => ({
      id: r.id,
      analysisId: r.analysis_id,
      kind: r.kind,
      payload: JSON.parse(r.payload) as unknown,
      createdAt: r.created_at,
    }));
  }

  close(): void {
    this.db.close();
  }
}
