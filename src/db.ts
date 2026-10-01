import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

let dbInstance: DatabaseSync | null = null;
let dbPath = "";

function isOpen(db: DatabaseSync): boolean {
  try {
    db.prepare("SELECT 1").get();
    return true;
  } catch {
    return false;
  }
}

export function openDatabase(path: string): DatabaseSync {
  if (dbInstance) {
    if (dbPath === path && isOpen(dbInstance)) return dbInstance;
    try {
      dbInstance.close();
    } catch {
      // 已关闭则忽略
    }
    dbInstance = null;
  }
  mkdirSync(dirname(path), { recursive: true });
  const db = new DatabaseSync(path);
  db.exec("PRAGMA journal_mode = WAL;");
  db.exec("PRAGMA foreign_keys = ON;");
  initSchema(db);
  dbInstance = db;
  dbPath = path;
  return db;
}

export function getDb(): DatabaseSync {
  if (!dbInstance) throw new Error("数据库尚未初始化");
  return dbInstance;
}

function initSchema(db: DatabaseSync): void {
  db.exec(`
  CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    unit_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    point_count INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    v REAL NOT NULL
  );
  CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    psill REAL NOT NULL,
    length REAL NOT NULL,
    nugget REAL NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_points_dataset ON points(dataset_id);
  CREATE INDEX IF NOT EXISTS idx_models_dataset ON models(dataset_id);
  CREATE INDEX IF NOT EXISTS idx_reports_dataset ON reports(dataset_id);
  `);
}
