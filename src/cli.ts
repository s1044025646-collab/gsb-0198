#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { openDatabase } from "./db.js";
import {
  createDataset,
  createModel,
  getDataset,
  getModel,
  getReport,
  listDatasets,
  listModels,
  listReports,
  saveReport,
} from "./store.js";
import { prepareData } from "./data.js";
import { leaveOneOut, predictGrid, predictPoint } from "./kriging.js";
import { KrigingError, type ModelParams, type Point, type TargetKind } from "./types.js";

interface ParsedArgs {
  positional: string[];
  flags: Record<string, string>;
}

function parseArgs(argv: string[]): ParsedArgs {
  const positional: string[] = [];
  const flags: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) {
        flags[key] = "true";
      } else {
        flags[key] = next;
        i++;
      }
    } else {
      positional.push(a);
    }
  }
  return { positional, flags };
}

function modelFromFlags(f: Record<string, string>): ModelParams {
  return {
    psill: Number(f.psill),
    length: Number(f.length),
    nugget: Number(f.nugget ?? 0),
  };
}

function loadPointsFromFile(path: string): Point[] {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (Array.isArray(raw)) return raw as Point[];
  if (Array.isArray(raw.points)) return raw.points as Point[];
  throw new KrigingError("INVALID_INPUT", "导入文件需为测点数组或含 points 字段");
}

function out(payload: unknown): void {
  process.stdout.write(JSON.stringify(payload, null, 2) + "\n");
}

function usage(): void {
  process.stdout.write(
    [
      "普通克里金 CLI",
      "用法: krig <命令> [参数]",
      "命令:",
      "  init-db                                 初始化数据库",
      "  import --file <json> [--name n] [--unit u]  导入数据集",
      "  datasets                                列出数据集",
      "  dataset --dataset <id>                  查看数据集（含测点）",
      "  inspect --dataset <id> --psill --length [--nugget]  数据检查/重复点策略",
      "  add-model --dataset <id> --psill --length [--nugget] [--label]  保存模型快照",
      "  models --dataset <id>                   列出模型快照",
      "  predict --dataset <id> --model <id> --x --y [--target latent|observation]",
      "  grid --dataset <id> --model <id> --xmin --ymin --xmax --ymax --nx --ny [--target]",
      "  loocv --dataset <id> --model <id>      逐点留一交叉验证",
      "  compare --dataset <id> --specs <json>  多组参数对照（至少2组）",
      "  reports [--dataset <id>]               列出报告",
      "  export --report <id> --out <file>      导出报告 JSON",
      "",
    ].join("\n"),
  );
}

async function main(): Promise<void> {
  const { positional, flags } = parseArgs(process.argv.slice(2));
  const cmd = positional[0];
  const dataDir = flags["data-dir"]
    ? resolve(flags["data-dir"])
    : resolve(process.cwd(), "data");
  const dbPath = resolve(dataDir, "kriging.db");

  if (!cmd || cmd === "-h" || cmd === "--help" || cmd === "help") {
    usage();
    return;
  }
  if (cmd !== "init-db") {
    openDatabase(dbPath);
  }

  switch (cmd) {
    case "init-db": {
      mkdirSync(dataDir, { recursive: true });
      openDatabase(dbPath);
      out({ ok: true, dbPath });
      break;
    }
    case "import": {
      const points = loadPointsFromFile(resolve(flags.file ?? ""));
      const rec = createDataset(flags.name ?? "导入数据集", points, flags.unit ?? "");
      const { points: _p, ...meta } = rec;
      out(meta);
      break;
    }
    case "datasets":
      out({ datasets: listDatasets() });
      break;
    case "dataset": {
      const ds = getDataset(Number(flags.dataset));
      if (!ds) throw new KrigingError("INVALID_INPUT", "数据集不存在");
      out(ds);
      break;
    }
    case "inspect": {
      const ds = getDataset(Number(flags.dataset));
      if (!ds) throw new KrigingError("INVALID_INPUT", "数据集不存在");
      const model = modelFromFlags(flags);
      const prepared = prepareData(ds.points, model);
      out({
        raw: ds.points.length,
        afterDedup: prepared.points.length,
        issues: prepared.issues,
      });
      break;
    }
    case "add-model": {
      const params = modelFromFlags(flags);
      out(createModel(Number(flags.dataset), params, flags.label ?? ""));
      break;
    }
    case "models":
      out({ models: listModels(Number(flags.dataset)) });
      break;
    case "predict":
    case "grid":
    case "loocv": {
      const ds = getDataset(Number(flags.dataset));
      const model = getModel(Number(flags.model));
      if (!ds || !model || model.datasetId !== ds.id)
        throw new KrigingError("INVALID_INPUT", "数据集/模型缺失或不匹配");
      const target = (flags.target === "observation"
        ? "observation"
        : "latent") as TargetKind;
      let result: unknown;
      let kind: string;
      if (cmd === "predict") {
        result = predictPoint(
          ds.points,
          model.params,
          { x: Number(flags.x), y: Number(flags.y) },
          target,
        );
        kind = "predict";
      } else if (cmd === "grid") {
        result = predictGrid(
          ds.points,
          model.params,
          {
            xmin: Number(flags.xmin),
            ymin: Number(flags.ymin),
            xmax: Number(flags.xmax),
            ymax: Number(flags.ymax),
            nx: Number(flags.nx),
            ny: Number(flags.ny),
          },
          target,
        );
        kind = "grid";
      } else {
        result = leaveOneOut(ds.points, model.params);
        kind = "loocv";
      }
      const reportId = saveReport(kind, ds.id, model.id, result);
      out({ reportId, result });
      break;
    }
    case "compare": {
      const ds = getDataset(Number(flags.dataset));
      if (!ds) throw new KrigingError("INVALID_INPUT", "数据集不存在");
      const specs = JSON.parse(flags.specs ?? "[]") as ModelParams[];
      if (specs.length < 2)
        throw new KrigingError("INVALID_INPUT", "--specs 至少含两组参数");
      out({
        comparisons: specs.map((params, i) => ({
          index: i,
          model: params,
          summary: leaveOneOut(ds.points, params).summary,
        })),
      });
      break;
    }
    case "reports":
      out({ reports: listReports(flags.dataset ? Number(flags.dataset) : undefined) });
      break;
    case "export": {
      const report = getReport(Number(flags.report));
      if (!report) throw new KrigingError("INVALID_INPUT", "报告不存在");
      const outPath = resolve(flags.out ?? "report.json");
      mkdirSync(dirname(outPath), { recursive: true });
      writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");
      out({ ok: true, outPath });
      break;
    }
    default:
      usage();
  }
}

main().catch((err: unknown) => {
  if (err instanceof KrigingError) {
    process.stderr.write(`错误[${err.code}]: ${err.message}\n`);
  } else {
    process.stderr.write(`错误: ${(err as Error).message}\n`);
  }
  process.exitCode = 1;
});
