#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { checkObservations } from "./dataset.js";
import {
  leaveOneOut,
  predictGrid,
  predictPoint,
  type GridSpec,
} from "./kriging.js";
import { Store } from "./store.js";
import type { CovarianceModel, Observation, Point } from "./types.js";

const dataDir = resolve(process.env.KRIGING_DATA_DIR ?? "data");
const store = new Store(resolve(dataDir, "kriging.sqlite"));

function print(obj: unknown): void {
  process.stdout.write(JSON.stringify(obj, null, 2) + "\n");
}

function fail(msg: string): never {
  process.stderr.write(`错误: ${msg}\n`);
  process.exit(1);
}

function modelFromFlags(v: {
  sill?: string; range?: string; noise?: string;
}): CovarianceModel {
  return {
    sill: Number(v.sill),
    range: Number(v.range),
    measurementNoiseVariance: Number(v.noise ?? 0),
  };
}

async function main(): Promise<void> {
  const mainArgs = process.argv.slice(2);
  const command = mainArgs[0];
  const rest = mainArgs.slice(1);

  switch (command) {
    case "import": {
      const { values } = parseArgs({
        args: rest,
        options: {
          file: { type: "string" },
          name: { type: "string" },
          "coord-unit": { type: "string" },
          "value-unit": { type: "string" },
        },
      });
      if (!values.file) fail("需要 --file <json>");
      const raw = JSON.parse(readFileSync(values.file, "utf8")) as {
        name?: string; observations?: Observation[];
      };
      const observations = Array.isArray(raw)
        ? (raw as Observation[])
        : raw.observations;
      if (!Array.isArray(observations)) fail("JSON 需为观测数组或含 observations 字段");
      const ds = store.createDataset(
        values.name ?? raw.name ?? "cli-dataset",
        observations,
        values["coord-unit"],
        values["value-unit"],
      );
      print({ datasetId: ds.id, count: ds.observations.length, check:
        checkObservations(observations, 0) });
      break;
    }
    case "check": {
      const { values } = parseArgs({
        args: rest,
        options: { dataset: { type: "string" }, noise: { type: "string" } },
      });
      const ds = store.getDataset(Number(values.dataset));
      if (!ds) fail("数据集不存在");
      print(checkObservations(ds.observations, Number(values.noise ?? 0)));
      break;
    }
    case "analyze": {
      const { values } = parseArgs({
        args: rest,
        options: {
          dataset: { type: "string" },
          name: { type: "string" },
          sill: { type: "string" },
          range: { type: "string" },
          noise: { type: "string" },
        },
      });
      if (!values.dataset || !values.sill || !values.range)
        fail("需要 --dataset --sill --range [--noise]");
      const a = store.createAnalysis(
        Number(values.dataset),
        values.name ?? "cli-analysis",
        modelFromFlags({ sill: values.sill, range: values.range, noise: values.noise }),
      );
      print({ analysisId: a.id, model: a.model });
      break;
    }
    case "predict": {
      const { values } = parseArgs({
        args: rest,
        options: {
          analysis: { type: "string" },
          x: { type: "string" }, y: { type: "string" },
          target: { type: "string" },
        },
      });
      const a = store.getAnalysis(Number(values.analysis));
      if (!a) fail("分析不存在");
      const target: Point = { x: Number(values.x), y: Number(values.y) };
      if (![target.x, target.y].every(Number.isFinite)) fail("--x/--y 必须是有限数");
      const result = predictPoint(
        { observations: a.observations, model: a.model },
        target,
        values.target === "observation" ? "observation" : "latent",
      );
      store.saveReport(a.id, "point", { target, result });
      print(result);
      break;
    }
    case "grid": {
      const { values } = parseArgs({
        args: rest,
        options: {
          analysis: { type: "string" },
          "x-min": { type: "string" }, "x-max": { type: "string" },
          "y-min": { type: "string" }, "y-max": { type: "string" },
          nx: { type: "string" }, ny: { type: "string" },
          target: { type: "string" },
          out: { type: "string" },
        },
      });
      const a = store.getAnalysis(Number(values.analysis));
      if (!a) fail("分析不存在");
      const spec: GridSpec = {
        xMin: Number(values["x-min"]), xMax: Number(values["x-max"]),
        yMin: Number(values["y-min"]), yMax: Number(values["y-max"]),
        nx: Number(values.nx), ny: Number(values.ny),
      };
      const output = predictGrid(
        { observations: a.observations, model: a.model },
        spec,
        values.target === "observation" ? "observation" : "latent",
      );
      store.saveReport(a.id, "grid", { spec, output });
      if (values.out) {
        mkdirSync(dirname(resolve(values.out)), { recursive: true });
        writeFileSync(values.out, JSON.stringify(output, null, 2));
        print({ nodes: output.nodes.length, writtenTo: resolve(values.out),
          illCount: output.illCount });
      } else print(output);
      break;
    }
    case "loocv": {
      const { values } = parseArgs({
        args: rest,
        options: { analysis: { type: "string" } },
      });
      const a = store.getAnalysis(Number(values.analysis));
      if (!a) fail("分析不存在");
      const output = leaveOneOut({ observations: a.observations, model: a.model });
      store.saveReport(a.id, "loocv", output);
      print(output);
      break;
    }
    case "compare-params": {
      const { values } = parseArgs({
        args: rest,
        options: {
          dataset: { type: "string" },
          sill: { type: "string" },
          noise: { type: "string" },
          ranges: { type: "string" },
        },
      });
      const ds = store.getDataset(Number(values.dataset));
      if (!ds || !values.ranges) fail("需要 --dataset --ranges 1,2 [--sill --noise]");
      const sill = Number(values.sill ?? 1);
      const noise = Number(values.noise ?? 0);
      const out = values.ranges.split(",").map((r) => {
        const model: CovarianceModel = {
          sill, range: Number(r), measurementNoiseVariance: noise,
        };
        const loo = leaveOneOut({ observations: ds.observations, model });
        return { range: Number(r), loocv: loo };
      });
      print({ datasetId: ds.id, comparisons: out });
      break;
    }
    case "export": {
      const { values } = parseArgs({
        args: rest,
        options: { analysis: { type: "string" }, out: { type: "string" } },
      });
      const a = store.getAnalysis(Number(values.analysis));
      if (!a) fail("分析不存在");
      const payload = { analysis: a, reports: store.listReports(a.id) };
      if (values.out) {
        mkdirSync(dirname(resolve(values.out)), { recursive: true });
        writeFileSync(values.out, JSON.stringify(payload, null, 2));
        print({ writtenTo: resolve(values.out) });
      } else print(payload);
      break;
    }
    case "list": {
      print({ datasets: store.listDatasets(), analyses: store.listAnalyses() });
      break;
    }
    default:
      process.stdout.write(
        [
          "普通克里金后端 CLI（所有坐标必须使用同一长度单位，值域单位自定）",
          "命令:",
          "  import  --file obs.json [--name N] [--coord-unit U] [--value-unit U]",
          "  check   --dataset ID [--noise V]",
          "  analyze --dataset ID --sill S --range R [--noise N2] [--name N]",
          "  predict --analysis ID --x X --y Y [--target latent|observation]",
          "  grid    --analysis ID --x-min .. --x-max .. --y-min .. --y-max .. --nx .. --ny .. [--target ...] [--out f.json]",
          "  loocv   --analysis ID",
          "  compare-params --dataset ID --ranges 1,2 [--sill S] [--noise N]",
          "  export  --analysis ID [--out report.json]",
          "  list",
          "",
          "环境变量: KRIGING_DATA_DIR（默认 ./data）",
        ].join("\n") + "\n",
      );
  }
  store.close();
}

main().catch((e: Error) => fail(e.message));
