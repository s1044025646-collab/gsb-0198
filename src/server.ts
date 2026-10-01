import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
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
import { prepareData, validateRaw } from "./data.js";
import { leaveOneOut, predictGrid, predictPoint } from "./kriging.js";
import { validateModel } from "./covariance.js";
import { KrigingError, type ModelParams, type Point } from "./types.js";

interface Context {
  params: Record<string, string>;
  query: Record<string, string>;
  body: any;
}

type Handler = (ctx: Context) => unknown | Promise<unknown>;

interface Route {
  method: string;
  pattern: RegExp;
  keys: string[];
  handler: Handler;
}

function route(method: string, path: string, handler: Handler): Route {
  const keys: string[] = [];
  const pattern = new RegExp(
    "^" +
      path.replace(/:([a-zA-Z]+)/g, (_m, k) => {
        keys.push(k);
        return "([^/]+)";
      }) +
      "$",
  );
  return { method, pattern, keys, handler };
}

function parseParams(body: any): ModelParams {
  return {
    psill: Number(body?.psill),
    length: Number(body?.length),
    nugget: Number(body?.nugget ?? 0),
  };
}

function loadAnalysis(datasetId: number, modelId: number) {
  const ds = getDataset(datasetId);
  if (!ds) throw new KrigingError("INVALID_INPUT", `数据集 ${datasetId} 不存在`);
  const model = getModel(modelId);
  if (!model || model.datasetId !== datasetId) {
    throw new KrigingError(
      "INVALID_INPUT",
      `模型 ${modelId} 不属于数据集 ${datasetId}，禁止混用两版测点`,
    );
  }
  validateModel(model.params);
  return { ds, model };
}

const routes: Route[] = [
  route("GET", "/api/health", () => ({ status: "ok" })),
  route("POST", "/api/datasets", (ctx) => {
    const body = ctx.body ?? {};
    const points = validateRaw(body.points);
    const name = String(body.name ?? "未命名数据集");
    const unitNote = String(body.unitNote ?? "");
    const rec = createDataset(name, points, unitNote);
    const { points: _p, ...meta } = rec;
    return { ...meta };
  }),
  route("GET", "/api/datasets", () => ({ datasets: listDatasets() })),
  route("GET", "/api/datasets/:id", (ctx) => {
    const ds = getDataset(Number(ctx.params.id));
    if (!ds) throw new KrigingError("INVALID_INPUT", "数据集不存在");
    return ds;
  }),
  route("POST", "/api/datasets/:id/inspect", (ctx) => {
    const ds = getDataset(Number(ctx.params.id));
    if (!ds) throw new KrigingError("INVALID_INPUT", "数据集不存在");
    const model = parseParams(ctx.body);
    validateModel(model);
    const prepared = prepareData(ds.points, model);
    return {
      pointCountRaw: ds.points.length,
      pointCountAfterDedup: prepared.points.length,
      issues: prepared.issues,
      bounds: boundsOf(ds.points),
    };
  }),
  route("POST", "/api/datasets/:id/models", (ctx) => {
    const datasetId = Number(ctx.params.id);
    if (!getDataset(datasetId))
      throw new KrigingError("INVALID_INPUT", "数据集不存在");
    const params = parseParams(ctx.body);
    validateModel(params);
    const label = String(ctx.body?.label ?? "");
    return createModel(datasetId, params, label);
  }),
  route("GET", "/api/datasets/:id/models", (ctx) => ({
    models: listModels(Number(ctx.params.id)),
  })),
  route("POST", "/api/datasets/:id/models/:mid/predict", (ctx) => {
    const { ds, model } = loadAnalysis(
      Number(ctx.params.id),
      Number(ctx.params.mid),
    );
    const x = Number(ctx.body?.x);
    const y = Number(ctx.body?.y);
    if (![x, y].every(Number.isFinite))
      throw new KrigingError("INVALID_INPUT", "x、y 必须为有限数值");
    const kind = ctx.body?.target === "observation" ? "observation" : "latent";
    const result = predictPoint(ds.points, model.params, { x, y }, kind);
    const reportId = saveReport("predict", ds.id, model.id, result);
    return { reportId, ...result };
  }),
  route("POST", "/api/datasets/:id/models/:mid/grid", (ctx) => {
    const { ds, model } = loadAnalysis(
      Number(ctx.params.id),
      Number(ctx.params.mid),
    );
    const g = ctx.body?.grid ?? {};
    const kind = ctx.body?.target === "observation" ? "observation" : "latent";
    const result = predictGrid(ds.points, model.params, g, kind);
    const reportId = saveReport("grid", ds.id, model.id, result);
    return { reportId, ...result };
  }),
  route("POST", "/api/datasets/:id/models/:mid/loocv", (ctx) => {
    const { ds, model } = loadAnalysis(
      Number(ctx.params.id),
      Number(ctx.params.mid),
    );
    const result = leaveOneOut(ds.points, model.params);
    const reportId = saveReport("loocv", ds.id, model.id, result);
    return { reportId, ...result };
  }),
  route("POST", "/api/datasets/:id/compare", (ctx) => {
    const ds = getDataset(Number(ctx.params.id));
    if (!ds) throw new KrigingError("INVALID_INPUT", "数据集不存在");
    const specs: ModelParams[] = Array.isArray(ctx.body?.models)
      ? ctx.body.models.map(parseParams)
      : [];
    if (specs.length < 2)
      throw new KrigingError("INVALID_INPUT", "compare 需提供至少两组模型参数");
    specs.forEach(validateModel);
    const out = specs.map((params, i) => {
      const loo = leaveOneOut(ds.points, params);
      return { index: i, model: params, summary: loo.summary };
    });
    return { datasetId: ds.id, comparisons: out };
  }),
  route("GET", "/api/reports", (ctx) => ({
    reports: listReports(
      ctx.query.datasetId ? Number(ctx.query.datasetId) : undefined,
    ),
  })),
  route("GET", "/api/reports/:rid", (ctx) => {
    const r = getReport(Number(ctx.params.rid));
    if (!r) throw new KrigingError("INVALID_INPUT", "报告不存在");
    return r;
  }),
];

function boundsOf(points: Point[]) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    xmin: Math.min(...xs),
    xmax: Math.max(...xs),
    ymin: Math.min(...ys),
    ymax: Math.max(...ys),
  };
}

async function readBody(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (chunks.length === 0) return {};
  const text = Buffer.concat(chunks).toString("utf8");
  try {
    return JSON.parse(text);
  } catch {
    throw new KrigingError("INVALID_INPUT", "请求体不是合法 JSON");
  }
}

export function createApp() {
  return createServer(async (req: IncomingMessage, res: ServerResponse) => {
    const send = (status: number, payload: unknown) => {
      res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
      res.end(JSON.stringify(payload, null, 2));
    };
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      const path = url.pathname;
      if (req.method === "GET" && (path === "/" || path === "/api")) {
        return send(200, {
          service: "ordinary-kriging-backend",
          endpoints: routes.map((r) => `${r.method} ${r.pattern.source.replace(/\\\^|\\\$|\(\[\^\/\]\+\)/g, ":x")}`),
        });
      }
      const body = ["POST", "PUT", "PATCH"].includes(req.method ?? "")
        ? await readBody(req)
        : {};
      for (const r of routes) {
        if (r.method !== req.method) continue;
        const m = r.pattern.exec(path);
        if (!m) continue;
        const params: Record<string, string> = {};
        r.keys.forEach((k, i) => (params[k] = decodeURIComponent(m[i + 1])));
        const query: Record<string, string> = {};
        url.searchParams.forEach((v, k) => (query[k] = v));
        const data = await r.handler({ params, body, query });
        return send(200, data);
      }
      send(404, { error: "NOT_FOUND", message: `无此路由：${req.method} ${path}` });
    } catch (err) {
      if (err instanceof KrigingError) {
        const status =
          err.code === "SINGULAR" || err.code === "ILL_CONDITIONED"
            ? 422
            : err.code === "DUPLICATE_CONFLICT"
              ? 409
              : 400;
        return send(status, { error: err.code, message: err.message });
      }
      const e = err as Error;
      send(500, { error: "INTERNAL", message: e.message });
    }
  });
}

function main() {
  const port = Number(process.env.PORT ?? process.argv[2] ?? 3000);
  const dataDir = process.env.KRIGING_DATA_DIR
    ? resolve(process.env.KRIGING_DATA_DIR)
    : resolve(process.cwd(), "data");
  openDatabase(resolve(dataDir, "kriging.db"));
  const server = createApp();
  server.listen(port, () => {
    console.log(`普通克里金后端已启动: http://localhost:${port}`);
    console.log(`数据目录: ${dataDir}`);
  });
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
