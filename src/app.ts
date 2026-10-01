import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import { URL } from "node:url";
import { checkObservations } from "./dataset.js";
import {
  leaveOneOut,
  predictGrid,
  predictPoint,
  type GridSpec,
} from "./kriging.js";
import { Store } from "./store.js";
import type { CovarianceModel, Observation, Point, PredictTarget } from "./types.js";

export interface AppDeps {
  store: Store;
}

function readJson(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      if (chunks.length === 0) return resolve({});
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch (e) {
        reject(new Error(`请求体不是合法 JSON: ${(e as Error).message}`));
      }
    });
    req.on("error", reject);
  });
}

function send(res: ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body, null, 2);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(json);
}

function asModel(raw: unknown): CovarianceModel {
  const m = raw as Partial<CovarianceModel>;
  return {
    sill: Number(m?.sill),
    range: Number(m?.range),
    measurementNoiseVariance: Number(m?.measurementNoiseVariance),
  };
}

export function createApp(deps: AppDeps): Server {
  const { store } = deps;

  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      const path = url.pathname.replace(/\/+$/, "") || "/";
      const method = req.method ?? "GET";

      if (method === "GET" && path === "/health") {
        return send(res, 200, { ok: true, service: "ordinary-kriging", dataLimits: {
          maxObservations: 400,
          maxGridNodes: 20000,
        } });
      }

      // ---- datasets ----
      if (method === "POST" && path === "/api/datasets") {
        const body = (await readJson(req)) as {
          name?: string;
          observations?: Observation[];
          coordinateUnit?: string;
          valueUnit?: string;
        };
        if (!Array.isArray(body.observations))
          return send(res, 400, { error: "observations 必须为数组" });
        const row = store.createDataset(
          body.name ?? `dataset-${Date.now()}`,
          body.observations,
          body.coordinateUnit,
          body.valueUnit,
        );
        const check = checkObservations(
          body.observations,
          0,
        );
        return send(res, 201, { dataset: row, check });
      }
      if (method === "GET" && path === "/api/datasets") {
        return send(res, 200, { datasets: store.listDatasets() });
      }
      let m = /^\/api\/datasets\/(\d+)$/.exec(path);
      if (method === "GET" && m) {
        const ds = store.getDataset(Number(m[1]));
        return ds
          ? send(res, 200, { dataset: ds })
          : send(res, 404, { error: "数据集不存在" });
      }
      m = /^\/api\/datasets\/(\d+)\/check$/.exec(path);
      if (method === "POST" && m) {
        const ds = store.getDataset(Number(m[1]));
        if (!ds) return send(res, 404, { error: "数据集不存在" });
        const body = (await readJson(req)) as { measurementNoiseVariance?: number };
        const nugget = Number(body.measurementNoiseVariance ?? 0);
        return send(res, 200, {
          check: checkObservations(ds.observations, Number.isFinite(nugget) ? nugget : 0),
        });
      }

      // ---- analyses (frozen model snapshot) ----
      if (method === "POST" && path === "/api/analyses") {
        const body = (await readJson(req)) as {
          datasetId?: number; name?: string; model?: unknown;
        };
        const analysis = store.createAnalysis(
          Number(body.datasetId),
          body.name ?? `analysis-${Date.now()}`,
          asModel(body.model),
        );
        return send(res, 201, { analysis });
      }
      if (method === "GET" && path === "/api/analyses") {
        return send(res, 200, { analyses: store.listAnalyses() });
      }
      m = /^\/api\/analyses\/(\d+)$/.exec(path);
      if (method === "GET" && m) {
        const a = store.getAnalysis(Number(m[1]));
        return a
          ? send(res, 200, { analysis: a })
          : send(res, 404, { error: "分析不存在" });
      }

      const loadAnalysis = (id: number) => store.getAnalysis(id);

      // ---- prediction ----
      m = /^\/api\/analyses\/(\d+)\/predict-point$/.exec(path);
      if (method === "POST" && m) {
        const a = loadAnalysis(Number(m[1]));
        if (!a) return send(res, 404, { error: "分析不存在" });
        const body = (await readJson(req)) as {
          target?: Point; predictWhat?: PredictTarget;
        };
        if (!body.target || ![body.target.x, body.target.y].every(Number.isFinite))
          return send(res, 400, { error: "target 必须包含有限数 x, y" });
        const result = predictPoint(
          { observations: a.observations, model: a.model },
          body.target,
          body.predictWhat === "observation" ? "observation" : "latent",
        );
        store.saveReport(a.id, "point", {
          target: body.target, predictWhat: result.predictWhat, result,
        });
        return send(res, 200, { result });
      }

      m = /^\/api\/analyses\/(\d+)\/predict-grid$/.exec(path);
      if (method === "POST" && m) {
        const a = loadAnalysis(Number(m[1]));
        if (!a) return send(res, 404, { error: "分析不存在" });
        const body = (await readJson(req)) as {
          grid?: GridSpec; predictWhat?: PredictTarget;
        };
        if (!body.grid) return send(res, 400, { error: "缺少 grid 参数" });
        const output = predictGrid(
          { observations: a.observations, model: a.model },
          body.grid,
          body.predictWhat === "observation" ? "observation" : "latent",
        );
        store.saveReport(a.id, "grid", {
          grid: body.grid, predictWhat: body.predictWhat ?? "latent", output,
        });
        return send(res, 200, { output });
      }

      // ---- cross validation ----
      m = /^\/api\/analyses\/(\d+)\/cross-validation$/.exec(path);
      if (method === "POST" && m) {
        const a = loadAnalysis(Number(m[1]));
        if (!a) return send(res, 404, { error: "分析不存在" });
        const output = leaveOneOut({
          observations: a.observations, model: a.model,
        });
        store.saveReport(a.id, "loocv", output);
        return send(res, 200, output);
      }

      // ---- reports & export ----
      if (method === "GET" && path === "/api/reports") {
        const aid = url.searchParams.get("analysisId");
        return send(res, 200, { reports: store.listReports(aid ? Number(aid) : undefined) });
      }
      m = /^\/api\/analyses\/(\d+)\/export$/.exec(path);
      if (method === "GET" && m) {
        const a = loadAnalysis(Number(m[1]));
        if (!a) return send(res, 404, { error: "分析不存在" });
        return send(res, 200, {
          analysis: a,
          reports: store.listReports(a.id),
        });
      }

      return send(res, 404, { error: `未找到路由 ${method} ${path}` });
    } catch (e) {
      return send(res, 400, { error: (e as Error).message });
    }
  });
}
