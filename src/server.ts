import { resolve } from "node:path";
import { createApp } from "./app.js";
import { Store } from "./store.js";

const port = Number(process.env.PORT ?? process.env.KRIGING_PORT ?? 3000);
if (!Number.isInteger(port) || port <= 0 || port > 65535) {
  throw new Error(`端口非法: ${process.env.PORT}`);
}
const dataDir = resolve(process.env.KRIGING_DATA_DIR ?? "data");
const dbPath = resolve(dataDir, "kriging.sqlite");

const store = new Store(dbPath);
const server = createApp({ store });

server.listen(port, () => {
  console.log(`[ordinary-kriging] 监听 http://localhost:${port}`);
  console.log(`[ordinary-kriging] 数据目录: ${dataDir}`);
  console.log("[ordinary-kriging] 仅后端 API，无地图/外部在线服务");
});

const shutdown = () => {
  server.close(() => {
    store.close();
    process.exit(0);
  });
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
