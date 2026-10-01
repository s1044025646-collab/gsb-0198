# 普通克里金空间插值与交叉验证后端（TypeScript + Node.js + SQLite）

Windows 本地可运行的**小规模二维测点**普通克里金（Ordinary Kriging, OK）实验服务。
只提供 **HTTP API** 与 **命令行（CLI）**，不含前端、地图或任何外部在线服务。
所有数据均为**虚构数据**；本服务不拟合真实地质参数，也不接入气象/GIS。

## 1. 模型口径（务必先读）

- 协方差模型：各向同性指数模型
  - `C(h) = psill * exp(-h / length)`
  - `psill`：基台（偏基台，>0）
  - `length`：相关长度（>0，与坐标同单位）
  - `nugget`：独立测量噪声方差（≥0）
- **噪声只加在观测协方差矩阵对角项**：观测对角为 `C(0)+nugget = psill+nugget`；
  目标点与观测的协方差右端项 `k` **不含 nugget**（目标不是某个带噪观测本身）。
- 预测目标分两种口径：
  - `latent`（默认）：预测**潜在场**，克里金方差 `σ² = C(0) − w′k − μ`。
  - `observation`：预测**新观测**，方差为 `σ²_latent + nugget`。
- 含噪测量**不会被当成必须精确穿过的真值**：`nugget>0` 时在观测位置也存在潜在场估计方差。

### 克里金方程组（自行组装，含无偏约束）

```
| K      1 | | w |   | k |
| 1'     0 | | μ | = | 1 |
```

- `K_ij = C(‖x_i−x_j‖)`，且 `K_ii = psill + nugget`
- `k_i  = C(‖x_i−x_0‖)`（目标点 x_0，不含噪声）
- 输出每个观测权重 `w`、拉格朗日乘子 `μ`、预测值 `ŷ = w′v`、方差与诊断。
- 权重恒满足 `Σw = 1`；**负权重不裁剪**（直接来自方程解），同时报告负权重数量。

### 数值可靠性

- 求解器为带部分选主元的 LU 分解（`src/solver.ts`），**不加入任何未声明的对角项**。
- 报告条件数估计（1-范数逆估计）、最小主元比、组装后 `A z − b` 的相对/最大残差、约束残差。
- **奇异**：直接报 `SINGULAR`（HTTP 422），不声称按原模型求解成功。
- **严重病态**：`rcond < 1e-8`、最小主元比过小或残差过大时标记 `illConditioned`。
- **负方差分级**：
  - 相对基台小于 `1e-9` 的负方差视为数值误差，钳制为 0 并标记 `clampedNegativeVariance`。
  - 相对基台小于约 `1e-6` 的负方差保留并提示；超过该量级标记
    `substantialNegativeVariance`（普通克里金在特定几何下可能出现小幅负估计方差，
    属已知现象，与求解失败区分）。

### 重复位置策略

- `nugget = 0` 且重合位置观测值**冲突**：**拒绝**（零噪声无法解释差异）。
- `nugget = 0` 且值一致：合并为一个测点。
- `nugget > 0`：等权（逆噪声方差）平均，聚合点独立噪声方差折减为 `nugget/k`（k 个重复）。

### 留一交叉验证（LOOCV）

- 逐点**真正删除**被预测测点（及其同位置副本，避免泄漏）后用其余点重训重算，
  **不用在原样点的自预测冒充验证**。
- 每折报告误差 `e = ŷ − y`、克里金方差、与观测同口径的预测方差、
  标准化残差 `e / √(σ²_latent+nugget)`。
- 汇总：ME、MAE、MSE、RMSE、标准化残差均值/RMSE、平均克里金方差、95% 覆盖率。

## 2. 数据量与单位约定

- 测点数量：**1..2000**；网格单元上限 **40000**（如 200×200）。
- 坐标与 `length` 必须使用**一致单位**（导入时用 `unitNote` 记录，如“米”）；
  `psill`、`nugget`、观测值的量纲需自洽。

## 3. 环境与运行

- 要求 **Node.js ≥ 22.5**（开发于 Node 25，使用内置实验模块 `node:sqlite`，无需原生编译）。
- 不依赖 Docker、WSL、付费服务或外部在线 API。

```powershell
npm install        # 安装开发依赖（typescript / tsx / @types/node）
npm run build      # 编译到 dist/
npm test           # 运行全部单元/数值测试
npm run demo       # 端到端演示（虚构数据、两种相关长度、失败输入、重启复现）

# HTTP 服务（端口与数据目录可配置）
npm start
# 或自定义：
$env:PORT="4000"; $env:KRIGING_DATA_DIR="E:\gsb-data"; npm start

# CLI（直接用 tsx 运行源码，也可在 build 后改跑 dist/cli.js）
npm run cli -- datasets
```

## 4. HTTP API

基址 `http://localhost:3000`（端口可用 `PORT` 或第一个参数覆盖）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | `/api/health` | 健康检查 |
| POST | `/api/datasets` | 导入数据集 `{name, unitNote, points:[{x,y,v}]}` |
| GET  | `/api/datasets` | 列出数据集 |
| GET  | `/api/datasets/:id` | 数据集详情（含测点） |
| POST | `/api/datasets/:id/inspect` | 数据检查（重复点策略、包围盒），body 给模型参数 |
| POST | `/api/datasets/:id/models` | 保存模型快照 `{psill,length,nugget,label}` |
| GET  | `/api/datasets/:id/models` | 列出模型快照 |
| POST | `/api/datasets/:id/models/:mid/predict` | 单点预测 `{x,y,target}` |
| POST | `/api/datasets/:id/models/:mid/grid` | 网格预测 `{grid:{xmin,...},target}` |
| POST | `/api/datasets/:id/models/:mid/loocv` | 留一交叉验证 |
| POST | `/api/datasets/:id/compare` | 多组参数对照 `{models:[{psill,length,nugget}, ...]}` |
| GET  | `/api/reports?datasetId=` | 列出报告 |
| GET  | `/api/reports/:rid` | 读取已存报告（含绑定指纹与一致性标记） |

错误：`400 INVALID_INPUT`、`409 DUPLICATE_CONFLICT`、`422 SINGULAR/ILL_CONDITIONED`。

### curl 示例

```powershell
Invoke-RestMethod -Uri http://localhost:3000/api/datasets -Method Post `
  -ContentType 'application/json' `
  -Body (@{name="demo"; unitNote="m"; points=(Get-Content data/sample-points.json -Raw | ConvertFrom-Json)} | ConvertTo-Json -Depth 5)
```

## 5. CLI

```powershell
npm run cli -- init-db
npm run cli -- import --file data/sample-points.json --name "样例" --unit "米"
npm run cli -- inspect --dataset 1 --psill 1 --length 3 --nugget 0
npm run cli -- add-model --dataset 1 --psill 1 --length 3 --nugget 0.1
npm run cli -- predict --dataset 1 --model 1 --x 4 --y 4 --target observation
npm run cli -- grid --dataset 1 --model 1 --xmin 0 --ymin 0 --xmax 8 --ymax 8 --nx 5 --ny 5
npm run cli -- loocv --dataset 1 --model 1
npm run cli -- compare --dataset 1 --specs '[{"psill":1,"length":1.5,"nugget":0.1},{"psill":1,"length":8,"nugget":0.1}]'
npm run cli -- reports
npm run cli -- export --report 1 --out exports/r1.json
```

默认数据目录为项目内 `./data`，可用 `--data-dir` 覆盖。

## 6. SQLite 持久化与一致性

- 表：`datasets / points / models / reports`（见 `src/db.ts`）。
- 数据集内容生成确定性**测点指纹**；每份报告外绑数据集 id、模型 id、模型参数、
  测点数量与指纹，并记录 `consistency.samePointVersion`，从而**同一分析不混用两版测点**，
  关闭进程重启后可读取报告原样复现（见 `tests/persistence.test.ts` 与 demo）。

## 7. 目录结构

```
src/
  types.ts       类型与错误码
  covariance.ts  指数协方差、克里金矩阵/右端组装
  solver.ts      LU 求解、条件数估计、残差校验
  data.ts        输入校验、重复点策略
  kriging.ts     单点/网格预测、LOOCV
  db.ts/store.ts SQLite 与报告快照
  server.ts/cli.ts
tests/           数值与持久化测试（含手算小系统、平移/旋转不变性、删除重算一致性）
scripts/demo.ts  端到端演示
data/sample-points.json  虚构样例
```

## 8. 已知限制与合理假设

- 仅支持固定的各向同性指数协方差，参数由使用者给定，不做参数拟合/变异函数估计。
- 面向小规模实验（≤2000 点、网格 ≤40000）；稠密求解复杂度约 O(n³)，不针对大规模数据优化。
- 条件数为估计量；极端近重复点会被标记病态而非自动“加噪修好”。
- 普通克里金不估计全局均值的不确定性之外的趋势；不支持各向异性、漂移或其他协方差族。
