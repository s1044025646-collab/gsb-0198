# 普通克里金空间插值与交叉验证后端（TypeScript + Node.js + SQLite）

Windows 本地可运行的**小规模二维测点普通克里金（Ordinary Kriging）实验服务**。
只提供 **HTTP API 与 CLI**，无前端、无地图、不连接任何在线气象/GIS/付费服务，
不拟合真实地质参数，所有演示数据均为**虚构数据**。

## 1. 范围与重要约定

- 输入：二维坐标 `(x, y)` 与标量观测值 `value`。
- 协方差模型：**固定、各向同性指数模型**
  `C(h) = sill · exp(-h / range)`，其中 `range` 为 1/e 相关长度。
- 观测模型：`z_i = S(x_i) + eps_i`，`eps_i ~ N(0, nugget)`，
  `nugget = measurementNoiseVariance` 为所有观测共享的**独立测量噪声方差**。
- 单位要求：**x、y 必须使用同一长度单位**，`range` 也使用该长度单位；
  观测值单位自定且全程一致。本程序不做任何单位换算。
- 规模上限（稠密线性求解器）：
  - 每个数据集最多 **400 个测点**；
  - 单次网格预测最多 **20000 个节点**（`nx*ny`）。
- 预测目标二选一（必须显式区分）：
  - `latent`：潜在场 `S(x0)`，方差基线为 `sill`；
  - `observation`：在 `x0` 处的一次**新观测** `S(x0)+eps0`，
    方差基线为 `sill + nugget`（新噪声 `eps0` 独立于所有历史 `eps_i`）。

> 含义上：含噪测量不会被当作必须精确穿过的“真值”。噪声只加在观测协方差
> 块 `C` 的对角线上；目标点与观测点的交叉协方差 `k` **不含**噪声项。

## 2. 克里金方程（自行组装，无外部数值库）

普通克里金要求权重无偏（权重和为 1）并最小化预测误差方差。组装增广方程组：

```
[ C    1 ] [w ]   [ k ]
[ 1ᵀ   0 ] [μ] = [ 1 ]
```

- `C_ij = Cov(z_i, z_j) = C(‖x_i-x_j‖)`，仅在 `i=j` 对角加 `nugget`；
- `k_i  = Cov(S(x0), z_i) = C(‖x0-x_i‖)`（不含噪声）；
- 解出各观测权重 `w_i` 与拉格朗日乘子 `μ`；
- 预测值 `ŷ = Σ w_i z_i`，恒有 `Σ w_i = 1`；
- 方差 `σ² = c00 − wᵀk − μ`。

实现采用部分选主元高斯消元（`src/matrix.ts`），并报告：

- 线性方程残差 `‖Aw−b‖∞` 的绝对与相对值；
- 无偏约束残差 `|Σw_i − 1|`；
- 1-范数条件数估计（Hager/Higham 迭代）与秩；
- **负权重不裁剪**，原样输出（屏蔽效应下负权重是正常现象）。

### 健康度与负方差策略

- 主元接近机器零 / 秩不足：`health = singular`，**明确报告**，
  不会偷偷加入未声明的对角项后声称仍按原模型求解。
- 条件数估计 > `1e10`：`health = ill_conditioned`，给出警告，结果仍返回但标注不可靠。
- 计算方差出现 **`-1e-8 < σ² < 0`** 的舍入量级微小负值：记为
  `clamped_tiny_negative` 并夹到 0；
  若为 **`σ² ≤ -1e-8`** 的实质性负值：记为 `negative_refused`，
  方差返回 `NaN` 并给出明确消息，二者严格区分。

### 重复位置策略（显式，不静默丢弃）

- **零噪声 + 同位置不同值**：直接拒绝（无法让无插值误差同时穿过冲突真值）。
- **零噪声 + 同位置同值**：合并为一个有效测点（重复行只会使 KKT 奇异而不增信息）。
- **有噪声 + 同位置多次读数**：按逆噪声方差合并为一个 BLUE 有效读数
  `v* = Σ(v_i/τ²)/Σ(1/τ²)`，有效噪声方差 `τ²*/k`（等噪声即均值、方差 `τ²/k`）。
  合并权重在输出中同时给出“有效点权重”和映射回原始测点的份额。

## 3. 留一交叉验证（LOOCV）

`POST /api/analyses/:id/cross-validation`（CLI：`loocv`）对每个测点：

1. **真正剔除该测点**，用其余点重新组装并求解；
2. 在被剔除点位置预测一次**新观测**（`observation`，含独立噪声方差）；
3. 报告误差 `e = 观测值 − 预测值`、标准化残差 `e/σ`、方程与约束残差、健康度；
4. 汇总：平均误差（偏差）、MAE、RMSE、标准化残差均方（理想 ≈1）、
   平均标准化残差（理想 ≈0）、95% 经验覆盖率。

这不是“用全部点在原样点自预测”的伪验证；测试中还逐折比对了
“留一结果”与“独立删除该点后重算”的结果完全一致。少于 2 个测点时明确报为不可行。

## 4. 数据与模型快照（不混用两版测点）

SQLite（Node 内置 `node:sqlite`，无需本地编译）包含三张表：

- `datasets`：原始测点；
- `analyses`：创建时**冻结**测点 JSON 与模型 JSON 快照；
- `reports`：每次单点/网格/留一结果（含方程残差与约束残差），供导出与重启复现。

已创建分析引用的是快照，数据集后续变更不会污染历史分析。

## 5. 环境要求

- Windows + Node.js **>= 22.5**（开发与验证使用 Node 25.5；使用内置 `node:sqlite`）。
- 仅开发依赖 `typescript` 与 `@types/node`，运行时零第三方依赖。
- 不需要 Docker / WSL / 付费服务 / 外部在线 API。

## 6. 构建、测试、启动、演示

```powershell
# 安装依赖
npm install

# 编译 TypeScript -> dist/
npm run build

# 运行全部测试（23 个：含手算核对、不变性、留一一致性、持久化、HTTP 全链路）
npm test

# 启动 HTTP 服务（默认端口 3000，数据目录 ./data）
npm start

# 运行端到端演示（两种相关长度网格+留一、病态输入、拒绝冲突、重启复现）
npm run demo
```

可配置项（环境变量）：

- `PORT` 或 `KRIGING_PORT`：HTTP 端口（默认 `3000`）；
- `KRIGING_DATA_DIR`：项目内数据目录（默认 `./data`），SQLite 文件位于其下。

```powershell
$env:PORT="8080"; $env:KRIGING_DATA_DIR=".\data"; npm start
```

## 7. HTTP API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查与规模上限 |
| POST | `/api/datasets` | 导入测点 |
| GET | `/api/datasets`、`/api/datasets/:id` | 列出/查看数据集 |
| POST | `/api/datasets/:id/check` | 数据检查（可带 `measurementNoiseVariance`） |
| POST | `/api/analyses` | 用 `{datasetId, model}` 创建冻结分析 |
| GET | `/api/analyses`、`/api/analyses/:id` | 列出/查看分析 |
| POST | `/api/analyses/:id/predict-point` | `{target:{x,y}, predictWhat}` 单点预测 |
| POST | `/api/analyses/:id/predict-grid` | `{grid:{xMin,xMax,yMin,yMax,nx,ny}, predictWhat}` |
| POST | `/api/analyses/:id/cross-validation` | 逐点留一交叉验证 |
| GET | `/api/analyses/:id/export` | 导出分析快照与全部报告 |
| GET | `/api/reports?analysisId=` | 查询历史报告 |

单点结果包含：每个（及每个有效）观测权重、`weightSum`、`lagrangeMultiplier`、
`prediction`、`varianceRaw`/`variance`/`varianceStdDev`、`varianceNote`、
`equationResidualInf/Rel`、`constraintResidual`、`condition1Estimate`、`rank`、
`health`、`messages`、重复点 `preparationNotes`。

### API 调用示例（PowerShell）

```powershell
$body = @'
{ "name":"demo",
  "observations":[
    {"x":0,"y":0,"value":1},
    {"x":1,"y":0,"value":0},
    {"x":0,"y":1,"value":2} ] }
'@
$ds = Invoke-RestMethod -Uri http://localhost:3000/api/datasets -Method Post -ContentType 'application/json' -Body $body
$an = Invoke-RestMethod -Uri http://localhost:3000/api/analyses -Method Post -ContentType 'application/json' `
  -Body (@{ datasetId = $ds.dataset.id; model = @{ sill = 1; range = 1; measurementNoiseVariance = 0.1 } } | ConvertTo-Json)
Invoke-RestMethod -Uri "http://localhost:3000/api/analyses/$($an.analysis.id)/predict-point" -Method Post `
  -ContentType 'application/json' -Body (@{ target = @{x=0.5;y=0.5}; predictWhat='latent' } | ConvertTo-Json)
Invoke-RestMethod -Uri "http://localhost:3000/api/analyses/$($an.analysis.id)/cross-validation" -Method Post
```

## 8. CLI 用法

```powershell
# 查看帮助与全部命令
npm run cli

# 导入示例虚构数据并检查
node dist/src/cli.js import --file examples/observations.example.json --name demo
node dist/src/cli.js list
node dist/src/cli.js check --dataset 1 --noise 0.04

# 创建分析（基台 sill=1，相关长度 range=1.5，噪声方差 0.04）
node dist/src/cli.js analyze --dataset 1 --sill 1 --range 1.5 --noise 0.04

# 单点预测（潜在场 / 新观测）
node dist/src/cli.js predict --analysis 1 --x 3 --y 2.5 --target latent
node dist/src/cli.js predict --analysis 1 --x 3 --y 2.5 --target observation

# 网格预测（可导出 JSON 文件）
node dist/src/cli.js grid --analysis 1 --x-min 0 --x-max 6 --y-min 0 --y-max 5 --nx 31 --ny 26 --out data/grid.json

# 逐点留一交叉验证
node dist/src/cli.js loocv --analysis 1

# 参数对照：同一数据集在多个相关长度下的留一表现
node dist/src/cli.js compare-params --dataset 1 --sill 1 --noise 0.04 --ranges 0.8,1.5,4

# 导出分析快照与报告（含方程残差/约束残差）
node dist/src/cli.js export --analysis 1 --out data/report.json
```

## 9. 测试与独立核对覆盖

`test/kriging.test.ts`、`test/store.app.test.ts` 覆盖：

- 线性求解器正确性与奇异矩阵识别（不加未声明对角项）；
- **单测点手算核对**：权重、乘子、预测、方差（含新观测噪声口径）；
- **对称双测点手算核对**：权重各 0.5、乘子、方差，并用独立 MSE 展开
  `c00 − 2wᵀk + wᵀCw` 交叉验证；
- 常量场、零噪精确穿点、非零噪声不强制穿过；
- 负权重不裁剪；重复位置（冲突拒绝 / 同值去重 / 有噪合并）；
- 观测值整体加常数预测等量平移；坐标整体平移与旋转不改变各向同性结果；
- 留一逐折与“独立删除后重算”一致，且留一折方差严格为正（排除自预测冒充）；
- 严重病态布局被标记；SQLite 重启后快照与报告逐值复现；
- HTTP 全链路与零噪冲突重复点返回 400。

## 10. 已知限制与合理假设

- 稠密求解器面向小规模实验（≤400 测点 / ≤20000 网格点），非大规模生产系统。
- 模型参数（基台、相关长度、噪声方差）由调用方给定，**不做参数拟合/估计**。
- 仅支持二维、各向同性指数协方差；不支持各向异性、块段克里金、趋势项。
- 条件数为 1-范数估计（非精确 SVD）；足以区分健康/严重病态，并非高精度数值证书。
- 单测点且距离很远时，普通克里金（均值未知）方差趋于 `2·sill`，
  与“已知均值的简单克里金”不同，这是方法本身的正确行为。
