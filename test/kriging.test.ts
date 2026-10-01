import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { checkObservations } from "../src/dataset.js";
import { solveLinear } from "../src/matrix.js";
import {
  leaveOneOut,
  predictPoint,
  type KrigingInput,
} from "../src/kriging.js";
import type { CovarianceModel, Observation } from "../src/types.js";

const model: CovarianceModel = {
  sill: 1,
  range: 1,
  measurementNoiseVariance: 0,
};

function obs(data: Array<[number, number, number]>): Observation[] {
  return data.map(([x, y, value]) => ({ x, y, value }));
}

describe("线性求解器", () => {
  it("准确求解小型对称正定系统", () => {
    const A = [
      [2, 1],
      [1, 3],
    ];
    const b = [3, 4];
    const r = solveLinear(A, b);
    assert.ok(!r.singular);
    assert.ok(Math.abs(r.x[0]! - 1) < 1e-12);
    assert.ok(Math.abs(r.x[1]! - 1) < 1e-12);
    assert.ok(r.residualRel < 1e-14);
  });

  it("对奇异矩阵报告 singular，而不是静默加对角项", () => {
    const A = [
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
    ];
    const r = solveLinear(A, [1, 2, 3]);
    assert.equal(r.singular, true);
    assert.ok(r.rank < 3);
  });
});

describe("单测点手算核对", () => {
  const input: KrigingInput = {
    observations: obs([[0, 0, 5]]),
    model,
  };

  it("权重=1，单点 mu = C(h) - sill（<=0），方差 = 2(sill - C(h))", () => {
    const r = predictPoint(input, { x: 2, y: 0 }, "latent");
    assert.equal(r.health, "ok");
    assert.ok(Math.abs(r.weights[0]!.weight - 1) < 1e-12);
    assert.ok(Math.abs(r.weightSum - 1) < 1e-12);
    // mu = k - C = e^{-2} - 1 under the symmetric KKT block convention.
    assert.ok(Math.abs(r.lagrangeMultiplier - (Math.exp(-2) - 1)) < 1e-12);
    assert.ok(Math.abs(r.prediction - 5) < 1e-12);
    // With an unknown constant mean, a single reading carries weight 1 and
    // the OK variance is 2*(sill - C(h)) (verified by direct MSE expansion).
    const expected = 2 * (1 - Math.exp(-2));
    assert.ok(Math.abs(r.variance - expected) < 1e-12);
    assert.ok(r.constraintResidual < 1e-12);
    assert.ok(r.equationResidualRel < 1e-12);
  });

  it("预测新观测时方差额外包含测量噪声方差", () => {
    const noisy: KrigingInput = {
      observations: obs([[0, 0, 5]]),
      model: { ...model, measurementNoiseVariance: 0.25 },
    };
    const latent = predictPoint(noisy, { x: 0, y: 0 }, "latent");
    const fresh = predictPoint(noisy, { x: 0, y: 0 }, "observation");
    // At the same location w = 1 (nearest reading), mu = -nugget:
    // latent MSE equals the nugget ...
    assert.ok(Math.abs(latent.variance - 0.25) < 1e-12);
    // ... while predicting a NEW noisy observation has measured-vs-measured
    // MSE 2*nugget (two independent epsilons).
    assert.ok(Math.abs(fresh.variance - 0.5) < 1e-12);
  });
});

describe("对称双测点手算核对", () => {
  // 测点 p1=(-1,0), p2=(1,0)，查询原点；range=1，sill=1，无噪声。
  // C11=C22=1，C12=e^{-2}=q。对称性给出 w1=w2=1/2，mu = (1+q)/2 - 1。
  const q = Math.exp(-2);
  const input: KrigingInput = {
    observations: obs([
      [-1, 0, 3],
      [1, 0, 7],
    ]),
    model,
  };

  it("权重各 0.5，拉格朗日乘子与方差符合手算值", () => {
    const r = predictPoint(input, { x: 0, y: 0 }, "latent");
    assert.ok(Math.abs(r.weights[0]!.weight - 0.5) < 1e-12);
    assert.ok(Math.abs(r.weights[1]!.weight - 0.5) < 1e-12);
    assert.ok(Math.abs(r.prediction - 5) < 1e-12);
    // C w + mu = k: 0.5(1+q) + mu = e^{-1} => mu = e^{-1} - 0.5(1+q).
    const mu = Math.exp(-1) - (1 + q) / 2;
    assert.ok(Math.abs(r.lagrangeMultiplier - mu) < 1e-12);
    const k = Math.exp(-1);
    // Independent MSE check (unknown constant mean, symmetric w = 0.5):
    //   C00 - 2 w^T k + w^T C w
    const wCw = 0.25 * 1 + 0.5 * q + 0.25 * 1;
    const expectedVar = 1 - 2 * (0.5 * k + 0.5 * k) + wCw;
    assert.ok(Math.abs(r.variance - expectedVar) < 1e-12);
    // And this must equal the KKT identity C00 - w^T k - mu.
    assert.ok(Math.abs(r.variance - (1 - k - mu)) < 1e-12);
    assert.ok(r.variance >= 0);
  });
});

describe("常量场与精确插值", () => {
  const points: Observation[] = obs([
    [0, 0, 4],
    [1, 0, 4],
    [0, 1, 4],
    [2, 2, 4],
  ]);

  it("常量观测场在任意查询点预测为常量", () => {
    const r = predictPoint({ observations: points, model }, { x: 5, y: -3 });
    assert.ok(Math.abs(r.prediction - 4) < 1e-10);
    assert.ok(Math.abs(r.weightSum - 1) < 1e-10);
  });

  it("零噪声时在样点位置精确穿过观测值且方差为 0", () => {
    const r = predictPoint({ observations: points, model }, { x: 1, y: 0 });
    assert.ok(Math.abs(r.prediction - 4) < 1e-10);
    assert.ok(Math.abs(r.variance) < 1e-10);
  });

  it("非零噪声时样点处不再强制精确穿过，潜在场预测向邻点收缩", () => {
    const data: Observation[] = obs([
      [0, 0, 0],
      [0.0001, 0, 10],
    ]);
    const m2: CovarianceModel = { sill: 1, range: 1, measurementNoiseVariance: 1 };
    const r = predictPoint(
      { observations: data, model: m2 },
      { x: 0, y: 0 },
      "latent",
    );
    // 两个几乎重合、噪声极大的测点应接近各给 0.5 权重，预测≈5，不精确穿过 0。
    assert.ok(Math.abs(r.weights[0]!.weight - 0.5) < 1e-3);
    assert.ok(Math.abs(r.prediction - 5) < 1e-2);
    assert.ok(r.variance > 0);
  });
});

describe("负权重不裁剪", () => {
  it("屏蔽效应下允许出现负权重并如实报告", () => {
    const m2: CovarianceModel = { sill: 1, range: 2, measurementNoiseVariance: 0 };
    const data: Observation[] = obs([
      [0, 0, 1],
      [1, 0, 2],
      [0.5, 0.001, 3],
      [2, 0, 1],
      [1, 1, 0],
    ]);
    const r = predictPoint(
      { observations: data, model: m2 },
      { x: -1.47, y: 2.11 },
    );
    assert.ok(r.weights.some((w) => w.weight < 0));
    assert.ok(Math.abs(r.weightSum - 1) < 1e-10);
    assert.ok(r.variance >= 0);
  });
});

describe("平移/旋转不变性与观测平移", () => {
  const data: Observation[] = obs([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 2],
    [2, 3, -1],
  ]);

  it("坐标整体平移不改变各向同性预测与方差", () => {
    const a = predictPoint({ observations: data, model }, { x: 1.5, y: 1.5 });
    const shifted = data.map((o) => ({ ...o, x: o.x + 100, y: o.y - 50 }));
    const b = predictPoint(
      { observations: shifted, model },
      { x: 101.5, y: -48.5 },
    );
    assert.ok(Math.abs(a.prediction - b.prediction) < 1e-10);
    assert.ok(Math.abs(a.variance - b.variance) < 1e-10);
  });

  it("坐标旋转不改变各向同性预测与方差", () => {
    const c = Math.cos(0.7), s = Math.sin(0.7);
    const rot = data.map((o) => ({
      ...o,
      x: c * o.x - s * o.y,
      y: s * o.x + c * o.y,
    }));
    const t0 = { x: 1.5, y: 1.5 };
    const a = predictPoint({ observations: data, model }, t0);
    const b = predictPoint(
      { observations: rot, model },
      { x: c * t0.x - s * t0.y, y: s * t0.x + c * t0.y },
    );
    assert.ok(Math.abs(a.prediction - b.prediction) < 1e-10);
    assert.ok(Math.abs(a.variance - b.variance) < 1e-10);
  });

  it("观测值整体加常数 c，预测等量加 c，方差不变", () => {
    const target = { x: 1.2, y: 0.7 };
    const a = predictPoint({ observations: data, model }, target);
    const shifted = data.map((o) => ({ ...o, value: o.value + 37 }));
    const b = predictPoint({ observations: shifted, model }, target);
    assert.ok(Math.abs(b.prediction - (a.prediction + 37)) < 1e-10);
    assert.ok(Math.abs(a.variance - b.variance) < 1e-12);
  });
});

describe("重复位置策略", () => {
  it("零噪声且值不同的重合测点被拒绝", () => {
    const data: Observation[] = obs([
      [0, 0, 1],
      [0, 0, 2],
    ]);
    const check = checkObservations(data, 0);
    assert.ok(check.errors.length > 0);
    assert.throws(() => predictPoint({ observations: data, model }, { x: 1, y: 1 }));
  });

  it("零噪声完全相同的重合测点保留并正常工作", () => {
    const data: Observation[] = obs([
      [0, 0, 3],
      [0, 0, 3],
      [1, 0, 4],
    ]);
    const check = checkObservations(data, 0);
    assert.equal(check.errors.length, 0);
    const r = predictPoint({ observations: data, model }, { x: 0.5, y: 0 });
    assert.equal(r.health, "ok");
    assert.ok(Math.abs(r.weightSum - 1) < 1e-10);
  });

  it("有噪声时同位置不同值允许，各自通过对角噪声区分", () => {
    const data: Observation[] = obs([
      [0, 0, 1],
      [0, 0, 2],
    ]);
    const m2: CovarianceModel = { sill: 1, range: 1, measurementNoiseVariance: 0.5 };
    const check = checkObservations(data, 0.5);
    assert.equal(check.errors.length, 0);
    const r = predictPoint({ observations: data, model: m2 }, { x: 0, y: 0 });
    assert.equal(r.health, "ok");
    assert.ok(Math.abs(r.prediction - 1.5) < 1e-9);
  });
});

describe("留一交叉验证", () => {
  const data: Observation[] = obs([
    [0, 0, 1],
    [1, 0, 0.5],
    [0, 1, 2],
    [2, 2, -1],
    [3, 1, 0],
  ]);

  it("每个折确实排除该点，与独立删除后重算一致", () => {
    const out = leaveOneOut({ observations: data, model });
    assert.equal(out.feasible, true);
    assert.equal(out.folds.length, data.length);
    for (const fold of out.folds!) {
      const training = data.filter((_, j) => j !== fold.index);
      const manual = predictPoint(
        { observations: training, model },
        { x: fold.heldOut.x, y: fold.heldOut.y },
        "observation",
      );
      assert.ok(Math.abs(manual.prediction - fold.prediction) < 1e-10);
      assert.ok(Math.abs(manual.variance - fold.variance) < 1e-10);
      // 自预测冒充检查：零噪声下用全部点预测自己会得到 0 方差/0 误差，
      // 而留一折的预测方差必须为正。
      assert.ok(fold.variance > 1e-9);
    }
    const s = out.summary!;
    assert.ok(Math.abs(s.meanError) < 1e-12 || Number.isFinite(s.meanError));
    assert.ok(s.rootMeanSquareError >= s.meanAbsoluteError - 1e-12);
    assert.ok(s.coverage95 >= 0 && s.coverage95 <= 1);
  });

  it("单测点无法留一并明确说明", () => {
    const out = leaveOneOut({ observations: obs([[0, 0, 1]]), model });
    assert.equal(out.feasible, false);
    assert.ok(out.reason);
  });
});

describe("不可靠输入：严重病态/奇异布局", () => {
  it("零噪声下测点几乎完全重合使 KKT 严重病态时如实报告", () => {
    const data: Observation[] = obs([
      [0, 0, 1],
      [1e-12, 0, 2],
      [2e-12, 0, 1.5],
    ]);
    const r = predictPoint({ observations: data, model }, { x: 1, y: 1 });
    assert.notEqual(r.health, "ok");
    assert.ok(r.messages.length > 0);
    assert.ok(
      r.equationResidualRel > 1e-10 ||
        r.condition1Estimate > 1e10 ||
        r.health === "singular",
    );
  });

  it("合并说明被返回：零噪同值重复点被去重为有效测点", () => {
    const data: Observation[] = obs([
      [0, 0, 1],
      [0, 0, 1],
      [1, 0, 2],
    ]);
    const r = predictPoint({ observations: data, model }, { x: 0.5, y: 0 });
    assert.equal(r.health, "ok");
    assert.equal(r.effectiveWeights.length, 2);
    assert.ok(r.preparationNotes.some((n) => n.includes("去重")));
  });
});
