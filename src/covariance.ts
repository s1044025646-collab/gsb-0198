import type { CovarianceModel, Point } from "./types.js";

export function euclid(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Isotropic exponential covariance C(h) = sill * exp(-h / range). */
export function covariance(h: number, model: CovarianceModel): number {
  return model.sill * Math.exp(-h / model.range);
}

/**
 * Covariance between two observation locations. Independent measurement
 * noise is added only when i === j (diagonal of the observation block).
 */
export function obsCovariance(
  a: Point,
  b: Point,
  i: number,
  j: number,
  model: CovarianceModel,
): number {
  let c = covariance(euclid(a, b), model);
  if (i === j) c += model.measurementNoiseVariance;
  return c;
}

/**
 * Covariance between a prediction target and an observation.
 *
 * The target point and the observations are distinct random locations, so
 * cross-covariance never carries the observation nugget - regardless of
 * whether the requested target is the latent field or a NEW observation.
 * (Predicting a new observation only changes the variance baseline.)
 */
export function targetObsCovariance(
  target: Point,
  obs: Point,
  model: CovarianceModel,
): number {
  return covariance(euclid(target, obs), model);
}

/**
 * Variance baseline at the target:
 *  - "latent":      Var(S(x0))        = sill
 *  - "observation": Var(S(x0)+eps0)   = sill + measurementNoiseVariance
 * The new-observation noise eps0 is independent of every existing eps_i.
 */
export function targetVariance(
  model: CovarianceModel,
  target: "latent" | "observation",
): number {
  return target === "observation"
    ? model.sill + model.measurementNoiseVariance
    : model.sill;
}

export function validateModel(m: CovarianceModel): string[] {
  const errs: string[] = [];
  if (!Number.isFinite(m.sill) || m.sill <= 0)
    errs.push("sill 必须为有限正数");
  if (!Number.isFinite(m.range) || m.range <= 0)
    errs.push("range（相关长度）必须为有限正数");
  if (!Number.isFinite(m.measurementNoiseVariance) || m.measurementNoiseVariance < 0)
    errs.push("measurementNoiseVariance（测量噪声方差）必须为非负有限数");
  return errs;
}
