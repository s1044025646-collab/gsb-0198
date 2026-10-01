// Shared domain types.

export interface Point {
  x: number;
  y: number;
}

export interface Observation extends Point {
  /** Scalar measured value z = field(x,y) + independent measurement noise. */
  value: number;
  /** Optional label for readable weight reporting. */
  label?: string;
}

/**
 * Fixed isotropic exponential covariance model:
 *   C(h) = sill * exp(-h / range)
 * where |range| is the isotropic correlation length (1/e distance).
 *
 * Observation model: z_i = S(x_i) + eps_i, eps_i ~ N(0, nugget),
 * with nugget = measurementNoiseVariance shared by every observation.
 */
export interface CovarianceModel {
  /** Sill / partial sill: variance of the latent field S at h = 0, > 0. */
  sill: number;
  /** Isotropic correlation length (1/e distance), in coordinate units, > 0. */
  range: number;
  /**
   * Independent measurement-noise (nugget) variance added ONLY to diagonal
   * entries of the observation-observation covariance block, >= 0.
   */
  measurementNoiseVariance: number;
}

export type PredictTarget = "latent" | "observation";

/** Numerical health classification of a solved Kriging system. */
export type SolveHealth = "ok" | "ill_conditioned" | "singular";

export const LIMITS = {
  /** Maximum observations per dataset for the dense solver. */
  maxObservations: 400,
  /** Maximum grid nodes for one grid prediction. */
  maxGridNodes: 20000,
  /** Default condition-number threshold above which health = ill_conditioned. */
  illConditionedCond: 1e10,
  /**
   * A computed variance above -this* (and below 0) is treated as round-off
   * and clamped to 0. Values more negative are a substantive failure.
   */
  tinyNegativeVarianceTol: 1e-8,
} as const;
