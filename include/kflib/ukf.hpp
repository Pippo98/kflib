#pragma once

#include "kf_base.hpp"

/**
 * @file ukf.hpp
 * @brief Unscented Kalman Filter (UKF) utilities and API.
 *
 * Provides Merwe-scaled sigma point parameters, function types used by the
 * UKF, and the `UnscentedKalmanFilter` class. Notes:
 * - Batch mode: when enabled, the user-provided batch state function receives
 *   all sigma points at once as columns of a matrix. The caller (user)
 *   is responsible for handling an internal loop over columns if needed.
 * - `RTSSmoother` is provided for Rauch–Tung–Striebel smoothing of recorded
 *   states, covariances and inputs in post-processing.
 */

/**
 * @brief Parameters for Merwe-scaled sigma point.
 * @param alpha Spread of the sigma points.
 * @param beta Prior knowledge about the distribution (2 is optimal for
 * Gaussian).
 * @param kappa Secondary scaling parameter (often n - 3).
 */
struct MerweScaledSigmaPointsParams {
  double alpha;
  double beta;
  double kappa;

  double lambda;
  Eigen::VectorXd meanWeights;
  Eigen::VectorXd covarianceWeights;
};

/** Sigma points container (each column is a sigma point). */
typedef Eigen::MatrixXd SigmaPoints;

/** Constraint function applied to states (optional). */
typedef Eigen::VectorXd (*constraint_function_t)(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &input,
                                                 void *userData);

/**
 * Batch state function signature: receives a matrix whose columns are the
 * sigma points. Returns a matrix with corresponding transformed columns.
 */
typedef Eigen::MatrixXd (*state_function_batch_t)(const Eigen::MatrixXd &state,
                                                  const Eigen::VectorXd &input,
                                                  void *userData);
/**
 * Batch measurement function signature: receives a matrix whose columns are the
 * sigma points. Returns a matrix with the measurements estimates.
 */
typedef Eigen::MatrixXd (*measurement_function_batch_t)(
    const Eigen::MatrixXd &state, const Eigen::VectorXd &input, void *userData);

/**
 * @brief Unscented Kalman Filter implementation.
 *
 * Supports Merwe-scaled sigma points, optional batch processing of sigma
 * points, state constraints, and an RTSSmoother for offline smoothing.
 */
class UnscentedKalmanFilter : public KalmanFilterBase {
public:
  UnscentedKalmanFilter();

  /**
   * @brief Configure Merwe-scaled sigma point parameters.
   * @param alpha Spread of the sigma points.
   * @param beta Prior knowledge about the distribution (2 is optimal for
   * Gaussian).
   * @param kappa Secondary scaling parameter (often n - 3).
   */
  void setMerweScaledSigmaPointsParams(double alpha, double beta, double kappa);
  void setMerweScaledSigmaPointsParams(MerweScaledSigmaPointsParams params);

  /** Set the scalar (per-sigma) state update function f(x,u). */
  void setStateUpdateFunction(state_function_t stateUpdateFunction);
  /** Set the scalar measurement function h(x,u). */
  void setMeasurementFunction(measurement_function_t measurementFuction);

  /** Enable processing where the state function is called with all sigma points
   * at once. */
  void enableBatchMode();
  /** Disable batch mode (default behavior: single-column processing). */
  void disableBatchMode();
  /** Set a batch-style state update function that accepts all sigma points. */
  void setStateUpdateFunctionBatch(state_function_batch_t stateUpdateFunction);
  void
  setMeasurementFunctionBatch(measurement_function_batch_t measurementFunction);

  using KalmanFilterBase::predict;
  void predict(const Eigen::VectorXd &input) override;
  void update(const Eigen::VectorXd &measurements) override;

  /** Compute Merwe weights and populate `params`. */
  void
  computeMerweScaledSigmaPointsWeights(MerweScaledSigmaPointsParams &params);
  /** Generate sigma points from `state` and covariance `P`. */
  void computeMerweScaledSigmaPoints(const Eigen::VectorXd &state,
                                     const Eigen::MatrixXd &P,
                                     SigmaPoints &outPoints);
  /** Compute mean and covariance from sigma `points` and add extra covariance.
   */
  void computeMeanAndCovariance(const SigmaPoints &points,
                                const Eigen::MatrixXd &additionalCovariance,
                                Eigen::VectorXd &outX, Eigen::MatrixXd &outP);

  /** Set a function to enforce constraints on sigma-point-derived states. */
  void setStateConstraintsFunction(constraint_function_t constraintFunction);

  /**
   * @brief Rauch–Tung–Striebel smoother.
   * @param states In/out vector of state estimates (will be smoothed in place).
   * @param covariances In/out vector of state covariances.
   * @param inputs Vector of inputs corresponding to each timestep.
   */
  void RTSSmoother(std::vector<Eigen::VectorXd> &states,
                   std::vector<Eigen::MatrixXd> &covariances,
                   const std::vector<Eigen::VectorXd> &inputs);

private:
  Eigen::MatrixXd
  computeKalmanGain(const SigmaPoints &stateSigmaPoints,
                    const Eigen::MatrixXd &measuresSigmas,
                    const Eigen::VectorXd &stateEstimate,
                    const Eigen::VectorXd &measureEstimate,
                    const Eigen::MatrixXd &measurementCovariance) const;
  void constrainSigmaPoints(SigmaPoints &sigmaPoints);

private:
  MerweScaledSigmaPointsParams sigmaParams;

  Eigen::VectorXd inputs;
  SigmaPoints stateSigmaPoints;

  state_function_t stateFunction;
  measurement_function_t measurementFunction;
  constraint_function_t constraintFunction;

  bool useBatchMode = false;
  state_function_batch_t stateFunctionBatch;
  measurement_function_batch_t measurementFunctionBatch;
};
