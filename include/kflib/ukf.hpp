#pragma once

#include "kf_base.hpp"

struct MerweScaledSigmaPointsParams {
  double alpha;
  double beta;
  double kappa;

  double lambda;
  Eigen::VectorXd meanWeights;
  Eigen::VectorXd covarianceWeights;
};

typedef Eigen::MatrixXd SigmaPoints;
typedef Eigen::VectorXd (*constraint_function_t)(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &input,
                                                 void *userData);

typedef Eigen::MatrixXd (*state_function_batch_t)(const Eigen::MatrixXd &state,
                                                  const Eigen::VectorXd &input,
                                                  void *userData);

// Unscented Kalman Filter
class UnscentedKalmanFilter : public KalmanFilterBase {
public:
  UnscentedKalmanFilter();

  // Kappa should be number of states - 3 for typical UKF behavior
  void setMerweScaledSigmaPointsParams(double alpha, double beta, double kappa);

  void setStateUpdateFunction(state_function_t stateUpdateFunction);
  void setMeasurementFunction(measurement_function_t measurementFuction);

  void enableBatchMode();
  void disableBatchMode();
  void setStateUpdateFunctionBatch(state_function_batch_t stateUpdateFunction);

  using KalmanFilterBase::predict;
  void predict(const Eigen::VectorXd &input) override;
  void update(const Eigen::VectorXd &measurements) override;

  void
  computeMerweScaledSigmaPointsWeights(MerweScaledSigmaPointsParams &params);
  void computeMerweScaledSigmaPoints(const Eigen::VectorXd &state,
                                     const Eigen::MatrixXd &P,
                                     SigmaPoints &outPoints);
  void computeMeanAndCovariance(const SigmaPoints &points,
                                const Eigen::MatrixXd &additionalCovariance,
                                Eigen::VectorXd &outX, Eigen::MatrixXd &outP);

  void setStateConstraintsFunction(constraint_function_t constraintFunction);

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
};
