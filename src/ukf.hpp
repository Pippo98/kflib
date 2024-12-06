#pragma once

#include "kf_base.hpp"

struct MerweScaledSigmaPoints {
  Eigen::MatrixXd sigmas;
  Eigen::VectorXd meanWeights;
  Eigen::VectorXd covarianceWeights;
};

typedef Eigen::VectorXd (*constraint_function_t)(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &input,
                                                 void *userData);

// Unscented Kalman Filter
class UnscentedKalmanFilter : public KalmanFilterBase {
 public:
  UnscentedKalmanFilter();

  const Eigen::VectorXd &getInputs() const;

  void setMerweScaledSigmaPointsParams(double alpha, double beta, double kappa);

  void setStateUpdateFunction(state_function_t stateUpdateFunction);
  void setMeasurementFunction(measurement_function_t measurementFuction);

  using KalmanFilterBase::predict;
  void predict(const Eigen::VectorXd &input) override;
  void update(const Eigen::VectorXd &measurements) override;

  void computeMerweScaledSigmaPoints(const Eigen::VectorXd &state,
                                     const Eigen::MatrixXd &P,
                                     MerweScaledSigmaPoints &outPoints);
  void computeMeanAndCovariance(const MerweScaledSigmaPoints &points,
                                const Eigen::MatrixXd &additionalCovariance,
                                Eigen::VectorXd &outX, Eigen::MatrixXd &outP);

  void setStateConstraintsFunction(constraint_function_t constraintFunction);

  void RTSSmoother(std::vector<Eigen::VectorXd> &states,
                   std::vector<Eigen::MatrixXd> &covariances,
                   const std::vector<Eigen::VectorXd> &inputs);

 private:
  Eigen::MatrixXd computeKalmanGain(
      const MerweScaledSigmaPoints &stateSigmaPoints,
      const Eigen::MatrixXd &measuresSigmas,
      const Eigen::VectorXd &stateEstimate,
      const Eigen::VectorXd &measureEstimate,
      const Eigen::MatrixXd &measurementCovariance);
  void constrainSigmaPoints(MerweScaledSigmaPoints &sigmaPoints);

 private:
  double sigmaPointsAlpha;
  double sigmaPointsBeta;
  double sigmaPointsKappa;

  Eigen::VectorXd inputs;

  MerweScaledSigmaPoints stateSigmaPoints;

  state_function_t stateFunction;
  measurement_function_t measurementFunction;
  constraint_function_t constraintFunction;
};
