#include <Eigen/LU>
#include <cassert>
#include <iostream>
#include <ostream>
#include <unsupported/Eigen/MatrixFunctions>

#include "ekf.hpp"
#include "kf.hpp"
#include "kf_base.hpp"
#include "ukf.hpp"

void KalmanFilterBase::setUserData(void *data) { userData = data; }

void KalmanFilterBase::setState(const Eigen::VectorXd &state) { X = state; }
void KalmanFilterBase::setStateCovariance(
    const Eigen::MatrixXd &stateCovariance) {
  P = stateCovariance;
}
void KalmanFilterBase::setProcessCovariance(
    const Eigen::MatrixXd &processCovariance) {
  Q = processCovariance;
}
void KalmanFilterBase::setMeasurementCovariance(
    const Eigen::MatrixXd &measurementCovariance) {
  R = measurementCovariance;
}

const Eigen::VectorXd &KalmanFilterBase::getState() const { return X; }
const Eigen::MatrixXd &KalmanFilterBase::getCovariance() const { return P; }

void KalmanFilterBase::print() const { printToStream(std::cout); }
void KalmanFilterBase::printToStream(std::ostream &stream) const {
  stream << "-- STATE --\n";
  stream << X << "\n";
  stream << "-- COVARIANCE --\n";
  stream << P << std::endl;
}

// Linear Kalman Filter
void KalmanFilter::setStateUpdateMatrix(const Eigen::MatrixXd &A_) {
  A = A_;
  B = Eigen::MatrixXd();
}

void KalmanFilter::setStateUpdateMatrices(const Eigen::MatrixXd &A_,
                                          const Eigen::MatrixXd &B_) {
  A = A_;
  B = B_;
}
void KalmanFilter::setMeasurementMatrix(const Eigen::MatrixXd &H_) { H = H_; }

void KalmanFilter::predict(const Eigen::VectorXd &input) {
  X = A * X;
  if (B.size() != 0 && input.size() != 0) {
    X += B * input;
  }
  P = A * P * A.transpose();
  if (Q.size() != 0) {
    P += Q;
  }
}
void KalmanFilter::update(const Eigen::VectorXd &measurements) {
  Eigen::MatrixXd S = H * P * H.transpose();
  if (R.size() != 0) {
    S += R;
  }
  auto K = P * H.transpose() * S.inverse();
  X = X + K * (measurements - H * X);
  Eigen::MatrixXd I(X.rows(), X.rows());
  I.setIdentity();
  P = (I - K * H) * P;
}

void ExtendedKalmanFilter::setStateUpdateFunction(
    state_function_t stateFunction_) {
  stateFunction = stateFunction_;
}
void ExtendedKalmanFilter::setMeasurementFunction(
    measurement_function_t measurementFunction_) {
  measurementFunction = measurementFunction_;
}
void ExtendedKalmanFilter::setStateJacobian(
    state_jacobian_function_t functionThatReturnsF_) {
  stateJacobian = functionThatReturnsF_;
}
void ExtendedKalmanFilter::setMeasurementJacobian(
    measurement_jacobian_function_t functionThatReturnsH_) {
  measurementJacobian = functionThatReturnsH_;
}
void ExtendedKalmanFilter::predict(const Eigen::VectorXd &inputs_) {
  inputs = inputs_;
  const auto &F = stateJacobian(X, inputs, userData);
  X = stateFunction(X, inputs, userData);
  P = F * P * F.transpose();
  if (Q.size() != 0) {
    P += Q;
  }
}
void ExtendedKalmanFilter::update(const Eigen::VectorXd &measurements) {
  const auto &H = measurementJacobian(measurements, inputs, userData);
  Eigen::MatrixXd S = H * P * H.transpose();
  if (R.size() != 0) {
    S += R;
  }
  auto K = P * H.transpose() * S.inverse();

  auto zEst = measurementFunction(X, inputs, userData);
  X = X + K * (measurements - zEst);

  Eigen::MatrixXd I(X.rows(), X.rows());
  I.setIdentity();
  P = (I - K * H) * P;
}

UnscentedKalmanFilter::UnscentedKalmanFilter() {
  setMerweScaledSigmaPointsParams(1.0, 2.0, 0.0);
  stateFunction = nullptr;
  measurementFunction = nullptr;
  constraintFunction = nullptr;
}

void UnscentedKalmanFilter::setMerweScaledSigmaPointsParams(double alpha,
                                                            double beta,
                                                            double kappa) {
  sigmaParams.alpha = alpha;
  sigmaParams.beta = beta;
  sigmaParams.kappa = kappa;
  computeMerweScaledSigmaPointsWeights(sigmaParams);
}

void UnscentedKalmanFilter::setStateUpdateFunction(
    state_function_t stateFunction_) {
  stateFunction = stateFunction_;
}
void UnscentedKalmanFilter::setMeasurementFunction(
    measurement_function_t measurementFunction_) {
  measurementFunction = measurementFunction_;
}

void UnscentedKalmanFilter::setStateConstraintsFunction(
    constraint_function_t constraintFunction_) {
  constraintFunction = constraintFunction_;
}

const Eigen::VectorXd &UnscentedKalmanFilter::getInputs() const {
  return inputs;
}

void UnscentedKalmanFilter::computeMerweScaledSigmaPointsWeights(
    MerweScaledSigmaPointsParams &params) {
  size_t n = X.rows();
  size_t numSigmas = 2 * n + 1;

  params.meanWeights.resize(numSigmas);
  params.covarianceWeights.resize(numSigmas);

  params.lambda = params.alpha * params.alpha * (n + params.kappa) - n;

  double allWeightsValue = 1 / (2 * (n + params.lambda));
  params.meanWeights.setConstant(allWeightsValue);
  params.covarianceWeights.setConstant(allWeightsValue);

  params.meanWeights(0) = params.lambda / (n + params.lambda);
  params.covarianceWeights(0) = params.lambda / (n + params.lambda) +
                                (1 - params.alpha * params.alpha + params.beta);
}
void UnscentedKalmanFilter::computeMerweScaledSigmaPoints(
    const Eigen::VectorXd &state, const Eigen::MatrixXd &P,
    SigmaPoints &outPoints) {
  assert(state.size() != 0);
  size_t n = state.rows();
  size_t numSigmas = 2 * n + 1;
  if (sigmaParams.covarianceWeights.cols() != (int)numSigmas) {
    computeMerweScaledSigmaPointsWeights(sigmaParams);
  }
  outPoints.resize(n, numSigmas);

  outPoints.col(0) = state;

  Eigen::MatrixXd U = ((n + sigmaParams.lambda) * P).sqrt();
  for (size_t i = 0; i < n; ++i) {
    outPoints.col(i + 1) = state + U.col(i);
    outPoints.col(i + n + 1) = state - U.col(i);
  }
}

void UnscentedKalmanFilter::computeMeanAndCovariance(
    const SigmaPoints &points, const Eigen::MatrixXd &additionalCovariance,
    Eigen::VectorXd &outX, Eigen::MatrixXd &outP) {
  outP.resize(points.rows(), points.rows());
  outP.setZero();
  outX = points * sigmaParams.meanWeights;
  auto y = points;
  for (Eigen::Index i = 0; i < y.cols(); i++) {
    y.col(i) -= outX;
  }
  outP = y * (sigmaParams.covarianceWeights.asDiagonal() * y.transpose());
  if (additionalCovariance.size() != 0) {
    outP += additionalCovariance;
  }
}
void UnscentedKalmanFilter::constrainSigmaPoints(SigmaPoints &sigmaPoints) {
  if (!constraintFunction) {
    return;
  }
  for (Eigen::Index col = 0; col < sigmaPoints.cols(); ++col) {
    sigmaPoints.col(col) =
        constraintFunction(sigmaPoints.col(col), inputs, userData);
  }
}

Eigen::MatrixXd UnscentedKalmanFilter::computeKalmanGain(
    const SigmaPoints &stateSigmaPoints,
    const Eigen::MatrixXd &measureSigmaPoints,
    const Eigen::VectorXd &stateEstimate,
    const Eigen::VectorXd &measureEstimate,
    const Eigen::MatrixXd &measurementCovariance) {
  size_t n = X.rows();
  size_t m = measureEstimate.rows();
  size_t nSigmas = stateSigmaPoints.cols();

  Eigen::MatrixXd crossCovariance(n, m);
  crossCovariance.setZero();
  for (size_t i = 0; i < nSigmas; ++i) {
    crossCovariance +=
        sigmaParams.covarianceWeights(i) *
        ((stateSigmaPoints.col(i) - stateEstimate) *
         (measureSigmaPoints.col(i) - measureEstimate).transpose());
  }
  return crossCovariance * measurementCovariance.inverse();
}
void UnscentedKalmanFilter::predict(const Eigen::VectorXd &inputs_) {
  inputs = inputs_;
  computeMerweScaledSigmaPoints(X, P, stateSigmaPoints);
  constrainSigmaPoints(stateSigmaPoints);
  for (Eigen::Index i = 0; i < stateSigmaPoints.cols(); ++i) {
    stateSigmaPoints.col(i) =
        stateFunction(stateSigmaPoints.col(i), inputs, userData);
  }
  constrainSigmaPoints(stateSigmaPoints);
  computeMeanAndCovariance(stateSigmaPoints, Q, X, P);
}
void UnscentedKalmanFilter::update(const Eigen::VectorXd &measurements) {
  SigmaPoints measureSigmaPoints = stateSigmaPoints;
  // computeMerweScaledSigmaPoints(X, P, measureSigmaPoints);
  Eigen::MatrixXd measuresFromEstimate(measurements.rows(),
                                       stateSigmaPoints.cols());
  for (Eigen::Index i = 0; i < stateSigmaPoints.cols(); ++i) {
    measuresFromEstimate.col(i) =
        measurementFunction(stateSigmaPoints.col(i), inputs, userData);
  }
  measureSigmaPoints = measuresFromEstimate;

  Eigen::VectorXd zEst;
  Eigen::MatrixXd Pz;
  computeMeanAndCovariance(measureSigmaPoints, R, zEst, Pz);
  auto K =
      computeKalmanGain(stateSigmaPoints, measuresFromEstimate, X, zEst, Pz);

  X = X + K * (measurements - zEst);
  P = P - K * Pz * K.transpose();

  if (constraintFunction) {
    X = constraintFunction(X, inputs, userData);
  }
}

void UnscentedKalmanFilter::RTSSmoother(
    std::vector<Eigen::VectorXd> &xSmooth,
    std::vector<Eigen::MatrixXd> &pSmooth,
    const std::vector<Eigen::VectorXd> &inputs) {
  std::vector<Eigen::VectorXd> xIn = xSmooth;
  std::vector<Eigen::MatrixXd> PIn = pSmooth;
  size_t n = xIn.front().rows();

  Eigen::VectorXd xb;
  Eigen::MatrixXd Pb;
  SigmaPoints sigma;
  SigmaPoints sigmaPredicted(n, 2 * n + 1);
  for (int i = xIn.size() - 2; i >= 0; i--) {
    computeMerweScaledSigmaPoints(xSmooth[i], pSmooth[i], sigma);

    for (int s = 0; s < sigma.cols(); s++) {
      sigmaPredicted.col(s) = stateFunction(sigma.col(s), inputs[i], userData);
    }

    computeMeanAndCovariance(sigmaPredicted, Q, xb, Pb);

    Eigen::MatrixXd K =
        computeKalmanGain(sigma, sigmaPredicted, xIn[i], xb, Pb);

    xSmooth[i] += K * (xSmooth[i + 1] - xb);
    pSmooth[i] += K * (pSmooth[i + 1] - Pb) * K.transpose();
  }
}
