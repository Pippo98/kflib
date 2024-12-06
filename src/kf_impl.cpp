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
  sigmaPointsAlpha = 1.0;
  sigmaPointsBeta = 2.0;
  sigmaPointsKappa = 0.0;
  stateFunction = nullptr;
  measurementFunction = nullptr;
  constraintFunction = nullptr;
}

void UnscentedKalmanFilter::setMerweScaledSigmaPointsParams(double alpha,
                                                            double beta,
                                                            double kappa) {
  sigmaPointsAlpha = alpha;
  sigmaPointsBeta = beta;
  sigmaPointsKappa = kappa;
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

void UnscentedKalmanFilter::computeMerweScaledSigmaPoints(
    const Eigen::VectorXd &state, const Eigen::MatrixXd &P,
    MerweScaledSigmaPoints &outPoints) {
  assert(state.size() != 0);
  size_t n = state.rows();
  double lambda =
      sigmaPointsAlpha * sigmaPointsAlpha * (n + sigmaPointsKappa) - n;

  double allWeightsValue = 1 / (2 * (n + lambda));

  outPoints.sigmas.resize(n, 2 * n + 1);
  outPoints.meanWeights.resize(outPoints.sigmas.cols());
  outPoints.covarianceWeights.resize(outPoints.sigmas.cols());
  outPoints.meanWeights.setConstant(allWeightsValue);
  outPoints.covarianceWeights.setConstant(allWeightsValue);

  outPoints.sigmas.col(0) = state;
  outPoints.meanWeights(0) = lambda / (n + lambda);
  outPoints.covarianceWeights(0) =
      lambda / (n + lambda) +
      (1 - sigmaPointsAlpha * sigmaPointsAlpha + sigmaPointsBeta);

  Eigen::MatrixXd U = ((n + lambda) * P).sqrt();
  for (size_t i = 0; i < n; ++i) {
    outPoints.sigmas.col(i + 1) = state + U.col(i);
    outPoints.sigmas.col(i + n + 1) = state - U.col(i);
  }
}

void UnscentedKalmanFilter::computeMeanAndCovariance(
    const MerweScaledSigmaPoints &points,
    const Eigen::MatrixXd &additionalCovariance, Eigen::VectorXd &outX,
    Eigen::MatrixXd &outP) {
  outP.resize(points.sigmas.rows(), points.sigmas.rows());
  outP.setZero();
  outX = points.sigmas * points.meanWeights;
  auto y = points.sigmas;
  for (Eigen::Index i = 0; i < y.cols(); i++) {
    y.col(i) -= outX;
  }
  outP = y * (points.covarianceWeights.asDiagonal() * y.transpose());
  if (additionalCovariance.size() != 0) {
    outP += additionalCovariance;
  }
}
void UnscentedKalmanFilter::constrainSigmaPoints(
    MerweScaledSigmaPoints &sigmaPoints) {
  if (!constraintFunction) {
    return;
  }
  for (Eigen::Index col = 0; col < sigmaPoints.sigmas.cols(); ++col) {
    sigmaPoints.sigmas.col(col) =
        constraintFunction(sigmaPoints.sigmas.col(col), inputs, userData);
  }
}

Eigen::MatrixXd UnscentedKalmanFilter::computeKalmanGain(
    const MerweScaledSigmaPoints &stateSigmaPoints,
    const Eigen::MatrixXd &measureSigmaPoints,
    const Eigen::VectorXd &stateEstimate,
    const Eigen::VectorXd &measureEstimate,
    const Eigen::MatrixXd &measurementCovariance) {
  size_t n = X.rows();
  size_t m = measureEstimate.rows();
  size_t nSigmas = stateSigmaPoints.sigmas.cols();

  Eigen::MatrixXd crossCovariance(n, m);
  crossCovariance.setZero();
  for (size_t i = 0; i < nSigmas; ++i) {
    crossCovariance +=
        stateSigmaPoints.covarianceWeights(i) *
        ((stateSigmaPoints.sigmas.col(i) - stateEstimate) *
         (measureSigmaPoints.col(i) - measureEstimate).transpose());
  }
  return crossCovariance * measurementCovariance.inverse();
}
void UnscentedKalmanFilter::predict(const Eigen::VectorXd &inputs_) {
  inputs = inputs_;
  computeMerweScaledSigmaPoints(X, P, stateSigmaPoints);
  constrainSigmaPoints(stateSigmaPoints);
  for (Eigen::Index i = 0; i < stateSigmaPoints.sigmas.cols(); ++i) {
    stateSigmaPoints.sigmas.col(i) =
        stateFunction(stateSigmaPoints.sigmas.col(i), inputs, userData);
  }
  constrainSigmaPoints(stateSigmaPoints);
  computeMeanAndCovariance(stateSigmaPoints, Q, X, P);
}
void UnscentedKalmanFilter::update(const Eigen::VectorXd &measurements) {
  MerweScaledSigmaPoints measureSigmaPoints = stateSigmaPoints;
  // computeMerweScaledSigmaPoints(X, P, measureSigmaPoints);
  Eigen::MatrixXd measuresFromEstimate(measurements.rows(),
                                       stateSigmaPoints.sigmas.cols());
  for (Eigen::Index i = 0; i < stateSigmaPoints.sigmas.cols(); ++i) {
    measuresFromEstimate.col(i) =
        measurementFunction(stateSigmaPoints.sigmas.col(i), inputs, userData);
  }
  measureSigmaPoints.sigmas = measuresFromEstimate;

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

  Eigen::VectorXd xb;
  Eigen::MatrixXd Pb;
  MerweScaledSigmaPoints sigma;
  MerweScaledSigmaPoints sigmaPredicted;
  for (int i = xIn.size() - 2; i >= 0; i--) {
    computeMerweScaledSigmaPoints(xSmooth[i], pSmooth[i], sigma);
    sigmaPredicted = sigma;

    for (int s = 0; s < sigma.sigmas.cols(); s++) {
      sigmaPredicted.sigmas.col(s) =
          stateFunction(sigma.sigmas.col(s), inputs[i], userData);
    }

    computeMeanAndCovariance(sigmaPredicted, Q, xb, Pb);

    Eigen::MatrixXd K =
        computeKalmanGain(sigma, sigmaPredicted.sigmas, xIn[i], xb, Pb);

    xSmooth[i] += K * (xSmooth[i + 1] - xb);
    pSmooth[i] += K * (pSmooth[i + 1] - Pb) * K.transpose();
  }
}
