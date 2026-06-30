#include "kflib/ekf.hpp"

#include <Eigen/LU>

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
  U = inputs_;
  const auto &F = stateJacobian(X, U, userData);
  X = stateFunction(X, U, userData);
  P = F * P * F.transpose();
  if (Q.size() != 0) {
    P += Q;
  }
}
void ExtendedKalmanFilter::update(const Eigen::VectorXd &measurements) {
  const auto &H = measurementJacobian(X, U, userData);
  Eigen::MatrixXd S = H * P * H.transpose();
  if (R.size() != 0) {
    S += R;
  }
  auto K = P * H.transpose() * S.inverse();

  auto zEst = measurementFunction(X, U, userData);
  X = X + K * (measurements - zEst);

  Eigen::MatrixXd I(X.rows(), X.rows());
  I.setIdentity();
  P = (I - K * H) * P;
}

