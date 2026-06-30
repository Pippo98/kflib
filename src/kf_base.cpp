#include <iostream>

#include "kflib/kf_base.hpp"

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

const Eigen::VectorXd &KalmanFilterBase::getInputs() const { return U; }

void KalmanFilterBase::print() const { printToStream(std::cout); }
void KalmanFilterBase::printToStream(std::ostream &stream) const {
  stream << "-- STATE --\n";
  stream << X << "\n";
  stream << "-- COVARIANCE --\n";
  stream << P << std::endl;
}
