#pragma once

#include "kf_base.hpp"

// Linear Kalman Filter
class KalmanFilter : public KalmanFilterBase {
 public:
  void setStateUpdateMatrix(const Eigen::MatrixXd &A);
  void setStateUpdateMatrices(const Eigen::MatrixXd &A,
                              const Eigen::MatrixXd &B);
  void setMeasurementMatrix(const Eigen::MatrixXd &H);

  void predict(const Eigen::VectorXd &input) override;
  void update(const Eigen::VectorXd &measurements) override;

 private:
  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  Eigen::MatrixXd H;
};
