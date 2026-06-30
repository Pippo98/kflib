// Simple example: constant velocity Kalman filter
// State: [position; velocity]

#include <random>
#include <cstdio>
#include <iostream>

#include "kflib/kf.hpp"

// Indexes for state vector
enum { IDX_POS = 0, IDX_VEL = 1 };
// Index for measurement vector
enum { MEAS_POS = 0 };

int main() {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  const double dt = 1.0; // time step

  // Create filter
  KalmanFilter kf;

  // State is [pos; vel]
  VectorXd x0(2);
  x0(IDX_POS) = 0.0;
  x0(IDX_VEL) = 1.0; // initial true velocity
  kf.setState(x0);

  // Initial covariance
  MatrixXd P0 = MatrixXd::Identity(2, 2) * 0.1;
  kf.setStateCovariance(P0);

  // State transition A
  MatrixXd A(2, 2);
  A << 1.0, dt,
       0.0, 1.0;
  kf.setStateUpdateMatrix(A);

  // Measurement matrix H (we measure position only)
  MatrixXd H(1, 2);
  H << 1.0, 0.0;
  kf.setMeasurementMatrix(H);

  // Process noise covariance Q
  MatrixXd Q = MatrixXd::Zero(2, 2);
  // simple model: small acceleration noise affecting velocity and position
  Q(0, 0) = 0.01;
  Q(1, 1) = 0.01;
  kf.setProcessCovariance(Q);

  // Measurement noise covariance R (scalar)
  MatrixXd R(1, 1);
  R(0, 0) = 0.5; // measurement noise variance
  kf.setMeasurementCovariance(R);

  // Random generators for true process noise and measurement noise
  std::mt19937 rng(12345);
  std::normal_distribution<double> procNoise(0.0, 0.1);
  std::normal_distribution<double> measNoise(0.0, std::sqrt(R(0, 0)));

  // True state
  VectorXd xt = x0;

  std::printf("%3s, %12s, %12s, %12s, %12s, %12s\n",
              "t", "true_pos", "true_vel", "est_pos", "est_vel",
              "meas_pos");
  for (int t = 0; t < 20; ++t) {
    // Simulate true system
    double acc = procNoise(rng); // random acceleration noise
    xt(IDX_POS) += xt(IDX_VEL) * dt + 0.5 * acc * dt * dt;
    xt(IDX_VEL) += acc * dt;

    // Measurement (position only)
    VectorXd z(1);
    z(MEAS_POS) = xt(IDX_POS) + measNoise(rng);

    // Kalman filter predict + update
    kf.predict();
    kf.update(z);

    VectorXd xe = kf.getState();

    std::printf("%3d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f\n",
          t, xt(IDX_POS), xt(IDX_VEL), xe(IDX_POS), xe(IDX_VEL), z(0));
  }

  return 0;
}
