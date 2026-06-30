// Unscented Kalman filter example: turn model
// State: [x, y, v, psi, omega]
// Measurement: position [x; y]

#include <random>
#include <cstdio>
#include <iostream>

#include "kflib/ukf.hpp"

// Indexes for state vector
enum { IDX_X = 0, IDX_Y = 1, IDX_V = 2, IDX_PSI = 3, IDX_OMEGA = 4 };
// Indexes for measurement vector
enum { MEAS_X = 0, MEAS_Y = 1 };

// Struct holding additional data
struct UserData {
  double dt;
};

// Nonlinear state transition function for turn model
Eigen::VectorXd turn_model_state_func(const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *userData) {
  UserData *data = static_cast<UserData*>(userData);
  double dt = data->dt;

  double x = state(IDX_X);
  double y = state(IDX_Y);
  double v = state(IDX_V);
  double psi = state(IDX_PSI);
  double omega = state(IDX_OMEGA);

  // Dynamics:
  // x_dot = v * cos(psi)
  // y_dot = v * sin(psi)
  // psi_dot = omega
  // v and omega are assumed constant (no acceleration input)

  double new_x = x + v * cos(psi) * dt;
  double new_y = y + v * sin(psi) * dt;
  double new_v = v;
  double new_psi = psi + omega * dt;
  double new_omega = omega;

  Eigen::VectorXd next_state(5);
  next_state << new_x, new_y, new_v, new_psi, new_omega;
  return next_state;
}

// Measurement function: position [x; y]
Eigen::VectorXd turn_model_measurement_func(const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *userData) {
  Eigen::VectorXd meas(2);
  meas << state(IDX_X), state(IDX_Y);
  return meas;
}

int main() {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  const double dt = 0.1; // time step
  UserData userData;
  userData.dt = dt;

  // Create UKF with 5 states and 2 measurements
  UnscentedKalmanFilter ukf;
  ukf.setUserData(&userData);

  // Set state and measurement functions
  ukf.setStateUpdateFunction(turn_model_state_func);
  ukf.setMeasurementFunction(turn_model_measurement_func);

  // Set default sigma points parameters (can be customized)
  MerweScaledSigmaPointsParams params;
  params.alpha = 0.001;
  params.beta = 2;
  params.kappa = 0;
  ukf.setMerweScaledSigmaPointsParams(params);

  // Initial state [x; y; v; psi; omega]
  VectorXd x0(5);
  x0 << 0.0, 0.0, 1.0, 0.0, 0.1; // Start with small angular rate
  ukf.setState(x0);

  // Initial covariance
  MatrixXd P0 = MatrixXd::Identity(5, 5) * 0.01;
  ukf.setStateCovariance(P0);

  // Process noise covariance Q (small acceleration and angular noise)
  MatrixXd Q = MatrixXd::Zero(5, 5);
  Q(IDX_V, IDX_V) = 0.0001; // velocity noise
  Q(IDX_OMEGA, IDX_OMEGA) = 0.0001; // angular rate noise
  ukf.setProcessCovariance(Q);

  // Measurement noise covariance R
  MatrixXd R(2, 2);
  R.setIdentity();
  R(0, 0) = 0.01; // x noise
  R(1, 1) = 0.01; // y noise
  ukf.setMeasurementCovariance(R);

  // Random generators
  std::mt19937 rng(12345);
  std::normal_distribution<double> procNoiseV(0.0, 0.01);
  std::normal_distribution<double> procNoiseOmega(0.0, 0.001);
  std::normal_distribution<double> measNoiseX(0.0, std::sqrt(R(0, 0)));
  std::normal_distribution<double> measNoiseY(0.0, std::sqrt(R(1, 1)));

  // True state
  VectorXd xt = x0;

  std::printf("%3s, %12s, %12s, %12s, %12s, %12s, %12s, %12s\n",
              "t", "true_x", "true_y", "est_x", "est_y", "meas_x", "meas_y", "true_psi");

  for (int t = 0; t < 100; ++t) {
    // Simulate true system with process noise
    xt(IDX_V) += procNoiseV(rng);
    xt(IDX_OMEGA) += procNoiseOmega(rng);

    // Update true state using Euler integration
    double x = xt(IDX_X);
    double y = xt(IDX_Y);
    double v = xt(IDX_V);
    double psi = xt(IDX_PSI);
    double omega = xt(IDX_OMEGA);

    x += v * cos(psi) * dt;
    y += v * sin(psi) * dt;
    psi += omega * dt;

    xt(IDX_X) = x;
    xt(IDX_Y) = y;
    xt(IDX_PSI) = psi;

    // Measurement (position with noise)
    double meas_x = x + measNoiseX(rng);
    double meas_y = y + measNoiseY(rng);

    // Predict and update
    ukf.predict();
    VectorXd z(2);
    z << meas_x, meas_y;
    ukf.update(z);

    // Get estimates
    VectorXd x_est = ukf.getState();

    std::printf("%3d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f\n",
                t, xt(IDX_X), xt(IDX_Y), x_est(IDX_X), x_est(IDX_Y), meas_x, meas_y, xt(IDX_PSI));
  }

  return 0;
}