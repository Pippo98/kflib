// Extended Kalman filter example: pendulum model
// State: [theta (angle), omega (angular velocity)]
// Measurement: angle (theta)

#include <random>
#include <cstdio>
#include <iostream>

#include "kflib/ekf.hpp"

// Indexes for state vector
enum { IDX_THETA = 0, IDX_OMEGA = 1 };
// Index for measurement vector
enum { MEAS_POS = 0 };
// Struct holding additiona data that can be used inside
// the state update measurement or jacobian functions
struct UserData {
    double dt;
};

// Nonlinear state transition function for pendulum
// dx/dt = [omega; -g/L * sin(theta)]
// Assuming L=1, g=9.81
Eigen::VectorXd pendulum_state_func(const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *data) {
  UserData *userData = static_cast<UserData *>(data);
  double theta = state(IDX_THETA);
  double omega = state(IDX_OMEGA);

  // Simple Euler integration (for demonstration purposes)
  double new_theta = theta + omega * userData->dt;
  double new_omega = omega - 9.81 * sin(theta) * userData->dt;

  Eigen::VectorXd next_state(2);
  next_state << new_theta, new_omega;
  return next_state;
}

// Measurement function: position = sin(theta)
Eigen::VectorXd pendulum_measurement_func(const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *userData) {
  Eigen::VectorXd meas(1);
  meas(MEAS_POS) = state(IDX_THETA);
  return meas;
}

// State Jacobian for pendulum dynamics
Eigen::MatrixXd pendulum_state_jacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *data) {
  UserData *userData = static_cast<UserData *>(data);
  double theta = state(IDX_THETA);
  Eigen::MatrixXd F(2, 2);
  F << 1.0, userData->dt,
       -9.81 * cos(theta) * userData->dt, 1.0;
  return F;
}

// Measurement Jacobian
Eigen::MatrixXd pendulum_measurement_jacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *userData) {
  Eigen::MatrixXd H(1, 2);
  H << 1.0, 0.0;
  return H;
}

int main() {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  const double dt = 0.1; // time step
  UserData userData;
  userData.dt = dt;

  // Create EKF with custom functions
  ExtendedKalmanFilter ekf;
  ekf.setStateUpdateFunction(pendulum_state_func);
  ekf.setMeasurementFunction(pendulum_measurement_func);
  ekf.setStateJacobian(pendulum_state_jacobian);
  ekf.setMeasurementJacobian(pendulum_measurement_jacobian);

  ekf.setUserData(&userData);

  // Initial state [theta; omega]
  VectorXd x0(2);
  x0 << 0.1, 0.0; // Small initial angle
  ekf.setState(x0);

  // Initial covariance
  MatrixXd P0 = MatrixXd::Identity(2, 2) * 0.01;
  ekf.setStateCovariance(P0);

  // Process noise covariance Q
  MatrixXd Q = MatrixXd::Identity(2, 2) * 0.0001;
  ekf.setProcessCovariance(Q);

  // Measurement noise covariance R
  MatrixXd R(1, 1);
  R(0, 0) = 1 * 3.1415 / 180.0; // measurement noise variance
  ekf.setMeasurementCovariance(R);

  // Random generators
  std::mt19937 rng(12345);
  std::normal_distribution<double> procNoise(0.0, 0.01);
  std::normal_distribution<double> measNoise(0.0, std::sqrt(R(0, 0)));

  // True state
  VectorXd xt = x0;

  std::printf("%3s, %12s, %12s, %12s, %12s, %12s\n",
              "t", "true_theta", "true_omega", "est_theta", "est_omega",
              "meas_theta");

  for (int t = 0; t < 50; ++t) {
    // Simulate true system with process noise
    double acc_noise = procNoise(rng);
    xt(IDX_OMEGA) -= 9.81 * sin(xt(IDX_THETA)) * dt + acc_noise;
    xt(IDX_THETA) += xt(IDX_OMEGA) * dt;

    // Measurement (angle with noise)
    double meas_theta = xt(IDX_THETA) + measNoise(rng);

    // Predict and update
    ekf.predict();
    VectorXd z(1);
    z << meas_theta;
    ekf.update(z);

    // Get estimates
    VectorXd x_est = ekf.getState();

    std::printf("%3d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f\n",
                t, xt(IDX_THETA), xt(IDX_OMEGA), x_est(IDX_THETA), x_est(IDX_OMEGA), meas_theta);
  }

  return 0;
}
