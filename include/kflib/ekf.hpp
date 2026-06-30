#pragma once

#include "kflib/kf_base.hpp"

/**
 * @file ekf.hpp
 * @brief Extended Kalman filter (EKF) interface and callback types.
 *
 * Provides function-pointer types for nonlinear state/measurement
 * functions and their Jacobians, and a small `ExtendedKalmanFilter`
 * class that uses those callbacks to perform EKF predict/update steps.
 */

/**
 * @brief Signature for a function that returns the state Jacobian F.
 *
 * Returns the Jacobian matrix of the state transition with respect to the
 * state, given the current state and input.
 */
typedef Eigen::MatrixXd (*state_jacobian_function_t)(
    const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *userData);

/**
 * @brief Signature for a function that returns the measurement Jacobian H.
 */
typedef Eigen::MatrixXd (*measurement_jacobian_function_t)(
    const Eigen::VectorXd &state, const Eigen::VectorXd &input, void *userData);

/**
 * @brief Extended Kalman filter using user-supplied nonlinear models.
 *
 * Supply a `state_function_t` and `measurement_function_t` together with
 * their Jacobians to use this EKF. Derived behavior is implemented in the
 * corresponding source file; this header only exposes the setters and the
 * `predict`/`update` interface.
 */
class ExtendedKalmanFilter : public KalmanFilterBase {
 public:
  /** @brief Set the nonlinear state transition function f(x,u). */
  void setStateUpdateFunction(state_function_t stateUpdateFunction);
  /** @brief Set the nonlinear measurement function h(x,u). */
  void setMeasurementFunction(measurement_function_t measurementFuction);
  /** @brief Set the function that returns the state Jacobian F(x,u). */
  void setStateJacobian(state_jacobian_function_t functionThatReturnsF);
  /** @brief Set the function that returns the measurement Jacobian H(x,u). */
  void setMeasurementJacobian(
      measurement_jacobian_function_t functionThatReturnsH);

  using KalmanFilterBase::predict;
  /** @brief Predict step using the nonlinear state function and F Jacobian. */
  void predict(const Eigen::VectorXd &input) override;
  /** @brief Update step using the nonlinear measurement function and H Jacobian. */
  void update(const Eigen::VectorXd &measurements) override;

 private:
  /** Nonlinear state transition function f(x,u). */
  state_function_t stateFunction;
  /** Nonlinear measurement function h(x,u). */
  measurement_function_t measurementFunction;

  /** Jacobian of f with respect to state. */
  state_jacobian_function_t stateJacobian;
  /** Jacobian of h with respect to state. */
  measurement_jacobian_function_t measurementJacobian;
};
