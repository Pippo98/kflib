/**
 * @file kf_base.hpp
 * @brief Minimal base class and types for Kalman filter implementations.
 *
 * This header provides a lightweight abstract base class `KalmanFilterBase`
 * and function-pointer types used to describe system and measurement
 * transition functions. Implementations should derive from
 * `KalmanFilterBase` and provide concrete `predict` and `update` logic.
 *
 * Typical usage:
 * - construct a derived filter,
 * - call `setState`, `setStateCovariance`, `setProcessCovariance`, and
 *   `setMeasurementCovariance` to initialize, then
 * - repeatedly call `predict(input)` and `update(measurements)`.
 */

#pragma once

#include <Eigen/Core>
#include <ostream>

/**
 * @brief Signature for the state transition function.
 *
 * The function receives the current state and an optional input vector and
 * returns the predicted next state. `userData` is an opaque pointer for
 * extra context.
 */
typedef Eigen::VectorXd (*state_function_t)(const Eigen::VectorXd &state,
                                            const Eigen::VectorXd &input,
                                            void *userData);

/**
 * @brief Signature for the measurement function.
 *
 * Given the current state (and optional inputs) returns the expected
 * measurement vector.
 */
typedef Eigen::VectorXd (*measurement_function_t)(const Eigen::VectorXd &state,
                                                  const Eigen::VectorXd &input,
                                                  void *userData);

/**
 * @brief Abstract base class for Kalman filters.
 *
 * Provides common storage for state, covariance, and noise matrices and
 * declares the core `predict`/`update` interface. Concrete filters must
 * implement `predict(const Eigen::VectorXd &input)` and
 * `update(const Eigen::VectorXd &measurements)`.
 */
class KalmanFilterBase {
public:
  /**
   * @brief Attach user-defined context passed to function callbacks.
   * @param data Opaque pointer stored as `userData`.
   */
  void setUserData(void *data);

  /** @brief Set the current state vector. */
  void setState(const Eigen::VectorXd &state);
  /** @brief Set the state covariance matrix P. */
  void setStateCovariance(const Eigen::MatrixXd &stateCovariance);
  /** @brief Set the process noise covariance matrix Q. */
  void setProcessCovariance(const Eigen::MatrixXd &processCovariance);
  /** @brief Set the measurement noise covariance matrix R. */
  void setMeasurementCovariance(const Eigen::MatrixXd &measurementCovariance);

  /** @brief Convenience overload that calls `predict` with empty input. */
  void predict() { predict(Eigen::VectorXd()); }
  /** @brief Predict step; must be implemented by derived filters. */
  virtual void predict(const Eigen::VectorXd &input) = 0;
  /** @brief Update step; must be implemented by derived filters. */
  virtual void update(const Eigen::VectorXd &measurements) = 0;

  /** @brief Return the current state vector (const reference). */
  const Eigen::VectorXd &getState() const;
  /** @brief Return the current state covariance matrix (const reference). */
  const Eigen::MatrixXd &getCovariance() const;
  /** @brief Return the most recent inputs vector (const reference). */
  const Eigen::VectorXd &getInputs() const;

  /** @brief Print a short human-readable summary to stdout. */
  void print() const;
  /** @brief Print a short human-readable summary to the given stream. */
  void printToStream(std::ostream &stream) const;

protected:
  /** Opaque pointer available to user callbacks. */
  void *userData;

  /** Current state vector X. */
  Eigen::VectorXd X; // State
  /** Last used inputs vector U. */
  Eigen::VectorXd U; // Inputs
  /** State covariance matrix P. */
  Eigen::MatrixXd P; // State covariance
  /** Process noise covariance Q. */
  Eigen::MatrixXd Q; // Process covariance
  /** Measurement noise covariance R. */
  Eigen::MatrixXd R; // Measurement covariance
};
