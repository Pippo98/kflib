#pragma once

#include "kf_base.hpp"

/**
 * @brief Simple linear Kalman filter implementation.
 *
 * Implements the standard discrete-time linear Kalman filter with optional
 * control inputs. The heavy-lifting math is implemented in the .cpp file;
 * this class stores the system matrices and exposes setters for them.
 */
class KalmanFilter : public KalmanFilterBase {
 public:
  /**
   * @brief Set the state transition matrix A (X_k = A * X_{k-1}).
   * @param A State transition matrix.
   */
  void setStateUpdateMatrix(const Eigen::MatrixXd &A);

  /**
   * @brief Set the state transition matrix A and control matrix B.
   * @param A State transition matrix.
   * @param B Control input matrix (applied to the input vector).
   */
  void setStateUpdateMatrices(const Eigen::MatrixXd &A,
                              const Eigen::MatrixXd &B);

  /** @brief Set the measurement matrix H (z = H * x). */
  void setMeasurementMatrix(const Eigen::MatrixXd &H);

  using KalmanFilterBase::predict;
  /** @brief Predict step using optional input vector. */
  void predict(const Eigen::VectorXd &input) override;
  /** @brief Update step with a measurement vector. */
  void update(const Eigen::VectorXd &measurements) override;

 private:
  /** State transition matrix. */
  Eigen::MatrixXd A;
  /** Control input matrix (optional). */
  Eigen::MatrixXd B;
  /** Measurement matrix. */
  Eigen::MatrixXd H;
};
