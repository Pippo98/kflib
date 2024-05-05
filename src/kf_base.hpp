#pragma once

#include <Eigen/Core>
#include <ostream>

#include "eigen/Eigen/src/Core/Matrix.h"

typedef Eigen::VectorXd (*state_function_t)(const Eigen::VectorXd &state,
                                            const Eigen::VectorXd &input,
                                            void *userData);
typedef Eigen::VectorXd (*measurement_function_t)(const Eigen::VectorXd &state,
                                                  const Eigen::VectorXd &input,
                                                  void *userData);

class KalmanFilterBase {
public:
  void setUserData(void *data);

  void setState(const Eigen::VectorXd &state);
  void setStateCovariance(const Eigen::MatrixXd &stateCovariance);
  void setProcessCovariance(const Eigen::MatrixXd &processCovariance);
  void setMeasurementCovariance(const Eigen::MatrixXd &measurementCovariance);

  void predict() { predict(Eigen::VectorXd()); }
  virtual void predict(const Eigen::VectorXd &input) = 0;
  virtual void update(const Eigen::VectorXd &measurements) = 0;

  Eigen::VectorXd getState();
  Eigen::MatrixXd getCovariance();

  void print();
  void printToStream(std::ostream &stream);

protected:
  void *userData;

  Eigen::VectorXd X; // State
  Eigen::MatrixXd P; // State covariance
  Eigen::MatrixXd Q; // Process covariance
  Eigen::MatrixXd R; // Measurement covariance
};
