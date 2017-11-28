#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"
#include "tools.h"

class KalmanFilter {

public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  //whether filter has been correctly initialized
  bool is_initialized_ = false;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   */
  void Init(Eigen::VectorXd &x_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_t Time between k and k+1 in s
   * @param noise_ax X-acceleration noise
   * @param noise_ay Y-acceleration noise
   */
  void Predict(const float delta_t, const float noise_ax, const float noise_ay);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   * @param R Measurement covariance matrix
   * @param H Measurement matrix
   */
  void Update(const Eigen::VectorXd &z, const Eigen::MatrixXd &R, const Eigen::MatrixXd &H);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   * @param R Measurement covariance matrix
   */
  void UpdateEKF(const Eigen::VectorXd &z, const Eigen::MatrixXd &R);

private:

  //for computing Jacobians
  Tools tools;

  /**
   * Computes the Process Covariance Matrix
   * @param delta_t Time between k and k+1 in s
   * @param noise_ax X-acceleration noise
   * @param noise_ay Y-acceleration noise
   */
  Eigen::MatrixXd ComputeProcessCovariance(const float delta_t, const float noise_ax, const float noise_ay);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param H Measurement matrix
   * @param R Measurement covariance matrix
   * @param y delta between expected and measured
   */  
  void ComputeNewEstimate(const MatrixXd &H, const MatrixXd &R, const VectorXd y);  

  /**
   * Checks whether is_initialized_ has been set to true an throws an exception otherwise.
   */
  void CheckInit();
};

#endif /* KALMAN_FILTER_H_ */
