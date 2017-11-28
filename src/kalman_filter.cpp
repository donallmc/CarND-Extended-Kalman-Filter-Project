#include "kalman_filter.h"
#include <math.h>
#include <iostream>
#include "tools.h"

using namespace std; 
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in) {
  x_ = x_in;
  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1000, 0,
    0, 0, 0, 1000;

  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1;
  is_initialized_ = true;
}

void KalmanFilter::Predict(const float delta_t,
			   const float noise_ax,
			   const float noise_ay) {
  CheckInit();
  F_(0, 2) = delta_t;
  F_(1, 3) = delta_t;
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + ComputeProcessCovariance(delta_t,
							   noise_ax,
							   noise_ay);
}

MatrixXd KalmanFilter::ComputeProcessCovariance(const float delta_t,
						const float noise_ax,
						const float noise_ay) {
  float delta_t_2 = delta_t * delta_t;
  float delta_t_3 = pow(delta_t, 3) / 2.0;
  float delta_t_4 = pow(delta_t, 4) / 4.0;

  MatrixXd Q = MatrixXd(4, 4);
  Q << delta_t_4 * noise_ax, 0, delta_t_3 * noise_ax, 0,
    0, delta_t_4 * noise_ay, 0, delta_t_3 * noise_ay,
    delta_t_3 * noise_ax, 0, delta_t_2 * noise_ax, 0,
    0, delta_t_3 * noise_ay, 0, delta_t_2 * noise_ay;
  return Q;
}

void KalmanFilter::Update(const VectorXd &z, const MatrixXd &R, const MatrixXd &H) {
  CheckInit();
  VectorXd y = z - H * x_;
  ComputeNewEstimate(H, R, y);  
}

void KalmanFilter::UpdateEKF(const VectorXd &z, const MatrixXd &R) {
  CheckInit();
  MatrixXd H = tools.CalculateJacobian(x_);
  
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  VectorXd z_pred = VectorXd(3);
  z_pred << 0., 0., 0.;

  if (px != 0 && py != 0) {
    float rho = sqrt(px*px + py*py);
    float phi = atan2(py,px);
    float rho_dot = (px*vx + py*vy) / rho;
    z_pred << rho, phi, rho_dot;
  }

  VectorXd y = z - z_pred;

  while(y(1) < -M_PI || y(1) > M_PI ) {
    if (y(1) < -M_PI) {
      y(1) += 2*M_PI;
    }
    if (y(1) > M_PI) {
      y(1) -= 2*M_PI;
    }
  }

  ComputeNewEstimate(H, R, y);
}

void KalmanFilter::ComputeNewEstimate(const MatrixXd &H, const MatrixXd &R, const VectorXd y) {
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}

void KalmanFilter::CheckInit() {
  if (!is_initialized_) {
    throw std::runtime_error("Kalman Filter has not been correctly initialized.");
  }
}
