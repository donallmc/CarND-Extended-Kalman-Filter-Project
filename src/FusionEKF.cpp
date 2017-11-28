#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  previous_timestamp_ = 0;
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
        0, 0.0225;  
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
    0, 1, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  if (!ekf_.is_initialized_) {
    InitializeEKF(measurement_pack);
  } else {
    Predict(measurement_pack);
    Update(measurement_pack);
    PrintStateAndCovariance();
  }
}

void FusionEKF::InitializeEKF(const MeasurementPackage &measurement_pack) {    
  float px = 0.0, py = 0.0;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    float rho = measurement_pack.raw_measurements_[0];
    float phi = measurement_pack.raw_measurements_[1];
    px = rho * cos(phi);
    py = rho * sin(phi);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    px = measurement_pack.raw_measurements_[0];
    py = measurement_pack.raw_measurements_[1];
  }

  VectorXd x = VectorXd(4);
  x << px, py, 0.0, 0.0;  
  ekf_.Init(x);
    
  previous_timestamp_ = measurement_pack.timestamp_;
}

void FusionEKF::Predict(const MeasurementPackage &measurement_pack) {
  float delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.Predict(delta_t, noise_ax, noise_ay);
}

void FusionEKF::Update(const MeasurementPackage &measurement_pack) {
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, R_radar_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.Update(measurement_pack.raw_measurements_, R_laser_, H_laser_);
  } else {
    throw std::invalid_argument("Invalid measurement type:" + measurement_pack.sensor_type_);
  }  
}

void FusionEKF::PrintStateAndCovariance() {
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
