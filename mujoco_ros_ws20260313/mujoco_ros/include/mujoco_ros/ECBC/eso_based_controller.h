#ifndef JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_ESO_BASED_CONTROLLER_H
#define JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_ESO_BASED_CONTROLLER_H

#include "pid_controller.h"
#include "model_params.h"
#include <iostream>
#include <cmath>

struct EbcParams {
  double sample_time;
  double wo; // 观测器极点, 30
  double wc; // 控制器极点, 10
  double dist_lpf_freq;
  double min_rear_vel_mps;
};

class EsoBasedController {
public:
  void setParams(const EbcParams &_params, const double &_speed_ms);
  void updateSystemParams(const double &_speed_ms);
  void updateSystemParams(const double &_speed_ms, const double &_g);
  double getOutput(const double &_steer_ang, const double &_steer_vel, const double &_roll_ang, const double &_roll_vel, const double &_steer_ref_ang,
                   const bool &_enable_disturb_compen, const double &_max_steer_vel);
  double getEstimatedDisturbance();
  double getEstimatedEquilibrium();
  void getEsoStates(double *req_states);
  void getKc(double *req_array);
  void getKo(double *req_array);
  void resetStates();
  void resetStates(const double *_new_states);
  void update();

private:
  StateSpace extended_states_observer_;
  LowPassFilter disturbance_filter_;
  DerivativeC roll_vel_differ_;
  EbcParams params;
  double Ko[4][2] = {0};
  double Kc[3] = {0};
  double eso_inputs[1] = {0};
  double eso_states[4] = {0};
  double eso_d_states_inputs[4] = {0};
  double sys_states[3] = {0};
  double sys_ref[3] = {0};
  double sys_states_comp[3] = {0};
  double sys_input = 0;
  double estimated_equilibrium_roll = 0;
  double filtered_disturbance = 0;
  double filtered_disturbance2 = 0;
  double roll_acc = 0;

  double a1 = 0;
  double a2 = 0;
  double a4 = 0;
  double Kroll2steer;
  BikeModelParams bp;
};


#endif //JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_ESO_BASED_CONTROLLER_H
