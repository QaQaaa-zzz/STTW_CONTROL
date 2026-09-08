#include "ECBC/eso_based_controller.h"

void EsoBasedController::setParams(const EbcParams &_params, const double &_speed_ms) {
  params = _params;
  for (double &gain : Kc) gain = 0;
  extended_states_observer_.setParams(SsParams{params.sample_time,4,1,0, true,
                                               {0,0,0,0},
                                               {{0,0,0,0},{0,0,1,0},{a2,a4,0,1},{0,0,0,0}},
                                               {{1},{0},{a1},{0}},
                                               {},
                                               {}});
  // disturbance_filter_.setParams(LpfParams{params.sample_time, 2.0},0);
  disturbance_filter_.setParams(LpfParams{params.sample_time, params.dist_lpf_freq},0);
  roll_vel_differ_.setParams(DiffParams{params.sample_time, 0.5});
  updateSystemParams(_speed_ms);
}

void EsoBasedController::updateSystemParams(const double &_speed_ms) {
  // double V = max(0.05, _speed_ms); // 径向速度
  double V = max(params.min_rear_vel_mps, _speed_ms); // 径向速度
  double D = bp.m * bp.a * bp.h;
  // 线性化模型的系数
  double M1 = - D * V * bp.clam/bp.b;
  double M2 = -(bp.m * V * V * bp.h - bp.m * bp.a * bp.c * bp.g) * bp.clam/bp.b;
  M2 = deadZone(M2, 0.1);
  double M3 = bp.Ib; // 绕地惯量
  double M4 = -bp.m * bp.g * bp.h;
  a1 = M1/M3;
  a2 = M2/M3;
  a4 = -M4/M3;
  Kroll2steer = M4/M2;

  Ko[0][0] = 30;
  Ko[1][1] = 3*params.wo;
  Ko[2][1] = 3*pow(params.wo,2) + a4;
  Ko[3][1] = pow(params.wo,3);
  double M[3][3] = {{(-a2*a2),(a1*a2),(-a1*a1)},
                    {(-a2*a4),(a1*a4),(-a2)},
                    {( a1*a4),(-a2),a1}};
  double det = a4*a1*a1 - a2*a2;
  // if (abs(det) < 5) {
  //   cout << "det: " << det << endl;
  //   cout << "speed(m/s): " << _speed_ms << endl;
  //   cout << "a124: " << a1 << ", " << a2 << ", " << a4 << endl;
  // }
  det = deadZone(det, 2);
  if (abs(det) > 5) {
    double bc[3] = {(3*params.wc),(3*pow(params.wc,2)+a4),pow(params.wc, 3)};
    for (int i = 0; i < 3; ++i) {
      Kc[i] = 0;
      for (int j = 0; j < 3; ++j) {
        Kc[i] += M[i][j] * bc[j] / det;
      }
      // cout << "EBC::Kc" << i << ": " << Kc[i] << endl;
    }
  }

  extended_states_observer_.p_.A[2][0] = a2;
  extended_states_observer_.p_.A[2][1] = a4;
  extended_states_observer_.p_.B[2][0] = a1;
}

void EsoBasedController::updateSystemParams(const double &_speed_ms, const double &_g) {
  // double V = max(0.05, _speed_ms); // 径向速度
  double V = max(params.min_rear_vel_mps, _speed_ms); // 径向速度
  double D = bp.m * bp.a * bp.h;
  // 线性化模型的系数
  double M1 = - D * V * bp.clam/bp.b;
  double M2 = -(bp.m * V * V * bp.h - bp.m * bp.a * bp.c * _g) * bp.clam/bp.b;
  M2 = deadZone(M2, 0.1);
  double M3 = bp.Ib; // 绕地惯量
  double M4 = -bp.m * _g * bp.h;
  a1 = M1/M3;
  a2 = M2/M3;
  a4 = -M4/M3;
  Kroll2steer = M4/M2;

  Ko[0][0] = 30;
  Ko[1][1] = 3*params.wo;
  Ko[2][1] = 3*pow(params.wo,2) + a4;
  Ko[3][1] = pow(params.wo,3);
  double M[3][3] = {{(-a2*a2),(a1*a2),(-a1*a1)},
                    {(-a2*a4),(a1*a4),(-a2)},
                    {( a1*a4),(-a2),a1}};
  double det = a4*a1*a1 - a2*a2;
  // if (abs(det) < 5) {
  //   cout << "det: " << det << endl;
  //   cout << "speed(m/s): " << _speed_ms << endl;
  //   cout << "a124: " << a1 << ", " << a2 << ", " << a4 << endl;
  // }
  det = deadZone(det, 2);
  if (abs(det) > 5) {
    double bc[3] = {(3*params.wc),(3*pow(params.wc,2)+a4),pow(params.wc, 3)};
    for (int i = 0; i < 3; ++i) {
      Kc[i] = 0;
      for (int j = 0; j < 3; ++j) {
        Kc[i] += M[i][j] * bc[j] / det;
      }
      // cout << "EBC::Kc" << i << ": " << Kc[i] << endl;
    }
  }

  extended_states_observer_.p_.A[2][0] = a2;
  extended_states_observer_.p_.A[2][1] = a4;
  extended_states_observer_.p_.B[2][0] = a1;
}

double EsoBasedController::getOutput(const double &_steer_ang, const double &_steer_vel, const double &_roll_ang, const double &_roll_vel, const double &_steer_ref_ang,
                 const bool &_enable_disturb_compen, const double &_max_steer_vel) {
  sys_states[0] = _steer_ang;
  sys_states[1] = _roll_ang;
  sys_states[2] = _roll_vel;
  roll_acc = roll_vel_differ_.getOutput(_roll_vel);
  extended_states_observer_.getState(eso_states);
  double disturbance2;
  filtered_disturbance2 = roll_acc - a2*_steer_ang - a4*_roll_ang - a1*_steer_vel;
  //filtered_disturbance2 = disturbance_filter_.getOutput(disturbance2);
  filtered_disturbance = disturbance_filter_.getOutput(eso_states[3]);
  /*
  cout << "ESO::eso_states[0]: " << eso_states[0] << endl
       << "ESO::eso_states[1]: " << eso_states[1] << endl
       << "ESO::eso_states[2]: " << eso_states[2] << endl
       << "ESO::eso_states[3]: " << eso_states[3] << endl;
  cout << "ESO::actual_steer : " << _steer_ang << endl;
  */
  for (int i = 0; i < 4; ++i) {
    eso_d_states_inputs[i] = 0;
    for (int j = 0; j < 2; ++j) {
      eso_d_states_inputs[i] += Ko[i][j] * (sys_states[j] - eso_states[j]);
    }
  }
  //
  sys_ref[0] = _steer_ref_ang;
  sys_ref[1] = sys_ref[0] / Kroll2steer;
  sys_ref[2] = 0;
  // 0.5s后再开启扰动补偿
  sys_states_comp[0] = 0;
  sys_states_comp[1] = 0;
  sys_states_comp[2] = 0;
  if (_enable_disturb_compen) {
    sys_states_comp[1] = -filtered_disturbance/a4;
  }
  sys_input = 0;
  for (int i = 0; i < 3; ++i) {
    sys_input += Kc[i] * (sys_ref[i] - (sys_states[i]-sys_states_comp[i]));
  }
  sys_input = saturation(sys_input, _max_steer_vel);
  //eso_inputs[0] = sys_input;
  eso_inputs[0] = _steer_vel;
  extended_states_observer_.setInput(eso_inputs,eso_d_states_inputs);
  return sys_input;
}

double EsoBasedController::getEstimatedDisturbance() {
  return filtered_disturbance;
}

double EsoBasedController::getEstimatedEquilibrium() {
  return -filtered_disturbance/a4;
}

void EsoBasedController::getEsoStates(double *req_states) {
  for (int i = 0; i < 4; i++)
  {
    req_states[i] = eso_states[i];
  }
  req_states[0] = filtered_disturbance2;
}

void EsoBasedController::getKc(double *req_array) {
  for (int i = 0; i < 3; i++)
  {
    req_array[i] = Kc[i];
  }
}

void EsoBasedController::getKo(double *req_array) {
//
}

void EsoBasedController::resetStates() {
  extended_states_observer_.resetStates();
  disturbance_filter_.resetState(0);
  roll_vel_differ_.setParams(DiffParams{params.sample_time, 0.5});
  filtered_disturbance = filtered_disturbance2 = roll_acc = 0;
  estimated_equilibrium_roll = sys_input = 0;
  for (double &value : eso_states) value = 0;
  for (double &value : Kc) value = 0;
  updateSystemParams(1.0);
}
// Restore explicitly supplied observer state after clearing transient history.
void EsoBasedController::resetStates(const double *_new_states) {
  resetStates();
  extended_states_observer_.resetStates(_new_states);
  disturbance_filter_.resetState(_new_states[3]);
}

void EsoBasedController::update() {
  extended_states_observer_.update();
  disturbance_filter_.update();
  roll_vel_differ_.update();
}
