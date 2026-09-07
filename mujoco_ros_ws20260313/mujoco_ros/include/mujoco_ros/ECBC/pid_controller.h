#ifndef JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_PID_CONTROLLER_H
#define JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_PID_CONTROLLER_H

#include <iostream>
#include <vector>
using namespace std;

double deadZone(const double &_input, const double &_dead_zone);
double saturation(const double &_input, const double &_threshold);

struct PidParams {
  double sample_time;
  double kp;
  double ki;
  double kd;
  double kf;
  double diff_filter_time;
  //PidParams(): sample_time(0.005), kp(1), ki(0), kd(0), diff_filter_time(0.01) {};
};

struct DiffParams {
  double sample_time;
  double diff_filter_time;
  //DiffParams(): sample_time(0.005), diff_filter_time(0.01) {};
};

struct IntParams {
  double sample_time;
  //IntParams(): sample_time(0.005) {};
};

struct LpfParams {
  double sample_time;
  double low_cutoff_freq;
  //LpfParams(): low_cutoff_freq(1), sample_time(0.005) {};
};

struct Tf2ndParams {
  double sample_time;
  double den[3];
  double num[3];
  Tf2ndParams & operator= (const Tf2ndParams &_right) {
    if (this != &_right) {
      this->sample_time = _right.sample_time;
      for (int i = 0; i < 3; ++i) {
        this->den[i] = _right.den[i];
        this->num[i] = _right.num[i];
      }
    }
    return *this;
  }
};


class TransferFcn2nd {
public:
  TransferFcn2nd();
  TransferFcn2nd(const Tf2ndParams &_tf2_params, const double _init_state[2]);
  void setParams(const Tf2ndParams &_tf2_params, const double _init_state[2]);
  void enable(const double _init_state[2]);
  void disable();
  void update();
  void setInput(const double &_input);
  double getOutput();
  double getOutput(const double &_input);
  static int getPolyOrder(const double _poly[], const int &_num);

  Tf2ndParams params_;

  bool is_enabled;
  int order_;
  double input_;     //输入变量
  double state_[2];  //实际输出变量，即采样回来的输出变量: 实际倾斜角
  double d_state_[2];
  double output_;
//  double last_input_;
//  double last_state_;

private:
  void processParams();

};

class LowPassFilter {
public:
  LowPassFilter();
  LowPassFilter(const LpfParams &_lpf_params);
  void setParams(const LpfParams &_lpf_params, const double &_init_state);
  void enable(const double &_init_state);
  void disable();
  void update();
  void setInput(const double &_input);
  void resetState();
  void resetState(const double &_new_state);
  double getOutput();
  double getOutput(const double &_input);

  LpfParams params_;

  bool is_enabled;
  double a_;
  double b_;
  double input_;  //输入变量
  double state_;  //实际输出变量，即采样回来的输出变量: 实际倾斜角
  double d_state_;
  double output_;
//  double last_input_;
//  double last_state_;

private:
  void calculateAB();
};

class Derivative {
public:
  Derivative();
  Derivative(const DiffParams &_diff_params);
  void setParams(const DiffParams &_diff_params);
  void update();
  void setInput(const double &_input);
  double getOutput();
  double getOutput(const double &_input);
  DiffParams params_;
  LowPassFilter filter_;
  double input_;
  double last_input_;
  double f_output_;
  double last_f_output_;
  double output_;
  bool is_first_step_;
private:
};

class DerivativeC {
public:
  DerivativeC();
  DerivativeC(const DiffParams &_diff_params);
  void setParams(const DiffParams &_diff_params);
  void update();
  double getOutput(const double &_input);
  DiffParams params_;
  TransferFcn2nd tf2nd_;
  double input_;
  double output_;
private:
};

class Integrator {
public:
  Integrator();
  Integrator(const IntParams &_int_params);
  void setParams(const IntParams &_int_params, const bool &_reset_integrator);
  void reset();
  double update();
  void setInput(const double &_input);
  double getOutput();
  double getOutput(const double &_input);
  IntParams params_;
  double integral_;
  double input_;
private:
};

class PidController {
public:
  PidController();
  PidController(const PidParams &_pid_params);
  void setParams(const PidParams &_pid_params, const bool &_reset_integrator);
  void resetIntegrator();
  void enable();
  void disable();
  double getOutput(const double &_ref_pos, const double &_actual_pos);
  double getOutput(const double &_ref_pos, const double &_ref_vel,
                   const double &_actual_pos, const double &_actual_vel);
//  void setInput(const double &_ref_pos, const double &_ref_vel,
//                const double &_actual_pos, const double &_actual_vel);
//  void setInput(const double &_ref_pos, const double &_actual_pos);
//  double getOutput();

  void update();
  PidParams params_;
  DerivativeC differentiator_;
  Integrator integrator_;
//  double ref_pos_;
//  double ref_vel_;
//  double actual_pos_;
//  double actual_vel_;
  double pos_err_;
  double vel_err_;
  double pos_err_int_;
  double output_;
  bool is_enabled_;
  bool inner_diffor_enabled;

private:

};

struct SsParams {
  double sample_time;
  int state_num;
  int input_num;
  int output_num;
  bool d_states_input_enabled;
  vector<double> init_states;
  vector<vector<double>> A;
  vector<vector<double>> B;
  vector<vector<double>> C;
  vector<vector<double>> D;
//  SsParams() {
//    init_states = vector<double>(state_num);
//    A = vector<vector<double>>(state_num, vector<double>(state_num));
//    B = vector<vector<double>>(state_num, vector<double>(input_num));
//    C = vector<vector<double>>(output_num, vector<double>(state_num));
//    D = vector<vector<double>>(output_num, vector<double>(input_num));
//  }
};

class StateSpace {
public:
  StateSpace();
  StateSpace(const SsParams &_params);
  void setParams(const SsParams &_params);
  void resetStates(const double *_new_states);
  void resetStates();
  void getState(double *_states_req, const double *_inputs, const double *_d_states_inputs);
  void getState(double *_states_req);
  void setInput(const double *_inputs, const double *_d_states_inputs);
  void update();
  SsParams p_;
private:
  vector<double> states_;
  vector<double> d_states_;
};


#endif //JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_PID_CONTROLLER_H
