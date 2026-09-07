#include "ECBC/pid_controller.h"

double deadZone(const double &_input, const double &_dead_zone) {
  int sign = (_input < 0)? -1:1;
  return sign * max(abs(_input), _dead_zone);
}

double saturation(const double &_input, const double &_threshold) {
  int sign = (_input < 0)? -1:1;
  return sign * min(abs(_input), _threshold);
}

/**********************************************************
 *
 * PID控制器
 *
***********************************************************/
PidController::PidController() {
  is_enabled_ = true;
}

PidController::PidController(const PidParams &_pid_params) {
  setParams(_pid_params, true);
}

void PidController::setParams(const PidParams &_pid_params, const bool &_reset_integrator) {
  is_enabled_ = true;
  params_ = _pid_params;
  if (_reset_integrator) pos_err_int_ = 0;
  // 初始化微分器
  DiffParams diff_params = {params_.sample_time, params_.diff_filter_time};
  differentiator_.setParams(diff_params);
  // 初始化积分器
  IntParams int_params = {params_.sample_time};
  integrator_.setParams(int_params, _reset_integrator);
}

void PidController::resetIntegrator() {
  integrator_.reset();
  pos_err_int_ = 0;
}

void PidController::enable() {
  is_enabled_ = true;
  resetIntegrator();
}

void PidController::disable() {
  is_enabled_ = false;
}

double PidController::getOutput(const double &_ref_pos, const double &_ref_vel,
                                const double &_actual_pos, const double &_actual_vel) {
  inner_diffor_enabled = false;
  pos_err_ = _ref_pos - _actual_pos;
  if (is_enabled_) {
    vel_err_ = _ref_vel - _actual_vel;
    pos_err_int_ = integrator_.getOutput(pos_err_);
    output_ = pos_err_ * params_.kp + pos_err_int_ * params_.ki + vel_err_ * params_.kd + _ref_pos * params_.kf;
  } else {
    output_ = 0;
  }
  return output_;
}

double PidController::getOutput(const double &_ref_pos, const double &_actual_pos) {
  inner_diffor_enabled = true;
  pos_err_ = _ref_pos - _actual_pos;
  if (is_enabled_) {
    vel_err_ = differentiator_.getOutput(pos_err_);
    pos_err_int_ = integrator_.getOutput(pos_err_);
    output_ = pos_err_ * params_.kp + pos_err_int_ * params_.ki + vel_err_ * params_.kd + _ref_pos * params_.kf;

  } else {
    output_ = 0;
  }
  return output_;
}

void PidController::update() {
  if (inner_diffor_enabled) {
    differentiator_.update();
  }
  integrator_.update();
}

/**********************************************************
 *
 * 微分器：差分器+低通滤波器
 *
***********************************************************/
Derivative::Derivative() {
  f_output_ = 0;
  last_f_output_ = 0;
  output_ = 0;
  is_first_step_ = true;
}

Derivative::Derivative(const DiffParams &_diff_params) {
  setParams(_diff_params);
}

void Derivative::setParams(const DiffParams &_diff_params) {
  f_output_ = 0;
  last_f_output_ = 0;
  output_ = 0;
  is_first_step_ = true;

  params_ = _diff_params;
  LpfParams lpf_params = {params_.sample_time, 1.0/params_.diff_filter_time};
  filter_.setParams(lpf_params,0);
}

void Derivative::setInput(const double &_input) {
  input_ = _input;
  filter_.setInput(_input);
}

double Derivative::getOutput() {
  if (is_first_step_) {
    output_ = 0;
    is_first_step_ = false;
  } else {
    output_ = (input_ - last_input_) / params_.sample_time;
  }
  f_output_ = filter_.getOutput(output_);
  return f_output_;
}

double Derivative::getOutput(const double &_input) {
  setInput(_input);
  return getOutput();
}

void Derivative::update() {
  filter_.update();
  last_input_ = input_;
  last_f_output_ = f_output_;
}

/**********************************************************
 *
 * 连续微分器，传递函数法，比离散超前一个步长。
 *
***********************************************************/
DerivativeC::DerivativeC() {
  setParams(DiffParams {0.005, 0.01});
}

DerivativeC::DerivativeC(const DiffParams &_diff_params) {
  setParams(_diff_params);
}

void DerivativeC::setParams(const DiffParams &_diff_params) {
  input_ = 0;
  output_ = 0;
  params_ = _diff_params;
  Tf2ndParams tf2_params = {params_.sample_time, 1, params_.diff_filter_time, 0,
                            0, 1, 0};
  double init_state[2] = {0, 0};
  tf2nd_.setParams(tf2_params, init_state);
}

void DerivativeC::update() {
  tf2nd_.update();
}

double DerivativeC::getOutput(const double &_input) {
  input_ = _input;
  output_ = tf2nd_.getOutput(input_);
  return output_;
}

/**********************************************************
 *
 * 积分器
 *
***********************************************************/
Integrator::Integrator() {
  integral_ = 0;
  input_ = 0;
}

Integrator::Integrator(const IntParams &_int_params) {
  setParams(_int_params, true);
}

void Integrator::setParams(const IntParams &_int_params, const bool &_reset_integrator) {
  params_ = _int_params;
  if (_reset_integrator) {
    reset();
  }
}

void Integrator::reset() {
  input_ = 0;
  integral_ = 0;
}

double Integrator::update() {
  integral_ += input_ * params_.sample_time;
  return integral_;
}

void Integrator::setInput(const double &_input) {
  input_ = _input;
}

double Integrator::getOutput() {
  return integral_;
}

double Integrator::getOutput(const double &_input) {
  setInput(_input);
  return getOutput();
}

/**********************************************************
 *
 * 低通滤波器
 *
***********************************************************/
LowPassFilter::LowPassFilter()
{
  input_ = 0;     //输入变量
  output_ = 0;
  a_ = 0;
  b_ = 1;
  is_enabled = false;
  state_ = 0;
  params_ = LpfParams {0.005, 0};
}

LowPassFilter::LowPassFilter(const LpfParams &_lpf_params) {
  input_ = 0;     //输入变量
  output_ = 0;
  a_ = 0;
  b_ = 1;
  setParams(_lpf_params, 0);
}

/* low_cutoff_freq_ > 0 时滤波器有效 */
void LowPassFilter::setParams(const LpfParams &lpf_params, const double &_init_state)
{
  if (lpf_params.low_cutoff_freq > 0)  {
    params_ = lpf_params;
    is_enabled = true;
    state_ = _init_state;
    calculateAB();
  } else {
    is_enabled = false;
  }
}

/* 打开滤波器 */
void LowPassFilter::enable(const double &_init_state)
{
  is_enabled = true;
  state_ = _init_state;
  calculateAB();
}

/* 关闭滤波器 */
void LowPassFilter::disable()
{
  is_enabled = false;
  calculateAB();
}

void LowPassFilter::calculateAB()
{
  if (is_enabled) {
//    a_ = 1 - p_.low_cutoff_freq * p_.sample_time;
//    b_ = p_.low_cutoff_freq * p_.sample_time;
    a_ = 1 / (1 + params_.low_cutoff_freq * params_.sample_time);
    b_ = 1 - a_;
  } else {
    a_ = 0;
    b_ = 1;
  }
}

void LowPassFilter::setInput(const double &_input) {
  input_ = _input;
}

void LowPassFilter::resetState() {
  state_ = 0;
}

void LowPassFilter::resetState(const double &_new_state) {
  state_ = _new_state;
}

double LowPassFilter::getOutput() {
  if (is_enabled) {
    output_ = state_;
  } else {
    output_ = input_;
  }
  return output_;
}

double LowPassFilter::getOutput(const double &_input) {
  setInput(_input);
  return getOutput();
}

void LowPassFilter::update() {
  if (is_enabled) {
    // state_ = a_ * state_ + b_ * input_;
    d_state_ = params_.low_cutoff_freq * (input_ - state_);
    state_ += d_state_ * params_.sample_time;
  } else {
    state_ = input_;
  }
}

TransferFcn2nd::TransferFcn2nd() {
  input_ = 0;     //输入变量
  output_ = 0;
  state_[0] = 0;
  state_[1] = 0;
  d_state_[0] = 0;
  d_state_[1] = 0;
  is_enabled = false;
  params_ = Tf2ndParams {0.005, 1,0,0,1,0,0};
}

TransferFcn2nd::TransferFcn2nd(const Tf2ndParams &_tf2_params, const double _init_state[2]) {
  input_ = 0;     //输入变量
  output_ = 0;
  state_[0] = 0;
  state_[1] = 0;
  d_state_[0] = 0;
  d_state_[1] = 0;
  setParams(_tf2_params, _init_state);
}

void TransferFcn2nd::setParams(const Tf2ndParams &_tf2_params, const double _init_state[2]) {
  params_ = _tf2_params;
  processParams();
  for (int i = 0; i < 2; ++i) {
    state_[i] = _init_state[i];
  }
  is_enabled = true;
}

void TransferFcn2nd::enable(const double _init_state[2]) {
  for (int i = 0; i < 2; ++i) {
    state_[i] = _init_state[i];
  }
  is_enabled = true;
}

void TransferFcn2nd::disable() {
  is_enabled = false;
}

void TransferFcn2nd::update() {
  if (order_ == 2) {
    d_state_[0] = state_[1];
    d_state_[1] = - params_.den[0] * state_[0] - params_.den[1] * state_[1] + input_;
    state_[0] += d_state_[0] * params_.sample_time;
    state_[1] += d_state_[1] * params_.sample_time;
  } else if (order_ == 1) {
    d_state_[0] = - params_.den[0] * state_[0] + input_;
    state_[0] += d_state_[0] * params_.sample_time;
  }
}

void TransferFcn2nd::setInput(const double &_input) {
  input_ = _input;
}

double TransferFcn2nd::getOutput() {
  if (order_ == 2) {
    output_ = (params_.num[0] - params_.den[0]*params_.num[order_]) * state_[0] +
              (params_.num[1] - params_.den[1]*params_.num[order_]) * state_[1] +
              + params_.num[order_] * input_;
  } else if (order_ == 1) {
    output_ = (params_.num[0] - params_.den[0]*params_.num[order_]) * state_[0] +
              + params_.num[order_] * input_;
  } else {
    output_ = params_.num[0] * input_;
  }
  return output_;
}

double TransferFcn2nd::getOutput(const double &_input) {
  setInput(_input);
  return getOutput();
}

void TransferFcn2nd::processParams() {
  int den_order = getPolyOrder(params_.den,3);
  int num_order = getPolyOrder(params_.num,3);
  if ((den_order == -1) || (num_order == -1)) {
    throw std::string("TransferFcn2nd:: The coefficients cannot all be zero.");
  }
  if (num_order > den_order) {
    throw std::string("TransferFcn2nd:: The order of the numerator should be less than that of the denominator.");
  }
  order_ = den_order;
  for (int i = 0; i < num_order + 1; ++i) {
    params_.num[i] = params_.num[i] / params_.den[order_];
  }
  for (int i = 0; i < den_order + 1; ++i) {
    params_.den[i] = params_.den[i] / params_.den[order_];
  }
  // std::cout << "order: " << order_ << std::endl;
  // std::cout << "den  : " << params_.den[0] << ' ' << params_.den[1] << ' ' << params_.den[2] << std::endl;
  // std::cout << "num  : " << params_.num[0] << ' ' << params_.num[1] << ' ' << params_.num[2] << std::endl;
}

int TransferFcn2nd::getPolyOrder(const double _poly[], const int &_num) {
  for (int i = 0; i < _num; ++i) {
    if (_poly[_num - 1 - i] != 0)
      return (_num - 1 - i);
  }
  return -1;
}

StateSpace::StateSpace() {
}

StateSpace::StateSpace(const SsParams &_params) {
  p_ = _params;
  states_ = vector<double> (p_.state_num);
  d_states_ = vector<double> (p_.state_num);
  for (int i = 0; i < p_.state_num; ++i) {
    states_[i] = p_.init_states[i];
  }
}

void StateSpace::setParams(const SsParams &_params) {
  p_ = _params;
  states_ = vector<double> (p_.state_num);
  d_states_ = vector<double> (p_.state_num);
  for (int i = 0; i < p_.state_num; ++i) {
    states_[i] = p_.init_states[i];
  }
}

void StateSpace::resetStates(const double *_new_states) {
  for (int i = 0; i < p_.state_num; ++i) {
    states_[i] = _new_states[i];
  }
  for (int i = 0; i < p_.state_num; ++i) {
    d_states_[i] = 0;
  }
}

void StateSpace::resetStates() {
  for (int i = 0; i < p_.state_num; ++i) {
    states_[i] = 0;
  }
  for (int i = 0; i < p_.state_num; ++i) {
    d_states_[i] = 0;
  }
}

void StateSpace::getState(double *_states_req, const double *_inputs, const double *_d_states_inputs) {
  for (int i = 0; i < p_.state_num; ++i) {
    *(_states_req + i) = states_[i];
  }
  for (int i = 0; i < p_.state_num; ++i) {
    d_states_[i] = 0;
    for (int j = 0; j < p_.state_num; ++j) {
      d_states_[i] += p_.A[i][j] * states_[j];
    }
    for (int k = 0; k < p_.input_num; ++k) {
      d_states_[i] += p_.B[i][k] * _inputs[k];
    }
    if (p_.d_states_input_enabled) {
      d_states_[i] += *(_d_states_inputs + i);
    }
  }
}

void StateSpace::getState(double *_states_req) {
  for (int i = 0; i < p_.state_num; ++i) {
    *(_states_req + i) = states_[i];
  }
}

void StateSpace::setInput( const double *_inputs, const double *_d_states_inputs) {
  for (int i = 0; i < p_.state_num; ++i) {
    d_states_[i] = 0;
    for (int j = 0; j < p_.state_num; ++j) {
      d_states_[i] += p_.A[i][j] * states_[j];
    }
    for (int k = 0; k < p_.input_num; ++k) {
      d_states_[i] += p_.B[i][k] * _inputs[k];
    }
    if (p_.d_states_input_enabled) {
      d_states_[i] += *(_d_states_inputs + i);
    }
  }
}
void StateSpace::update() {
  for (int i = 0; i < p_.state_num; ++i) {
    states_[i] += d_states_[i] * p_.sample_time;
  }
}
