#include <ros/ros.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <filesystem>
#include <ros/package.h>
#include "mujoco_ros/RobotState.h"
#include "mujoco_ros/ControlCmd.h"
#include "mujoco_ros/BoardCmd.h"
#include "mujoco_ros/ResidualCmd.h"
#include "ECBC/eso_based_controller.h"
#include "ECBC/residual_command.h"
#include <yaml-cpp/yaml.h>
#include "data_logger.h"

// 全局变量（缓存最新机器人状态）
mujoco_ros::RobotState latest_state;
bool first_state = false;
double control_start_time = 0;

int control_freq;
sttw::Limits command_limits;
sttw::Command residual_action;
bool residual_enabled=false, received_residual=false;
ros::WallTime state_received, residual_received;
double state_timeout=.05, residual_timeout=.05;
double eso_start_seconds=3.;
bool controller_needs_reset=false;

void residualCallback(const mujoco_ros::ResidualCmd::ConstPtr& msg) {
  residual_action={msg->steer_rate_residual,msg->rear_rate_residual};
  residual_received=ros::WallTime::now();
  const double age=(ros::Time::now()-msg->header.stamp).toSec();
  received_residual=!msg->header.stamp.isZero() && age>=0 && age<=residual_timeout;
}

// ECBC controller
EsoBasedController eso_based_controller;
LowPassFilter rear_vel_filter;
double eso_states[4] = {0};
EbcParams ebcparams;
LpfParams rearLPFparams;

int mode;
double target_steer_pos;
double target_forw_vel;


std::string log_dir;
std::string log_file_name;
bool startlog = true;
DataLogger logger;

// 机器人状态回调函数（缓存最新状态）
void stateCallback(const mujoco_ros::RobotState::ConstPtr& msg) {
  for (double value : {msg->time,msg->steer_pos,msg->steer_vel,msg->roll_ang,msg->roll_vel,
                       msg->yaw_vel,msg->rear_vel,msg->front_vel}) {
    if (!std::isfinite(value)) {
      ROS_WARN_THROTTLE(1.0,"Ignoring nonfinite robot state");
      controller_needs_reset=true;
      received_residual=false;
      state_received=ros::WallTime();
      return;
    }
  }
  if (first_state && msg->time<latest_state.time) controller_needs_reset=true;
  state_received=ros::WallTime::now();
  latest_state = *msg;
  
  if (!first_state){
    control_start_time = latest_state.time;
    std::cout << "控制开始时刻：" << control_start_time << " 秒" << std::endl;
    first_state = true;
  } 
  
}

// 缓存最新指令
void boardcmdCallback(const mujoco_ros::BoardCmd::ConstPtr& msg){
  if ((msg->mode!=0 && msg->mode!=1 && msg->mode!=9) || !std::isfinite(msg->target_steer_pos) ||
      !std::isfinite(msg->target_rear_vel)) {
    ROS_WARN("Ignoring invalid board command");
    return;
  }
  mode = msg->mode;
  target_steer_pos = msg->target_steer_pos;
  target_forw_vel = msg->target_rear_vel;

}

// 控制算法
mujoco_ros::ControlCmd computeControl() {
  mujoco_ros::ControlCmd cmd;

  // 获取最新状态
  double steer_pos = latest_state.steer_pos;
  double roll_ang = latest_state.roll_ang;
  double steer_vel = latest_state.steer_vel;
  double roll_vel = latest_state.roll_vel;
  double yaw_vel = latest_state.yaw_vel;
  double rear_vel = latest_state.rear_vel;
  double front_vel = latest_state.front_vel;
  double t = latest_state.time - control_start_time;

  bool enable_disturb_compen = false;
  if (first_state && t > eso_start_seconds){
    enable_disturb_compen = true;
  }
  // if (first_state && t > 9){
  //   target_steer_pos = 0;
  // }
  
  // target_steer_pos = -5.0*M_PI/180;
  // target_forw_vel = 6/3.6/0.1;

  eso_based_controller.updateSystemParams(rear_vel*0.1);
  double target_steer_vel = eso_based_controller.getOutput(steer_pos, steer_vel, -roll_ang, -roll_vel,
                  target_steer_pos, enable_disturb_compen, 4.0);
  target_steer_vel = saturation(target_steer_vel, 4.0);
  eso_based_controller.getEsoStates(eso_states);



  const bool residual_fresh=residual_enabled && received_residual &&
      (ros::WallTime::now()-residual_received).toSec()<=residual_timeout;
  const auto final_command=sttw::composeCommand({target_steer_vel,target_forw_vel},
      residual_action,steer_pos,1.0/control_freq,command_limits,residual_fresh);
  cmd.rear_vel = final_command.rear;
  cmd.steer_vel = final_command.steer;

  eso_based_controller.update();

  if (startlog){
    logger << t, rear_vel, front_vel
              , steer_pos, steer_vel
              , roll_ang, roll_vel, yaw_vel
              , eso_states[0], eso_states[1], eso_states[2], eso_states[3]
              , target_steer_pos, target_forw_vel, target_steer_vel
              , residual_action.steer, residual_action.rear, double(residual_fresh)
              , cmd.steer_vel, cmd.rear_vel;
    logger << endl;
  }

  return cmd;
}




int main(int argc, char**argv) {


  ros::init(argc, argv, "balance_control_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  ROS_INFO("balance_control_node!");

  ros::TransportHints hints;
  hints.tcpNoDelay(true);

  ros::Publisher cmd_pub = nh.advertise<mujoco_ros::ControlCmd>("control_cmd", 1, false);
  ros::Subscriber state_sub = nh.subscribe<mujoco_ros::RobotState>("robot_state", 1, stateCallback, ros::VoidConstPtr(),hints);
  ros::Subscriber boardcmd_sub = nh.subscribe<mujoco_ros::BoardCmd>("board_cmd", 1, boardcmdCallback, ros::VoidConstPtr(), hints);
  auto residual_sub=nh.subscribe<mujoco_ros::ResidualCmd>("residual_cmd",1,residualCallback,ros::VoidConstPtr(),hints);

  std::string config_path;
  private_nh.param<std::string>("config_path",config_path,ros::package::getPath("mujoco_ros")+"/config/controller.yaml");
  YAML::Node params_node = YAML::LoadFile(config_path);

  control_freq = params_node["control_freq"].as<int>();
  if (control_freq<=0) throw std::invalid_argument("control_freq must be positive");
  private_nh.param<std::string>("log_dir",log_dir,"/tmp/sttw_control");
  std::filesystem::create_directories(log_dir);
  log_dir+="/";
  private_nh.param("residual_enabled",residual_enabled,false);
  state_timeout=params_node["state_timeout"].as<double>(.05);
  residual_timeout=params_node["residual_timeout"].as<double>(.05);
  if (!(state_timeout>0) || !std::isfinite(state_timeout) || !(residual_timeout>0) || !std::isfinite(residual_timeout))
    throw std::invalid_argument("timeouts must be finite and positive");
  eso_start_seconds=params_node["eso_start_seconds"].as<double>(3.);
  command_limits.steer_position=params_node["steer_position_limit"].as<double>(.8);
  command_limits.steer_rate=params_node["steer_rate_limit"].as<double>(3.);
  command_limits.rear_rate=params_node["rear_rate_limit"].as<double>(60.);
  command_limits.steer_scale=params_node["steer_residual_scale"].as<double>(1.);
  command_limits.rear_scale=params_node["rear_residual_scale"].as<double>(5.);
  command_limits.strength=params_node["residual_strength"].as<double>(1.);
  sttw::composeCommand({}, {},0,1.0/control_freq,command_limits,false);
  log_file_name = params_node["control_node_log_file_name"].as<std::string>();
  startlog = params_node["control_start_log"].as<bool>();

  // ecbc 参数
  ebcparams.dist_lpf_freq = params_node["ebc_dist_lpf_freq"].as<float>();
  ebcparams.min_rear_vel_mps = params_node["ebc_min_rear_vel_mps"].as<float>();
  ebcparams.sample_time = 1.0 / control_freq;
  // ebcparams.sample_time = 0.001;
  ebcparams.wc = params_node["ebc_wc"].as<float>();
  ebcparams.wo = params_node["ebc_wo"].as<float>();

  rearLPFparams.low_cutoff_freq = params_node["rearLPF_low_cutoff_freq"].as<float>();
  rearLPFparams.sample_time = 1.0 / control_freq;
  // rearLPFparams.sample_time = 0.001;

  eso_based_controller.setParams(ebcparams, 1.0);
  rear_vel_filter.setParams(rearLPFparams, 0); 

  // 指令参数
  target_steer_pos = params_node["target_steer_pos_deg"].as<float>();
  target_steer_pos = target_steer_pos / 180.0*M_PI;
  target_forw_vel = params_node["target_forw_vel_kmh"].as<float>();
  target_forw_vel = target_forw_vel/3.6/0.1;


  if (startlog){      
    // 添加 CSV 日志头
    {
      string head_str[] = {
        "time", "rear_vel", "front_vel",
        "steer_pos", "steer_vel",
        "roll_ang", "roll_vel", "yaw_vel",
        "roll_acc_disturbance_estimate", "eso_roll", "eso_roll_rate", "eso_disturbance",
        "target_steer_pos", "target_rear_rate", "base_steer_rate",
        "residual_steer_normalized", "residual_rear_normalized", "residual_applied",
        "command_steer_rate", "command_rear_rate"
      };
      int head_length = sizeof(head_str) / sizeof(head_str[0]);
      logger.init(log_dir+log_file_name, true, head_str, head_length);
    }
  }


  
  ros::Rate rate(control_freq);
  while (ros::ok()) {

    if (!first_state) {
      ros::spinOnce();
      rate.sleep();
      continue;
    }

    ros::spinOnce();
    if (mode==9 || (ros::WallTime::now()-state_received).toSec()>state_timeout) {
      cmd_pub.publish(mujoco_ros::ControlCmd{});
      received_residual=false;
      controller_needs_reset=true;
      rate.sleep();
      continue;
    }
    if (controller_needs_reset) {
      eso_based_controller.resetStates();
      control_start_time=latest_state.time;
      received_residual=false;
      controller_needs_reset=false;
    }
    mujoco_ros::ControlCmd cmd = computeControl();
    cmd_pub.publish(cmd);

    
    rate.sleep();
  } // while(ros::ok())


  logger.close();
  return 0;
}
