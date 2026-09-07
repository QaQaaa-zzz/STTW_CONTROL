#include <ros/ros.h>
#include <iostream>
#include <memory>
#include <fstream>
#include "mujoco_ros/RobotState.h"
#include "mujoco_ros/ControlCmd.h"
#include "mujoco_ros/BoardCmd.h"
#include "ECBC/eso_based_controller.h"
#include <yaml-cpp/yaml.h>
#include "data_logger.h"

// 全局变量（缓存最新机器人状态）
mujoco_ros::RobotState latest_state;
bool first_state = false;
double control_start_time = 0;

int control_freq;

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
  latest_state = *msg;
  
  if (!first_state){
    control_start_time = latest_state.time;
    std::cout << "控制开始时刻：" << control_start_time << " 秒" << std::endl;
    first_state = true;
  } 
  
}

// 缓存最新指令
void boardcmdCallback(const mujoco_ros::BoardCmd::ConstPtr& msg){
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
  if (first_state && t > 3){
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



  cmd.rear_vel = target_forw_vel;
  cmd.steer_vel = target_steer_vel;

  eso_based_controller.update();

  if (startlog){
    logger << t, rear_vel, front_vel
              , steer_pos, steer_vel
              , roll_ang, roll_vel, yaw_vel
              , eso_states[0], eso_states[1], eso_states[2], eso_states[3];
    logger << endl;
  }

  return cmd;
}




int main(int argc, char**argv) {


  ros::init(argc, argv, "balance_control_node");
  ros::NodeHandle nh;
  ROS_INFO("balance_control_node!");

  ros::TransportHints hints;
  hints.tcpNoDelay(true);

  ros::Publisher cmd_pub = nh.advertise<mujoco_ros::ControlCmd>("control_cmd", 1, false);
  ros::Subscriber state_sub = nh.subscribe<mujoco_ros::RobotState>("robot_state", 1, stateCallback, ros::VoidConstPtr(),hints);
  ros::Subscriber boardcmd_sub = nh.subscribe<mujoco_ros::BoardCmd>("board_cmd", 1, boardcmdCallback, ros::VoidConstPtr(), hints);

  YAML::Node params_node = YAML::LoadFile("/home/ubuntu/mujoco_ros_ws/src/mujoco_ros/src/mujoco_ros_params.yaml");

  control_freq = params_node["control_freq"].as<int>();
  log_dir = "/home/ubuntu/mujoco_ros_ws/src/mujoco_ros/log/";
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
        "eso_states0", "eso_states1", "eso_states2", "eso_states3"
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
    mujoco_ros::ControlCmd cmd = computeControl();
    cmd_pub.publish(cmd);

    
    rate.sleep();
  } // while(ros::ok())


  logger.close();
  return 0;
}
