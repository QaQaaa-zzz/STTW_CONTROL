#include <iostream>
#include <time.h>
#include <math.h>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <unistd.h>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <chrono>
#include <ros/ros.h>
#include <ros/package.h>
#include "mujoco_ros/RobotState.h"
#include "mujoco_ros/ControlCmd.h"
#include "data_logger.h"
#include <yaml-cpp/yaml.h>

#include <mujoco/mujoco.h>


#include <eigen3/Eigen/Dense>
#define pi 3.14159265358979323846

//函数声明
void init_simulation(mjModel* m, mjData* d);
void create_model(mjModel** m, mjData** d, const char* model_file);
Eigen::Vector3d quatertion2euler(const double* quatertion);


// --- 路径配置 ---
std::string pkg_path;
std::string ws_path;
std::string mujoco_plugin_dir;
std::string model_path;
std::string log_dir;
std::string log_file_name;
bool startlog = false;
DataLogger logger;


// --- 全局变量用于存储传感器 ID ---
int sensor_id_act_steering_p = -1;
int sensor_id_act_rearwheel_p = -1;
int sensor_id_act_steering_v = -1;
int sensor_id_act_rearwheel_v = -1;
int sensor_id_act_frontwheel_v = -1;
int sensor_id_gyro_local = -1;
int sensor_id_imu_global = -1; //4D
int sensor_id_gyro_global = -1;
int sensor_id_steer_force = -1;

// --- MuJoCo 模型和数据 ---
mjModel* m = nullptr;
mjData* d = nullptr;
mujoco_ros::ControlCmd latest_cmd;  // 最新控制指令
bool first_cmd = false;  // 是否收到过控制指令

void cmdCallback(const mujoco_ros::ControlCmd::ConstPtr& msg) {
  latest_cmd = *msg;
  first_cmd = true;
}

void ros_controller(const mjModel* m, mjData* d){
  mjModel* m_modifiable = const_cast<mjModel*>(m);
  if (first_cmd){
    m_modifiable->opt.disableactuator = 2;
    d->ctrl[1] = -latest_cmd.rear_vel;
    d->ctrl[2] = latest_cmd.steer_vel;
    d->ctrl[3] = latest_cmd.steer_vel;
  }
  else{
    auto cur_t = d->time;  
    const double* imu_global_ptr  = d->sensordata + m->sensor_adr[sensor_id_imu_global];
    Eigen::Vector3d euler_angles = quatertion2euler(imu_global_ptr);
    double roll_ang = euler_angles[0] - pi/2;
    double v0 = 6/3.6/0.1;

    m_modifiable->opt.disableactuator = 4;
    d->ctrl[0] = 0 - 2.5*(roll_ang - 0);
    d->ctrl[1] = -std::min(5*cur_t*v0, v0);
  }
}

int main(int argc, char** argv) {

    // 初始化漂移参数
    ros::init(argc, argv, "mujoco_ros_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    ROS_INFO("mujoco_ros_node!");

    std::string config_path;
    pkg_path=ros::package::getPath("mujoco_ros");
    private_nh.param<std::string>("config_path",config_path,pkg_path+"/config/controller.yaml");
    YAML::Node params_node = YAML::LoadFile(config_path);
    startlog = params_node["mujoco_start_log"].as<bool>();
    log_file_name = params_node["mujoco_node_log_file_name"].as<std::string>();

    private_nh.param<std::string>("log_dir",log_dir,"/tmp/sttw_control");
    std::filesystem::create_directories(log_dir);
    log_dir+="/";
    private_nh.param<std::string>("plugin_dir",mujoco_plugin_dir,pkg_path+"/../model/mujoco_plugin");
    private_nh.param<std::string>("model_path",model_path,pkg_path+"/../model/scalebike_scene_matlab.xml");

    // 创建 MuJoCo 模型和数据结构
    bool load_plugins=true;
    private_nh.param("load_plugins",load_plugins,true);
    if (load_plugins) mj_loadAllPluginLibraries(mujoco_plugin_dir.c_str(),nullptr);
    create_model(&m, &d, model_path.c_str());
    init_simulation(m, d);  //设置传感器ID、控制回调函数及渲染器

    ros::TransportHints hints;
    hints.tcpNoDelay(true);

    ros::Publisher state_pub = nh.advertise<mujoco_ros::RobotState>("robot_state", 1, false);
    ros::Subscriber cmd_sub = nh.subscribe<mujoco_ros::ControlCmd>("control_cmd", 1, cmdCallback, ros::VoidConstPtr(), hints);

    if (startlog){      
      // 添加 CSV 日志头
      std::string head_str[] = {
      "time", "rear_vel", "front_vel",
      "steer_pos", "steer_vel",
      "roll_ang", "roll_vel", "yaw_vel"
      };
      int head_length = sizeof(head_str) / sizeof(head_str[0]);
      logger.init(log_dir+log_file_name, true, head_str, head_length);
    }
    

    //运行仿真
    const int sim_freq = 1000;  // ros循环频率
    const int publish_divisor = 5;  // 分频系数(1000/200=5), 数据发布频率
    int pub_step_counter = 0;  // 步计数器
    int timeout_count = 0;    //超时次数
    int single_cycle_sim_count = 5; // 单次ros循环中，mujoco前向动力学计算次数
    single_cycle_sim_count = params_node["single_cycle_sim_count"].as<int>();

    mj_resetDataKeyframe(m, d, -1);  // Reset keyframe data
    ros::Rate rate(sim_freq);
    
    // 用于记录数据
    double log_time, rear_vel, front_vel;
    double steer_pos, steer_vel;
    double roll_ang, roll_vel, yaw_vel;

    while(ros::ok()) {
      auto start = std::chrono::steady_clock::now();
      ros::spinOnce();

      // mujoco步长为2e-4s, 前向计算5次
      for(int i=0; i<single_cycle_sim_count; i++){
        mj_step(m, d);
      }
      
      
      // 每5步（5ms）发布一次机器人状态（200Hz）
      pub_step_counter++;
      if (pub_step_counter >= publish_divisor) {
        
        const double* imu_global_ptr  = d->sensordata + m->sensor_adr[sensor_id_imu_global];
        Eigen::Vector3d euler_angles = quatertion2euler(imu_global_ptr);
        
        const double* gyro_local_ptr  = d->sensordata + m->sensor_adr[sensor_id_gyro_local];
        Eigen::Vector3d gyro_local(gyro_local_ptr[0], gyro_local_ptr[1], gyro_local_ptr[2]);

        mujoco_ros::RobotState state_msg;
        state_msg.time = d->time;
        state_msg.steer_pos = d->sensordata[m->sensor_adr[sensor_id_act_steering_p]];
        state_msg.roll_ang = euler_angles[0] - pi/2;
        state_msg.steer_vel = d->sensordata[m->sensor_adr[sensor_id_act_steering_v]];
        state_msg.roll_vel = gyro_local[0];
        state_msg.yaw_vel = gyro_local[2];
        state_msg.rear_vel = - d->sensordata[m->sensor_adr[sensor_id_act_rearwheel_v]];
        state_msg.front_vel = - d->sensordata[m->sensor_adr[sensor_id_act_frontwheel_v]];

        state_pub.publish(state_msg);  // 发布状态

        pub_step_counter = 0;  // 重置计数器
      }
      

      if (startlog){

        log_time = d->time;  // 添加当前时间
        rear_vel = d->sensordata[m->sensor_adr[sensor_id_act_rearwheel_v]];//rear_vel
        front_vel = d->sensordata[m->sensor_adr[sensor_id_act_frontwheel_v]];//front_vel
        steer_pos = d->sensordata[m->sensor_adr[sensor_id_act_steering_p]];//steer_pos
        steer_vel = d->sensordata[m->sensor_adr[sensor_id_act_steering_v]];//steer_vel
        const double* imu_global_ptr  = d->sensordata + m->sensor_adr[sensor_id_imu_global];
        Eigen::Vector3d euler_angles = quatertion2euler(imu_global_ptr);
        roll_ang = euler_angles[0] - pi/2; // roll_ang
        const double* gyro_local_ptr  = d->sensordata + m->sensor_adr[sensor_id_gyro_local];
        Eigen::Vector3d gyro_local(gyro_local_ptr[0], gyro_local_ptr[1], gyro_local_ptr[2]);
        roll_vel = gyro_local[0]; // roll_vel
        yaw_vel = gyro_local[2]; // yaw_vel

        logger << log_time, rear_vel, front_vel
                  , steer_pos, steer_vel
                  , roll_ang, roll_vel, yaw_vel;
        logger << std::endl;

      }

      

      auto end = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      if (duration.count() > 1){
        timeout_count++;
        std::cout << "当前仿真时刻：" << d->time << "  单步执行时间：" << duration.count() << " 毫秒，累计超时次数：" << timeout_count << std::endl;
      }
      rate.sleep();
      
    } // while(ros::ok())

    logger.close();
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
} // main

// --- 函数：创建模型和数据结构 ---
void create_model(mjModel** m, mjData** d, const char* model_file) {
    char error[1000];
    *m = mj_loadXML(model_file, nullptr, error, 1000);
    if (!*m) {
        // mju_error(error);
        mju_error("%s", error);
        return;
    }
    *d = mj_makeData(*m);
    if (!*d) {
        mju_error("Could not create data structure");
        mj_deleteModel(*m);
        return;
    }
}


// --- 函数：设置传感器 ID ---
void setupSensorIDs(mjModel* m) {
  // 使用 mj_name2id 函数查找传感器的整数 ID
  // mjOBJ_SENSOR 是对象类型，字符串是 XML 中定义的传感器名称
  sensor_id_act_steering_p = mj_name2id(m, mjOBJ_SENSOR, "act_steering_p");
  sensor_id_act_rearwheel_p = mj_name2id(m, mjOBJ_SENSOR, "act_rearwheel_p");
  sensor_id_act_steering_v = mj_name2id(m, mjOBJ_SENSOR, "act_steering_v");
  sensor_id_act_rearwheel_v = mj_name2id(m, mjOBJ_SENSOR, "act_rearwheel_v");
  sensor_id_act_frontwheel_v = mj_name2id(m, mjOBJ_SENSOR, "act_frontwheel_v");
  sensor_id_gyro_local = mj_name2id(m, mjOBJ_SENSOR, "gyro_local");
  sensor_id_imu_global = mj_name2id(m, mjOBJ_SENSOR, "imu_global");
  sensor_id_gyro_global = mj_name2id(m, mjOBJ_SENSOR, "gyro_global");
  sensor_id_steer_force = mj_name2id(m, mjOBJ_SENSOR, "steer_force");

  // 添加错误检查，确保所有传感器都已找到
  if (sensor_id_act_steering_p < 0)    std::cerr << "错误: 未找到传感器 'act_steering_p'!" << std::endl;
  if (sensor_id_act_rearwheel_p < 0)   std::cerr << "错误: 未找到传感器 'act_rearwheel_p'!" << std::endl;
  if (sensor_id_act_steering_v < 0)    std::cerr << "错误: 未找到传感器 'act_steering_v'!" << std::endl;
  if (sensor_id_act_rearwheel_v < 0)   std::cerr << "错误: 未找到传感器 'act_rearwheel_v'!" << std::endl;
  if (sensor_id_act_frontwheel_v < 0)  std::cerr << "错误: 未找到传感器 'act_frontwheel_v'!" << std::endl;
  if (sensor_id_gyro_local < 0)        std::cerr << "错误: 未找到传感器 'gyro_local'!" << std::endl;
  if (sensor_id_imu_global < 0)        std::cerr << "错误: 未找到传感器 'imu_global'!" << std::endl;
  if (sensor_id_gyro_global < 0)       std::cerr << "错误: 未找到传感器 'gyro_global'!" << std::endl;
  if (sensor_id_steer_force < 0)       std::cerr << "错误: 未找到传感器 'steer_force'!" << std::endl;

}

// --- 函数：初始化仿真 ---
// 设置传感器 ID、控制回调函数和关键帧数据及渲染器，并清除上次仿真的frames
void init_simulation(mjModel* m, mjData* d) {

  // 查找设置传感器的ID
  setupSensorIDs(m);
  mjcb_control = ros_controller;
  d->time = 0.0;  // Reset simulation time

}


Eigen::Vector3d quatertion2euler(const double* quatertion){
  double q0 = quatertion[0];
  double q1 = quatertion[1];
  double q2 = quatertion[2];
  double q3 = quatertion[3];
  Eigen::Vector3d euler;
  euler[0] = atan2(2*(q0*q1+q2*q3),1-2*(q1*q1+q2*q2));
  euler[1] = asin(2*(q0*q2-q3*q1));
  euler[2] = atan2(2*(q0*q3+q1*q2),1-2*(q2*q2+q3*q3));
  return euler;
}
