#include <ros/ros.h>
#include <iostream>
#include <cstring>
#include <sstream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include "mujoco_ros/BoardCmd.h"

int main(int argc, char *argv[])
{
    //执行 ros 节点初始化
    ros::init(argc,argv,"balance_boardcmd_node");
    //创建 ros 节点句柄(非必须)
    ros::NodeHandle nh;
    //控制台输出
    ROS_INFO("Balance Board Command Node!");

    ros::Publisher pub_cmd = nh.advertise<mujoco_ros::BoardCmd>("board_cmd", 1);
    // mode = 0 启动
    // mode = 1 速度模式
    // mode = 9 关闭
    mujoco_ros::BoardCmd boardcmd_msg;


    std::string boardinput;
    int mode;
    double target_steer_pos;
    double target_rear_vel;

    ros::Rate r(10);
    while (ros::ok())
    {

        std::cout << std::endl;
        std::cout << "Split by space: mode, steer_pos(°), rear_vel(m/s)" << std::endl;
        std::cout << std::endl;
        
        std::getline(std::cin, boardinput);
        std::istringstream iss(boardinput);

        std::vector<double> inputs;
        double temp;
        while(iss >> temp){
            inputs.push_back(temp);
        }

        iss.clear();

        if (inputs.size()!=3 || !std::isfinite(inputs[0]) || !std::isfinite(inputs[1]) || !std::isfinite(inputs[2]) ||
            (inputs[0]!=0 && inputs[0]!=1 && inputs[0]!=9)) {
            ROS_WARN("Expected mode (0, 1, 9), finite steer degrees, finite forward m/s");
            continue;
        }

        boardcmd_msg.mode = inputs[0];
        boardcmd_msg.target_steer_pos = inputs[1] / 180.0 * 3.1415926;
        boardcmd_msg.target_rear_vel = inputs[2] / 0.1;

        // mode = inputs[0];
        // target_steer_pos = inputs[1];
        // target_rear_vel = inputs[2];
        // std::cout << "mode: " << mode << "  steer pos: " << target_steer_pos << "  rear vel: " << target_rear_vel << std::endl;

        pub_cmd.publish(boardcmd_msg);

        
        ros::spinOnce();
        r.sleep();  

    }


    return 0;

}
