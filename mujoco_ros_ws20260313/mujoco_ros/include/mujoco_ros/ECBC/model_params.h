#ifndef JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_MODEL_PARAMS_H
#define JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_MODEL_PARAMS_H
#include <cmath>
#include <iostream>

struct BikeModelParams {
  double g;      // 重力加速度
  double m;      // 整车质量
  double m_b;    // 前车体质量（除去后轮）
  double m_r;    // 后轮质量
  double m_f;    // 前轮质量
  double Ib;     // 质心绕地面轴惯量
  double Isfz;   // 叉和前轮，沿转向轴
  double Ify;    // 前轮惯量，轴
  double Iry;    // 后轮惯量，轴
  double It[3];  // 整车惯量，车体组合体坐标系
  double R;      // 车轮半径
  double lambda; // 前叉角
  double slam;   // sin(lambda);
  double clam;   // cos(lambda);
  double a;      // 质心距后轮距离
  double b;      // 轴距
  double c;      // 拖曳距
  double h;      // 质心高度
  double mu;     // 摩擦系数
  double r_br_x; // 前车体质心相对后轮质心 x 坐标
  double r_br_y; // y

  BikeModelParams(){
    lambda = 25*M_PI/180;
    m = 5.4;      // 整车实际质量,7.4kg，模型5.434
    Isfz = 0.002957;
    Ify = 0.002032;
    Iry = 0.002174;
    It[0] = 0.043929 * m / 5.434;
    It[1] = 0.120149;
    It[2] = 0.163714;
    R = 0.1;
    slam = sin(lambda);
    clam = cos(lambda);
    a = 0.164;      // 质心距后轮距离
    b = 0.408;      // 轴距
    c = 0.024;      // 拖曳距
    h = 0.2;      // 质心高度,0.194
    g = 9.8;      // 重力加速度
    Ib = m*pow(h,2) + It[0];     // 质心绕地面轴惯量
    mu = 0.8;
    // std::cout << "mass: " << m << std::endl;
  }
};
#endif //JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_MODEL_PARAMS_H
