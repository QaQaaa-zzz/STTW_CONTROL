#include <iostream>
#include <iomanip>
#include "ECBC/eso_based_controller.h"

int main() {
  EsoBasedController controller;
  controller.setParams(EbcParams{0.005,40,5,8,0.5},1.0);
  double speed, steer, steer_rate, roll, roll_rate, reference;
  double enabled;
  std::cout << std::setprecision(17);
  while (std::cin >> speed >> steer >> steer_rate >> roll >> roll_rate >> reference >> enabled) {
    controller.updateSystemParams(speed);
    double u = controller.getOutput(steer,steer_rate,roll,roll_rate,reference,enabled,4.0);
    std::cout << u << " " << controller.getEstimatedDisturbance() << " "
              << controller.getEstimatedEquilibrium() << "\n";
    controller.update();
  }
}
