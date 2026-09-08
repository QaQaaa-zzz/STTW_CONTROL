#include <cassert>
#include <cmath>
#include <limits>
#include "ECBC/residual_command.h"
int main() {
  using namespace sttw;
  auto u=composeCommand({1.25,12},{0,0},0,.005,{},true);
  assert(u.steer==1.25 && u.rear==12);
  u=composeCommand({2,10},{1,1},0,.005,{},true);
  assert(u.steer==3 && u.rear==15);
  u=composeCommand({2,10},{1,1},.8,.005,{},true);
  assert(u.steer==0);
  u=composeCommand({2,10},{1,1},0,.005,{},false);
  assert(u.steer==2 && u.rear==10);
  u=composeCommand({2,10},{std::numeric_limits<double>::quiet_NaN(),1},0,.005,{},true);
  assert(u.steer==2 && u.rear==10);
  bool rejected=false;
  try { composeCommand({0,0},{0,0},0,0,{},true); }
  catch (const std::invalid_argument&) { rejected=true; }
  assert(rejected);
}
