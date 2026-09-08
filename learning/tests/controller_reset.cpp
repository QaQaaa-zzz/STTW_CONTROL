#include <cassert>
#include <cmath>
#include "ECBC/eso_based_controller.h"
int main() {
  EsoBasedController used, fresh;
  EbcParams params{.005,40,5,8,.5};
  used.setParams(params,1.);
  fresh.setParams(params,1.);
  for(int i=0;i<30;++i) {
    used.updateSystemParams(2.);
    used.getOutput(.1,.1,.2,.3,.1,true,4.);
    used.update();
  }
  used.resetStates();
  const double a=used.getOutput(0,0,0,0,0,true,4.);
  const double b=fresh.getOutput(0,0,0,0,0,true,4.);
  double x[4],y[4]; used.getEsoStates(x); fresh.getEsoStates(y);
  assert(std::abs(a-b)<1e-12);
  for(int i=0;i<4;++i) assert(std::abs(x[i]-y[i])<1e-12);
}
