#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sttw {
struct Command { double steer=0, rear=0; };
struct Limits {
  double steer_position=.8, steer_rate=3., rear_rate=60.;
  double steer_scale=1., rear_scale=5., strength=1.;
};
inline Command composeCommand(Command base, Command normalized_residual, double steer,
                              double dt, Limits limits, bool residual_fresh) {
  if (!(dt>0) || !std::isfinite(dt) || !(limits.steer_position>0) || !(limits.steer_rate>0)
      || !(limits.rear_rate>0) || !std::isfinite(limits.steer_position)
      || !std::isfinite(limits.steer_rate) || !std::isfinite(limits.rear_rate)
      || !std::isfinite(limits.steer_scale) || limits.steer_scale<0
      || !std::isfinite(limits.rear_scale) || limits.rear_scale<0
      || !std::isfinite(limits.strength) || limits.strength<0 || limits.strength>1)
    throw std::invalid_argument("invalid command limits or timestep");
  if (!std::isfinite(base.steer) || !std::isfinite(base.rear) || !std::isfinite(steer)) return {};
  if (residual_fresh && std::isfinite(normalized_residual.steer) && std::isfinite(normalized_residual.rear)) {
    base.steer+=limits.strength*limits.steer_scale*std::clamp(normalized_residual.steer,-1.,1.);
    base.rear+=limits.strength*limits.rear_scale*std::clamp(normalized_residual.rear,-1.,1.);
  }
  double lower=std::clamp((-limits.steer_position-steer)/dt,-limits.steer_rate,limits.steer_rate);
  double upper=std::clamp((limits.steer_position-steer)/dt,-limits.steer_rate,limits.steer_rate);
  return {std::clamp(base.steer,lower,upper),std::clamp(base.rear,-limits.rear_rate,limits.rear_rate)};
}
}  // namespace sttw
