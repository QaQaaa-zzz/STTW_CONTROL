# Scalar effort-model diagnostic

{
  "static": {
    "effort_threshold": 0.014709383246684669,
    "calibration": {
      "n": 132,
      "correct": 109,
      "false_safe": 0,
      "conservative": 23,
      "success_recall": 0.8188976377952756,
      "failure_recall": 1.0
    },
    "speed_transfer": {
      "n": 132,
      "correct": 40,
      "false_safe": 92,
      "conservative": 0,
      "success_recall": 1.0,
      "failure_recall": 0.041666666666666664
    }
  },
  "dynamic": {
    "effort_threshold": 0.030839085298575167,
    "calibration": {
      "n": 132,
      "correct": 123,
      "false_safe": 0,
      "conservative": 9,
      "success_recall": 0.9291338582677166,
      "failure_recall": 1.0
    },
    "speed_transfer": {
      "n": 132,
      "correct": 44,
      "false_safe": 84,
      "conservative": 4,
      "success_recall": 0.8888888888888888,
      "failure_recall": 0.125
    }
  }
}

Static e=0 versus dynamic z_u, both with first-tick small-probe local force mapping. E=(z+KF)^2/W with R=1 and fixed-speed scalar unstable model. Threshold selected by balanced accuracy on 2.3m/s only, ties fewer false-safe then smaller threshold. 2.8m/s is DEV speed transfer; not final TEST. Unknown gamma is estimated, not asserted from rigid-body geometry. This mapping remains dependent on small physical probes and is not yet a cheap online input. No actuator peak feasibility enforced; effort is not joules. Omissions are incomplete first-probe states, not negative labels. Equilibrium recovery and full task recovery differ; failure here does not disprove the stated linear algebra.
