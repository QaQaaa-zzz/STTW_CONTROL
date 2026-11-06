#!/usr/bin/env bash
# Coordinate heavy jobs across sessions; keep all within-job GPU parallelism.
set -e
printf 'Waiting for the shared GPU slot: %s\n' "$*"
exec flock --verbose --no-fork /home/qy/STTW_CONTROL/research-hub/.gpu-job.lock "$@"
