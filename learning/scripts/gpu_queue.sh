#!/usr/bin/env bash
# One shared GPU slot across STTW worktrees; wait without polling.
set -e
printf 'Waiting for the shared GPU slot: %s\n' "$*"
exec flock --verbose --no-fork /home/qy/STTW_CONTROL/research-hub/.gpu0.lock "$@"
