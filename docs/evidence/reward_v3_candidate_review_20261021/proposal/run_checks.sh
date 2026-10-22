#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:?Usage: bash run_checks.sh /absolute/path/to/STTW_CONTROL/worktree [components.npz]}"
AUDIT="$REPO/docs/evidence/reward_preference_audit_20261021/results.json"
COMPONENTS="${2:-$REPO/runs/reward_preference_audit_20261021/reward_components.npz}"
OUT="$REPO/runs/reward_persistent_preference_v3_audit"
python "$PACKAGE/validate_local.py"
python "$PACKAGE/rescore_from_published.py" --audit "$AUDIT" --output "$OUT/aggregate"
if [[ ! -f "$COMPONENTS" ]]; then
  echo "Missing stepwise NPZ: $COMPONENTS" >&2
  echo "Aggregate scoring is not discounted validation. No training has been launched." >&2
  exit 2
fi
python "$PACKAGE/rescore_discounted_components.py" --audit "$AUDIT" --components "$COMPONENTS" --output "$OUT/discounted"
