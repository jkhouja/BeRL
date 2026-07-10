#!/bin/bash
# Dry-run tests for the experiments/lib/common.sh launcher: verify the anti-collapse /
# policy-shaping sweep knobs (entropy_coeff, think_only_pg, format_penalty) propagate into
# the emitted hydra command, and that unset defaults mirror ppo_trainer.yaml (back-compat).
#
# Pure dry-run (BERL_DRY_RUN=1): no GPU, no torch import, no training. Fast + hermetic.
# Run:  bash tests/launcher/test_launcher_knobs.sh
set -u
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
export REPO_DIR
fails=0

# Capture the dry-run command string for a given env override set.
emit() {
  BERL_DRY_RUN=1 BERL_NO_EXP_LOG=1 \
    EXP_ID="${EXP_ID:-test-knobs}" DATA_NAME=dcfg_smoke_mix DATA_TRAIN=/tmp/nonexistent.parquet \
    "$@" bash "$REPO_DIR/experiments/smoke_qwen2.5.sh" 2>&1
}

assert_has() {  # assert_has "<label>" "<needle>" "<haystack>"
  if printf '%s' "$3" | grep -qF -- "$2"; then
    echo "PASS: $1"
  else
    echo "FAIL: $1 — expected to find: $2"
    fails=$((fails + 1))
  fi
}

echo "== Test 1: defaults mirror ppo_trainer.yaml (unset => yaml defaults) =="
out="$(emit)"
assert_has "entropy_coeff default 0.001"  "actor_rollout_ref.actor.entropy_coeff=0.001" "$out"
assert_has "think_only_pg default False"  "actor_rollout_ref.actor.think_only_pg=False" "$out"
assert_has "actor.format_penalty def 0.0" "actor_rollout_ref.actor.format_penalty=0.0"  "$out"
assert_has "reward_model.format_penalty def 0.0" "reward_model.format_penalty=0.0"       "$out"

echo "== Test 2: swept values propagate (fp=5, ec=0.0, think_only_pg=True) =="
out="$(THINK_ONLY_PG=True ENTROPY_COEFF=0.0 FORMAT_PENALTY=5 emit)"
assert_has "entropy_coeff=0.0"       "actor_rollout_ref.actor.entropy_coeff=0.0"   "$out"
assert_has "think_only_pg=True"      "actor_rollout_ref.actor.think_only_pg=True"  "$out"
assert_has "actor.format_penalty=5"  "actor_rollout_ref.actor.format_penalty=5"    "$out"
assert_has "reward_model.format_penalty=5" "reward_model.format_penalty=5"          "$out"

echo "== Test 3: reward family + RM mode dimensions (log_prob, frozen RM) =="
out="$(REWARD_TYPE=log_prob USE_ACTOR_AS_RM=False emit)"
assert_has "reward_type=log_prob (reward_model)"     "+reward_model.reward_type=log_prob"     "$out"
assert_has "reward_type=log_prob (actor_rollout_ref)" "+actor_rollout_ref.reward_type=log_prob" "$out"

echo "== Test 4: power reward family with ll_min=-6 =="
out="$(REWARD_TYPE=power POWER_K=3 POWER_LL_MIN=-6.0 emit)"
assert_has "power_k=3"        "power_k=3"        "$out"
assert_has "power_ll_min=-6.0" "power_ll_min=-6.0" "$out"

echo
if [ "$fails" -eq 0 ]; then
  echo "ALL LAUNCHER KNOB TESTS PASSED"
  exit 0
else
  echo "$fails LAUNCHER KNOB TEST(S) FAILED"
  exit 1
fi
