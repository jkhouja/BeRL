#!/bin/bash
# Phase −1 "phase-stability" one-shot HP+reward sweep driver.
#
# Enumerates the SAME 96 Wave-1 cells encoded in the tracker rows PS001–PS096
# (project_planning/BeRL_experiments_tracker.md), in the SAME order, so the
# generated EXP_ID matches the tracker's "Exp ID" column exactly:
#
#   reward family (outermost): log_prob, power(k=3,ll_min=-6), power(k=5,ll_min=-6)
#   base 32 = RM{frozen,actor} × KL{0.01,0.05} × LR{5e-7,1e-6} × FP{0,5} × EC{0.0,0.001}
#
# Fixed per cell: think_only_pg=True, ROLLOUT_N=16, ctx 2048/512, COT=cot_eval
# (the tracker's "COT_FREEFORM" ⇒ the concrete cot_eval prompt, i.e. the 17f recipe).
#
# Runs SEQUENTIALLY on this node (each cell is GPU-bound). Use IDX_START/IDX_END to
# split across nodes or resume after a crash. Nothing launches under BERL_DRY_RUN=1.
#
# Usage:
#   bash experiments/phase_stability_sweep.sh                 # all 96 Qwen2.5 cells
#   IDX_START=1 IDX_END=32 bash experiments/phase_stability_sweep.sh   # log_prob block only
#   BERL_DRY_RUN=1 bash experiments/phase_stability_sweep.sh | less    # preview EXP_IDs/knobs
#   FAMILY=gemma  ONLY_IDX="3 5 9" bash experiments/phase_stability_sweep.sh  # Wave-2 subset
set -u
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"

FAMILY="${FAMILY:-qwen2.5}"          # qwen2.5 (Wave 1) | gemma | qwen3 (Wave 2)
IDX_START="${IDX_START:-1}"
IDX_END="${IDX_END:-96}"
ONLY_IDX="${ONLY_IDX:-}"             # optional space-separated whitelist of 1-based indices

case "$FAMILY" in
  qwen2.5) LAUNCHER="smoke_qwen2.5.sh"; MODEL_TAG_FAM="Qwen2.5"; DATA_NAME_DEF="dcfg_smoke_mix";       DATA_TRAIN_DEF="$REPO_DIR/data/dcfg_smoke_mix.parquet" ;;
  gemma)   LAUNCHER="smoke_gemma.sh";   MODEL_TAG_FAM="Gemma-2"; DATA_NAME_DEF="dcfg_smoke_mix_gemma"; DATA_TRAIN_DEF="$REPO_DIR/data/dcfg_smoke_mix_gemma.parquet" ;;
  qwen3)   LAUNCHER="smoke_qwen3.sh";   MODEL_TAG_FAM="Qwen3";   DATA_NAME_DEF="dcfg_smoke_mix_gemma"; DATA_TRAIN_DEF="$REPO_DIR/data/dcfg_smoke_mix_gemma.parquet" ;;
  *) echo "[sweep] unknown FAMILY=$FAMILY (want qwen2.5|gemma|qwen3)" >&2; exit 2 ;;
esac
DATA_NAME="${DATA_NAME:-$DATA_NAME_DEF}"
DATA_TRAIN="${DATA_TRAIN:-$DATA_TRAIN_DEF}"

# Fixed knobs (shared by every cell).
export TASK=behavior
export MODEL_FAMILY="$FAMILY"
# Dry-run previews must not litter experiments_logs/ (write_summary_md runs before the
# dry-run guard in common.sh), so suppress per-run summary md when previewing.
[ "${BERL_DRY_RUN:-0}" = "1" ] && export BERL_NO_EXP_LOG="${BERL_NO_EXP_LOG:-1}"
export COT_VAR="${COT_VAR:-cot_eval}"      # COT_FREEFORM ⇒ cot_eval
export THINK_ONLY_PG=True
export ROLLOUT_N="${ROLLOUT_N:-16}"
export MAX_PROMPT="${MAX_PROMPT:-2048}"
export MAX_RESP="${MAX_RESP:-512}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-999}"

# Swept dimensions — order matters (must reproduce the tracker enumeration).
REWARDS=( "log_prob|-|-|lp" "power|3|-6.0|pk3" "power|5|-6.0|pk5" )
RMS=( "frozen|f|False" "actor|a|True" )
KLS=( 0.01 0.05 )
LRS=( 5e-7 1e-6 )
FPS=( 0 5 )
ECS=( 0.0 0.001 )

idx=0
ran=0
for rew in "${REWARDS[@]}"; do
  IFS='|' read -r rtype pk llmin rcode <<<"$rew"
  for rm in "${RMS[@]}"; do
    IFS='|' read -r rm_name rm_short use_actor <<<"$rm"
    for kl in "${KLS[@]}"; do
      for lr in "${LRS[@]}"; do
        for fp in "${FPS[@]}"; do
          for ec in "${ECS[@]}"; do
            idx=$((idx + 1))
            # index/whitelist filtering
            [ "$idx" -lt "$IDX_START" ] && continue
            [ "$idx" -gt "$IDX_END" ] && continue
            if [ -n "$ONLY_IDX" ] && ! printf ' %s ' "$ONLY_IDX" | grep -q " $idx "; then continue; fi

            ps=$(printf "PS%03d" "$idx")
            exp_id="Phase-stability-Pm1swp_rm${rm_short}_kl${kl}_lr${lr}_fp${fp}_ec${ec}_${rcode}"

            echo "==================================================================="
            echo "[sweep] $ps ($idx/96) FAMILY=$FAMILY  EXP_ID=$exp_id"
            echo "[sweep]   reward=$rtype${pk:+ k=$pk}${llmin:+ ll_min=$llmin} rm=$rm_name kl=$kl lr=$lr fp=$fp ec=$ec"

            POWER_K_ENV="${pk/-/}"; [ "$pk" = "-" ] && POWER_K_ENV="2.0"
            LLMIN_ENV="$llmin";     [ "$llmin" = "-" ] && LLMIN_ENV="-8.0"

            EXP_ID="$exp_id" \
            DATA_NAME="$DATA_NAME" DATA_TRAIN="$DATA_TRAIN" \
            REWARD_TYPE="$rtype" POWER_K="$POWER_K_ENV" POWER_LL_MIN="$LLMIN_ENV" \
            USE_ACTOR_AS_RM="$use_actor" \
            KL="$kl" LR="$lr" FORMAT_PENALTY="$fp" ENTROPY_COEFF="$ec" \
              bash "$REPO_DIR/experiments/$LAUNCHER"
            rc=$?
            if [ "$rc" -ne 0 ]; then
              echo "[sweep] $ps FAILED (rc=$rc) — continuing to next cell" >&2
            fi
            ran=$((ran + 1))
          done
        done
      done
    done
  done
done

echo "[sweep] done. launched $ran cell(s) (idx $IDX_START..$IDX_END, family=$FAMILY)."
