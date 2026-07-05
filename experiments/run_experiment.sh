#!/bin/bash
# experiments/run_experiment.sh <EXP_ID> [-- extra main_ppo overrides]
#
# Dispatcher: look up <EXP_ID> in the machine-readable sidecar
# (project_planning/experiments.tsv, generated from the tracker by
# scripts/tracker_to_sidecar.py), export the row's knobs as env vars, and invoke the
# matching launcher experiments/train_<task>_<family>.sh.
#
# The sidecar is derived from the tracker; if a knob cell changed, regenerate first:
#   python scripts/tracker_to_sidecar.py
#
# You still MUST claim the row (Status=Processing) via the claim-experiment skill first,
# and provide DATA_TRAIN (the resolved dcfg parquet path) via env if the launcher default
# does not match the row's Data config name.
set -e

REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
SIDECAR="$REPO_DIR/project_planning/experiments.tsv"

EXP_ID="$1"; shift || true
if [ -z "$EXP_ID" ]; then
  echo "usage: $0 <EXP_ID> [extra main_ppo overrides...]" >&2
  exit 1
fi
[ -f "$SIDECAR" ] || { echo "sidecar not found: $SIDECAR (run: python scripts/tracker_to_sidecar.py)" >&2; exit 1; }

# Locate the row (exact match on the exp_id column, which is column 1).
ROW=$(awk -F'\t' -v id="$EXP_ID" 'NR>1 && $1==id {print; found=1} END{if(!found) exit 3}' "$SIDECAR") \
  || { echo "EXP_ID '$EXP_ID' not found in sidecar. Available:" >&2; cut -f1 "$SIDECAR" | tail -n +2 | sed 's/^/  /' >&2; exit 3; }

# Read the matched row into an associative array. NOTE: we must NOT use `read -a` with a
# tab IFS — tab is IFS-whitespace so empty columns collapse and shift every field. awk with
# FS='\t' preserves empty fields; we emit only non-empty KEY<TAB>VAL pairs.
declare -A K
while IFS=$'\t' read -r k v; do
  [ -n "$k" ] && K["$k"]="$v"
done < <(awk -F'\t' -v id="$EXP_ID" '
  NR==1 { for (i=1;i<=NF;i++) h[i]=$i; next }
  $1==id { for (i=1;i<=NF;i++) if ($i!="") print h[i] "\t" $i; exit }
' "$SIDECAR")

FAMILY="${MODEL_FAMILY:-${K[model_family]}}"
TASK="${TASK:-${K[task]}}"
[ -n "$FAMILY" ] || { echo "row '$EXP_ID' has no model_family (fix tracker Model cell, or pass MODEL_FAMILY=)" >&2; exit 4; }
if [ -z "$TASK" ]; then
  echo "row '$EXP_ID' has no resolved task — its Loss/reward type is a placeholder" >&2
  echo "  (e.g. =Pm1a_best/=stable). Resolve it from the upstream Completed row and rerun with" >&2
  echo "  TASK=<behavior|tom_rulebased|sft> plus REWARD_TYPE/POWER_K/POWER_LL_MIN/KL/LR as needed." >&2
  exit 4
fi

# Launcher filename uses a short task token: tom_rulebased -> "tom".
case "$TASK" in
  tom_rulebased) LAUNCHER_TASK=tom ;;
  *)             LAUNCHER_TASK="$TASK" ;;
esac
LAUNCHER="$REPO_DIR/experiments/train_${LAUNCHER_TASK}_${FAMILY}.sh"
[ -f "$LAUNCHER" ] || { echo "no launcher for task='$TASK' family='$FAMILY' ($LAUNCHER)" >&2; exit 5; }

# Map sidecar columns -> launcher env knobs. Env vars already set by the caller WIN over the
# sidecar (needed when resolving tracker placeholders like =Pm1a_best). Only fill unset knobs.
export EXP_ID
[ -z "${MODEL_PATH:-}" ]    && [ -n "${K[model_path]}" ]    && export MODEL_PATH="${K[model_path]}"
[ -z "${REWARD_TYPE:-}" ]   && [ -n "${K[reward_type]}" ]   && export REWARD_TYPE="${K[reward_type]}"
[ -z "${POWER_K:-}" ]       && [ -n "${K[power_k]}" ]       && export POWER_K="${K[power_k]}"
[ -z "${POWER_LL_MIN:-}" ]  && [ -n "${K[power_ll_min]}" ]  && export POWER_LL_MIN="${K[power_ll_min]}"
[ -z "${KL:-}" ]            && [ -n "${K[kl]}" ]            && export KL="${K[kl]}"
[ -z "${LR:-}" ]            && [ -n "${K[lr]}" ]            && export LR="${K[lr]}"
[ -z "${MAX_PROMPT:-}" ]    && [ -n "${K[max_prompt]}" ]    && export MAX_PROMPT="${K[max_prompt]}"
[ -z "${MAX_RESP:-}" ]      && [ -n "${K[max_resp]}" ]      && export MAX_RESP="${K[max_resp]}"
[ -z "${DATA_NAME:-}" ]     && [ -n "${K[data_name]}" ]     && export DATA_NAME="${K[data_name]}"
[ -z "${COT_VAR:-}" ]       && [ -n "${K[cot_var]}" ]       && export COT_VAR="${K[cot_var]}"
if [ -z "${USE_ACTOR_AS_RM:-}" ]; then
  case "${K[rm_mode]}" in
    actor)  export USE_ACTOR_AS_RM=True ;;
    frozen) export USE_ACTOR_AS_RM=False ;;
  esac
fi

# Numeric guard: KL/LR from the tracker may be symbolic (klLo/klHi/=stable) in placeholder rows.
for kn in KL LR; do
  val="${!kn:-}"
  if [ -n "$val" ] && ! printf '%s' "$val" | grep -qE '^-?[0-9.]+([eE]-?[0-9]+)?$'; then
    echo "[dispatch] $kn='$val' is not numeric (unresolved placeholder). Pass $kn=<value> explicitly." >&2
    exit 4
  fi
done

# DATA_TRAIN: prefer explicit env; else derive from DATA_NAME (dcfg stem -> data/<stem>.parquet).
if [ -z "${DATA_TRAIN:-}" ] && [ -n "${DATA_NAME:-}" ]; then
  cand="$REPO_DIR/data/${DATA_NAME}.parquet"
  if [ -f "$cand" ]; then
    export DATA_TRAIN="$cand"
  else
    echo "[dispatch] DATA_TRAIN not set and $cand missing." >&2
    echo "[dispatch] Build it first (python build_dataset.py --config scripts/configs/${DATA_NAME}.yaml)" >&2
    echo "[dispatch] or pass DATA_TRAIN=/path/to.parquet." >&2
    exit 6
  fi
fi

echo "[dispatch] EXP_ID=$EXP_ID -> $LAUNCHER (family=$FAMILY task=$TASK data=$DATA_NAME rm=${K[rm_mode]:-default})"
exec bash "$LAUNCHER" "$@"
