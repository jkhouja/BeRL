#!/bin/bash
# experiments/lib/common.sh — shared engine for all BeRL GRPO launchers.
#
# This file is NOT executed directly. Each launcher (experiments/train_<task>_<class>.sh)
# sets a handful of env knobs, sources this file, and calls `berl::run`.
#
# Design (see project_planning/BeRL_experiments_tracker.md §"Agent protocol"):
#   - One code path (a single `main_ppo` invocation) parameterised by env knobs.
#   - RUN_NAME = <EXP_ID>-<DATA_NAME>-<MODEL_TAG>-<PARAMS>-r<N>  (= WandB run name = log stem).
#     RUN_NAME_BASE = <EXP_ID>-<DATA_NAME> matches the tracker `Run name` cell.
#   - `-r<N>` (run index) auto-bumps on resubmission so a crashed run never clashes.
#   - Emits a self-contained reproducibility record at experiments_logs/<RUN_NAME_BASE>.md.
#   - WandB online is MANDATORY: WANDB_API_KEY comes from ~/.bashrc; we never unset it,
#     never set WANDB_MODE=offline, and always log to [console,wandb].
#
# TASK ∈ {behavior, tom_rulebased, sft}
# MODEL_FAMILY ∈ {qwen2.5, qwen3, gemma}

set -o pipefail

REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
berl::activate_env() {
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" >/dev/null 2>&1 || true
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook 2>/dev/null)"
  conda activate "${CONDA_ENV:-tom}"
  # WandB online is mandatory — do NOT unset WANDB_API_KEY or set WANDB_MODE=offline.
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[common] WARNING: WANDB_API_KEY is not set; expected it in ~/.bashrc for online logging." >&2
  fi
}

# Per-model-family knobs. Launchers may pre-set any of these to override.
berl::model_family_defaults() {
  case "$MODEL_FAMILY" in
    qwen2.5)
      export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
      REQUIRE_ANSWER_TAGS="${REQUIRE_ANSWER_TAGS:-True}"
      GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
      MAX_RESP="${MAX_RESP:-4096}"
      MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
      FOLD_SYSTEM_PROMPT="${FOLD_SYSTEM_PROMPT:-False}"
      ;;
    qwen3)
      export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
      REQUIRE_ANSWER_TAGS="${REQUIRE_ANSWER_TAGS:-False}"
      GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
      MAX_RESP="${MAX_RESP:-4096}"
      MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
      FOLD_SYSTEM_PROMPT="${FOLD_SYSTEM_PROMPT:-False}"
      ;;
    gemma)
      # Gemma2 needs FLASH_ATTN (not XFORMERS) for logits soft-capping.
      export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      REQUIRE_ANSWER_TAGS="${REQUIRE_ANSWER_TAGS:-False}"
      GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
      MAX_RESP="${MAX_RESP:-1024}"
      MODEL_PATH="${MODEL_PATH:-google/gemma-2-2b-it}"
      FOLD_SYSTEM_PROMPT="${FOLD_SYSTEM_PROMPT:-True}"
      ;;
    *)
      echo "[common] ERROR: unknown MODEL_FAMILY='$MODEL_FAMILY' (expected qwen2.5|qwen3|gemma)" >&2
      return 1
      ;;
  esac
  MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_PATH")}"
}

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
berl::build_params_tag() {
  local rm_type baseline_tag
  rm_type=$( [ "${USE_ACTOR_AS_RM:-False}" = "True" ] && echo "actorRM" || echo "frozenRM" )
  baseline_tag=$( [ "${SUBTRACT_BASELINE:-False}" = "True" ] && echo "baseline" || echo "nobaseline" )
  case "$TASK" in
    behavior)
      PARAMS_TAG="${rm_type}-${baseline_tag}-${REWARD_TYPE}-k${POWER_K}-llmin${POWER_LL_MIN}-lr${LR}-kl${KL}-n${ROLLOUT_N}"
      ;;
    tom_rulebased)
      PARAMS_TAG="rulebased-lr${LR}-kl${KL}-n${ROLLOUT_N}"
      ;;
    sft)
      PARAMS_TAG="sft-lr${LR}-ep${TOTAL_EPOCHS}"
      ;;
    *)
      PARAMS_TAG="lr${LR}-kl${KL}-n${ROLLOUT_N}"
      ;;
  esac
}

# RUN_NAME_BASE = <EXP_ID>-<DATA_NAME> ; full RUN_NAME adds -<MODEL_TAG>-<PARAMS>-r<N>.
berl::resolve_run_name() {
  : "${EXP_ID:?set EXP_ID (tracker Exp ID, or e.g. test-smoke)}"
  : "${DATA_NAME:?set DATA_NAME (tracker Data config name, dcfg_* stem)}"
  RUN_NAME_BASE="${EXP_ID}-${DATA_NAME}"
  local stem="${RUN_NAME_BASE}-${MODEL_TAG}-${PARAMS_TAG}"
  if [ -n "${RUN_INDEX:-}" ]; then
    :
  else
    # Auto-pick the next free run index by scanning existing logs for this stem.
    local maxn
    maxn=$(ls "$REPO_DIR"/logs/*/"${stem}"-r*.log 2>/dev/null \
             | grep -oE 'r[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
    RUN_INDEX=$(( ${maxn:-0} + 1 ))
  fi
  RUN_NAME="${stem}-r${RUN_INDEX}"
}

# ---------------------------------------------------------------------------
# Reproducibility record
# ---------------------------------------------------------------------------
berl::write_summary_md() {
  local md="$REPO_DIR/experiments_logs/${RUN_NAME_BASE}.md"
  mkdir -p "$REPO_DIR/experiments_logs"
  local commit; commit=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
  local host; host=$(hostname)
  local now; now=$(date -Iseconds)
  {
    if [ ! -f "$md" ]; then
      echo "# ${RUN_NAME_BASE}"
      echo
      echo "- **Exp ID:** ${EXP_ID}"
      echo "- **Data config:** ${DATA_NAME}"
      echo "- **Task:** ${TASK}   **Model family:** ${MODEL_FAMILY}"
      echo
      echo "## Hypothesis / question"
      echo "_(fill from the tracker row)_"
      echo
      echo "## Attempts"
      echo
    fi
    echo "### Attempt r${RUN_INDEX} — ${now}"
    echo
    echo "- **RUN_NAME:** \`${RUN_NAME}\`"
    echo "- **Host:** ${host}   **git:** \`${commit}\`   **conda env:** ${CONDA_ENV:-tom}"
    echo "- **Model:** \`${MODEL_PATH}\` (${MODEL_TAG})"
    echo "- **Data (train):** \`${DATA_TRAIN}\`"
    echo "- **Val files:** \`${VAL_FILES}\`"
    echo "- **Knobs:** reward=${REWARD_TYPE:-n/a} power_k=${POWER_K:-n/a} ll_min=${POWER_LL_MIN:-n/a} rm_mode=$( [ "${USE_ACTOR_AS_RM:-False}" = "True" ] && echo actor || echo frozen ) baseline=${SUBTRACT_BASELINE:-n/a} kl=${KL} lr=${LR} rollout_n=${ROLLOUT_N} epochs=${TOTAL_EPOCHS} max_ctx=${MAX_PROMPT}/${MAX_RESP} cot_var=${COT_VAR:-cot_eval} require_answer_tags=${REQUIRE_ANSWER_TAGS} entropy_coeff=${ENTROPY_COEFF:-0.001} think_only_pg=${THINK_ONLY_PG:-False} format_penalty=${FORMAT_PENALTY:-0.0}"
    echo "- **Env:** VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} GPU_MEM_UTIL=${GPU_MEM_UTIL} TP=${TP_SIZE:-2} n_gpus=${NUM_GPUS}"
    echo "- **WandB:** project=${PROJECT_NAME} run=${RUN_NAME} _(paste link after launch)_"
    echo "- **Log path:** \`logs/${TODAY}/${RUN_NAME}.log\`"
    echo
    echo "**Exact command:**"
    echo '```bash'
    printf '%s\n' "$BERL_CMD_STR"
    echo '```'
    echo
    echo "**How to rerun:** \`$BERL_RERUN\`"
    echo
    echo "**Findings:** _(fill on completion via log-results skill)_"
    echo
  } >> "$md"
  echo "[common] reproducibility record: $md"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
berl::run() {
  berl::activate_env
  berl::model_family_defaults || return 1

  # Shared defaults (override via env before sourcing).
  TASK="${TASK:?set TASK (behavior|tom_rulebased|sft)}"
  # GPU pinning: GPU_IDS (comma-sep) -> CUDA_VISIBLE_DEVICES; NUM_GPUS derived from it.
  if [ -n "${GPU_IDS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | grep -c .)
  fi
  NUM_GPUS="${NUM_GPUS:-8}"
  TP_SIZE="${TP_SIZE:-2}"
  TRAIN_BATCH="${TRAIN_BATCH:-32}"
  VAL_BATCH="${VAL_BATCH:-16}"
  MINI_BATCH="${MINI_BATCH:-128}"
  MICRO_BATCH="${MICRO_BATCH:-8}"
  ROLLOUT_N="${ROLLOUT_N:-16}"
  LR="${LR:-5e-7}"
  MAX_PROMPT="${MAX_PROMPT:-2048}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  SAVE_FREQ="${SAVE_FREQ:-50}"
  TEST_FREQ="${TEST_FREQ:-30}"
  PROJECT_NAME="${PROJECT_NAME:-TOM_EXP}"
  GRAD_CKPT="${GRAD_CKPT:-True}"
  DATA_TRAIN="${DATA_TRAIN:?set DATA_TRAIN (train parquet path)}"
  # --- Eval suite selection -------------------------------------------------
  # VAL_SUITE picks which benchmark parquets to evaluate on (override with an
  # explicit VAL_FILES=[...] to bypass).
  #   subsample300 — DEFAULT. Stratified representative subset:
  #                  min(300,available) rows per subtype (26 subtypes, ~7,800
  #                  prompts; ullman excluded, gsm8k numeric guardrail included).
  #                  ~3.5pp per-subtype CI; ~1.6 min eval on Qwen2.5-3B/8xH100.
  #                  Rebuild via examples/data_preprocess/build_eval_subsample.py.
  #   full         — every benchmark at full size (~40k prompts; ~8 min eval).
  #                  Use for final headline reporting, not every TEST_FREQ.
  #   core         — tomi/explore/hi + fantom (the legacy default).
  #   sanity       — data/cleaned_tom/eval_suite_sanity.parquet (~40/subtype).
  local ET="$REPO_DIR/data/cleaned_tom"
  if [ -z "${VAL_FILES:-}" ]; then
    case "${VAL_SUITE:-subsample300}" in
      core)
        VAL_FILES="[$ET/ToM_test_HiExTi_hint_v3.parquet,$ET/fantom_test_50pct.parquet]"
        VAL_METRIC_SUFFIX="${VAL_METRIC_SUFFIX:-_core}" ;;
      sanity)
        VAL_FILES="[$ET/eval_suite_sanity.parquet]"
        VAL_METRIC_SUFFIX="${VAL_METRIC_SUFFIX:-_sanity}" ;;
      full)
        VAL_FILES="[$ET/ToM_test_HiExTi_hint_v3.parquet,$ET/fantom_test_50pct.parquet,$ET/bigtom_test.parquet,$ET/dyntom_test.parquet,$ET/exploretom_infilled_test.parquet,$ET/mmlu_test.parquet,$ET/opentom_test.parquet,$ET/simpletom_test.parquet,$ET/tombench_test.parquet,$ET/gsm8k_test.parquet]"
        VAL_METRIC_SUFFIX="${VAL_METRIC_SUFFIX:-}" ;;   # full suite keeps canonical val/test_score/<src> names
      subsample300|*)
        VAL_FILES="[$ET/eval_subsample_300.parquet]"
        VAL_METRIC_SUFFIX="${VAL_METRIC_SUFFIX:-_sub300}" ;;
    esac
  fi
  # Suffix appended to every val metric name so a subset run's WandB series
  # (e.g. val/test_score/tomi_sub300) never overwrites the full-suite series.
  VAL_METRIC_SUFFIX="${VAL_METRIC_SUFFIX:-}"

  # Anti-collapse / policy-shaping knobs (added 238137a). Defaults mirror ppo_trainer.yaml
  # so unset behavior is unchanged; the phase-stability sweep varies these per cell.
  ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"   # actor_rollout_ref.actor.entropy_coeff
  THINK_ONLY_PG="${THINK_ONLY_PG:-False}"   # actor_rollout_ref.actor.think_only_pg
  FORMAT_PENALTY="${FORMAT_PENALTY:-0.0}"   # actor.format_penalty (+ reward_model.format_penalty)

  if [ "$TASK" = "behavior" ]; then
    KL="${KL:-0.05}"
    REWARD_TYPE="${REWARD_TYPE:-power}"
    POWER_K="${POWER_K:-2.0}"
    POWER_LL_MIN="${POWER_LL_MIN:--8.0}"
    USE_ACTOR_AS_RM="${USE_ACTOR_AS_RM:-True}"
    SUBTRACT_BASELINE="${SUBTRACT_BASELINE:-False}"
  else
    KL="${KL:-0.001}"
  fi

  berl::build_params_tag
  berl::resolve_run_name

  TODAY=$(date +%Y%m%d)
  mkdir -p "$REPO_DIR/logs/${TODAY}"
  local logfile="$REPO_DIR/logs/${TODAY}/${RUN_NAME}.log"

  if [ "$TASK" = "sft" ]; then
    echo "[common] TASK=sft is a [build] item — the verl SFT trainer path is not wired yet." >&2
    echo "[common] Set BERL_ALLOW_SFT=1 to attempt anyway once experiments/train_sft_*.sh implements it." >&2
    [ "${BERL_ALLOW_SFT:-0}" = "1" ] || return 2
  fi

  # ----- assemble the main_ppo argument list -----
  local -a ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="$DATA_TRAIN"
    data.val_files="$VAL_FILES"
    data.val_metric_suffix="$VAL_METRIC_SUFFIX"
    data.train_batch_size="$TRAIN_BATCH"
    data.val_batch_size="$VAL_BATCH"
    data.prompt_is_text=False
    +data.truncation=left
    data.max_prompt_length="$MAX_PROMPT"
    data.max_response_length="$MAX_RESP"
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.actor.optim.lr="$LR"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH"
    actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="$KL"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
    actor_rollout_ref.actor.think_only_pg="$THINK_ONLY_PG"
    actor_rollout_ref.actor.format_penalty="$FORMAT_PENALTY"
    actor_rollout_ref.model.enable_gradient_checkpointing="$GRAD_CKPT"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.grad_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$MICRO_BATCH"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.ref.log_prob_micro_batch_size="$MICRO_BATCH"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.kl_ctrl.kl_coef="$KL"
    trainer.critic_warmup=0
    "trainer.logger=['console','wandb']"
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$RUN_NAME"
    trainer.n_gpus_per_node="$NUM_GPUS"
    trainer.nnodes=1
    trainer.default_hdfs_dir=null
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
  )

  # Behavior reward (LM log-likelihood family). tom_rulebased omits the reward_model
  # block entirely and relies on data_source→rule-based scoring in main_ppo.
  if [ "$TASK" = "behavior" ]; then
    ARGS+=(
      reward_model.type=lm
      reward_model.enable=True
      reward_model.model.path="$MODEL_PATH"
      reward_model.micro_batch_size="$MICRO_BATCH"
      +reward_model.subtract_baseline="$SUBTRACT_BASELINE"
      +reward_model.use_actor_as_rm="$USE_ACTOR_AS_RM"
      +reward_model.reward_type="$REWARD_TYPE"
      +reward_model.power_k="$POWER_K"
      +reward_model.power_ll_min="$POWER_LL_MIN"
      reward_model.format_penalty="$FORMAT_PENALTY"
    )
    # Actor-as-RM reads the reward config off actor_rollout_ref (fsdp_workers.py:171),
    # while the frozen RewardModelWorker reads it off reward_model.*. Wire BOTH for every
    # family so actor-as-RM uses the intended reward_type/power params consistently — not
    # just Gemma. Previously these were Gemma-only, so Qwen2.5/Qwen3 actor-as-RM runs
    # silently fell back to the log_prob default. Still overridable via REWARD_TYPE /
    # POWER_K / POWER_LL_MIN env vars and trailing hydra "$@" overrides.
    ARGS+=(
      +actor_rollout_ref.reward_type="$REWARD_TYPE"
      +actor_rollout_ref.power_k="$POWER_K"
      +actor_rollout_ref.power_ll_min="$POWER_LL_MIN"
    )
  fi

  # Gemma folds the system prompt into the user turn (no system role in chat template).
  if [ "$FOLD_SYSTEM_PROMPT" = "True" ]; then
    ARGS+=( +data.fold_system_prompt=True )
  fi

  # Prompt-alignment Option A: when the CoT/system prompt varies from cot_eval, force the
  # identical system prompt into BOTH train and val loaders (see launch-experiment §4a).
  if [ -n "${SYSTEM_PROMPT:-}" ]; then
    ARGS+=( +data.system_prompt="$SYSTEM_PROMPT" )
  fi

  # Answer-tag expectation is authoritative via model-type parser; these flags are
  # informational/back-compat and kept aligned with the family (see §4b).
  ARGS+=(
    +reward_model.require_answer_tags="$REQUIRE_ANSWER_TAGS"
    +actor_rollout_ref.require_answer_tags="$REQUIRE_ANSWER_TAGS"
  )

  # Build a human-readable command string for the reproducibility record.
  BERL_CMD_STR="HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo"
  local a
  for a in "${ARGS[@]}"; do BERL_CMD_STR+=$' \\\n    '"$a"; done
  local launcher_task="$TASK"; [ "$TASK" = "tom_rulebased" ] && launcher_task="tom"
  BERL_RERUN="EXP_ID=$EXP_ID DATA_NAME=$DATA_NAME MODEL_PATH=$MODEL_PATH DATA_TRAIN=$DATA_TRAIN RUN_INDEX=$RUN_INDEX bash experiments/train_${launcher_task}_${MODEL_FAMILY}.sh"

  if [ "${BERL_NO_EXP_LOG:-0}" = "1" ]; then
    echo "[common] BERL_NO_EXP_LOG=1 — skipping experiments_logs reproducibility record (WandB still on)"
  else
    berl::write_summary_md
  fi

  echo "[common] launching RUN_NAME=$RUN_NAME"
  echo "[common] log: $logfile"
  cd "$REPO_DIR" || return 1
  if [ "${BERL_DRY_RUN:-0}" = "1" ]; then
    echo "[common] DRY RUN — command that would execute:"
    printf '%s\n' "$BERL_CMD_STR"
    [ "$#" -gt 0 ] && echo "[common] + extra overrides: $*"
    return 0
  fi
  set -x
  HYDRA_FULL_ERROR=1 RAY_BACKEND_LOG_LEVEL=debug \
    python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@" 2>&1 | tee "$logfile"
}
