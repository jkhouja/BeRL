#!/bin/bash
# Auto-restart supervisor for the Gemma2-2b GRPO run on a SHARED (non-exclusive) node.
# Motivation: the recipe (eager attn + soft-capping, use_remove_padding=False, dynamic-bsz
# memory capping, log-prob reward, actor-as-RM) is validated and trains stably. The only
# remaining failure mode is an EXTERNAL process that periodically grabs ~73GB on a random GPU
# and co-locates onto a GPU we are using mid-run, OOM-killing us. verl (this version) has no
# training-state resume, so we relaunch from scratch and rely on catching a long clean window.
#
# Each attempt: wait until >=4 GPUs are essentially free, pick the 4 most-free GPUs (to dodge
# the current squatter), then launch the training script pinned to those GPUs via GEMMA_GPUS.
set -u
REPO_DIR=$HOME/repo/BeRL
cd "$REPO_DIR"
# HF_TOKEN must be provided via the environment (e.g. export it in ~/.bashrc); do not hardcode secrets.
: "${HF_TOKEN:?HF_TOKEN is not set. Export it before running this script.}"
export HF_TOKEN

MAX_RETRIES=${MAX_RETRIES:-40}
NEED_GPUS=8
FREE_MB=4000            # a GPU is "free" if it uses < this many MB
SUP_LOG=logs/gemma2_supervisor.log
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUP_LOG"; }

retry=0
while [ "$retry" -lt "$MAX_RETRIES" ]; do
    # Guard: if a training process is already running (e.g. this supervisor was just started to
    # take over from a previous supervisor while a run is in flight), wait for it to finish rather
    # than launching a conflicting second run. Detect completion via the newest run log's marker.
    if ps -u "$USER" -o cmd 2>/dev/null | grep -q 'trainer.main[_]ppo'; then
        log "training already running; waiting for it to finish before managing restarts."
        while ps -u "$USER" -o cmd 2>/dev/null | grep -q 'trainer.main[_]ppo'; do sleep 60; done
        newest=$(ls -t logs/gemma2_2b_launch_*.out 2>/dev/null | head -1)
        if [ -n "$newest" ] && grep -q "Final validation metrics" "$newest" 2>/dev/null; then
            log "in-flight run completed (Final validation metrics in $newest); supervisor exiting."
            break
        fi
        log "in-flight run ended without completion marker; will relaunch."
        ray stop --force >/dev/null 2>&1
        sleep 20
    fi

    # Wait (up to ~30 min) for enough free GPUs.
    picked=""
    for w in $(seq 1 60); do
        mapfile -t free < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                             | awk -v m="$FREE_MB" -F', *' '$2 < m {print $1}')
        if [ "${#free[@]}" -ge "$NEED_GPUS" ]; then
            # Prefer the HIGHEST-index free GPUs. The external squatter on this shared node
            # repeatedly targets low indices (GPU 0/1 hit by 5+ external PIDs), while GPUs 4-7
            # stay pristine, so biasing toward high indices minimizes mid-run OOM collisions.
            picked=$(printf '%s\n' "${free[@]}" | sort -rn | head -n "$NEED_GPUS" | sort -n | paste -sd, -)
            break
        fi
        log "only ${#free[@]} free GPUs, waiting... (probe $w)"
        sleep 30
    done
    if [ -z "$picked" ]; then
        log "no $NEED_GPUS free GPUs after waiting; retry=$retry"
        retry=$((retry+1)); sleep 30; continue
    fi

    ts=$(date +%Y%m%d_%H%M%S)
    echo "$ts" > /tmp/gemma_ts.txt
    runlog="logs/gemma2_2b_launch_${ts}.out"

    # Warm-start from the MOST RECENTLY SAVED Gemma actor checkpoint if one exists (cumulative
    # progress across environmental restarts). save_pretrained dirs are HF-loadable. Match ONLY
    # this run's experiment name (logprob-reward-eager-fantom) so we never warm-start from an
    # older, buggy lineage (e.g. the FA2 logprob-reward-fantom or power-reward checkpoints).
    # NOTE: pick by MODIFICATION TIME (ls -t), NOT by step number: the global-step counter resets
    # to 0 on every warm-started restart, so verl repeatedly writes global_step_10/20; a numeric
    # sort could resume from a stale global_step_20 after a newer cycle overwrote global_step_10.
    # Newest-mtime is the true "latest weights".
    ckpt_root=$(ls -d checkpoints/*/*gemma-2-2b-it*logprob-reward-eager-fantom/actor 2>/dev/null | head -1)
    init_arg=""
    if [ -n "$ckpt_root" ]; then
        latest=$(ls -dt "$ckpt_root"/global_step_* 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$latest/config.json" ]; then
            init_arg="GEMMA_INIT_MODEL=$latest"
            log "warm-starting from checkpoint: $latest"
        fi
    fi

    log "attempt $retry: launching on GPUs [$picked] ${init_arg:+(warm-start)} -> $runlog"
    env $init_arg GEMMA_GPUS="$picked" bash experiments/gemma2_2b_all_dialogue.sh > "$runlog" 2>&1
    code=$?
    log "attempt $retry exited (code $code)."

    # If it ran to completion (verl prints this at the end of fit()), stop.
    if grep -q "Final validation metrics" "$runlog" 2>/dev/null; then
        log "detected completion marker (Final validation metrics); supervisor exiting."
        break
    fi

    # Clean any orphaned ray/workers before retrying.
    ray stop --force >/dev/null 2>&1
    retry=$((retry+1))
    sleep 20
done
log "supervisor done after $retry attempt(s)."
