#!/bin/bash
# SFT baseline — Qwen2.5 family (Q0 A2: supervised fine-tune on the gold next utterance).
#
# [build] NOT YET WIRED. The verl SFT trainer entrypoint differs from main_ppo
# (e.g. verl.trainer.fsdp_sft_trainer) and must be validated before use. This stub
# documents the intended interface so the dispatcher can route SFT rows once implemented.
#
# TODO(build): implement the SFT invocation here (dataset = gold utterance targets),
# then remove the guard below and the BERL_ALLOW_SFT gate in lib/common.sh.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen2.5
export TASK=sft
echo "[train_sft_qwen2.5] SFT path is a [build] item and is not implemented yet." >&2
echo "  Intended: verl SFT trainer on gold next-utterance targets (Q0 A2)." >&2
exit 2
