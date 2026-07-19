# E026 — data-recipe-P0_filter_surprise (Qwen2.5-3B)

- **RUN_NAME_BASE:** data-recipe-P0_filter_surprise-dcfg_mix_best_surprise
- **Exp #:** E026 | **Exp ID:** P0_filter_surprise
- **Run name (full):** P0_filter_surprise-dcfg_mix_best_surprise-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 (planned r1)
- **Owner_host:** h100-021-003
- **WandB:** TBD (added at launch)
- **Log:** TBD (added at launch)

## Hypothesis / question
Phase-0 S3 turn-filtering ablation (Qwen2.5 arm). Does training only on the **most
surprising / ToM-dependent** human turns (turn *quality*) beat training on the full
best-mix corpus (quantity)? This is the **surprise filter ON** arm: keep the 50% of
turns the frozen Qwen2.5-3B scorer finds least predictable (lowest `answer_pp` = avg
log-prob = highest perplexity = most info-asymmetric). Compare against:
- E027 filter_off (full best-mix corpus, `dcfg_mix_best`), and
- E028 filter_randlen (same size, response-length-matched random control).
Expected: if ToM-dependence matters, surprise-filtered data yields higher/robust AM/HM
ToM gains per training example than the full corpus and the random-length control.

## Implementation details (resolved knobs)
Mirror of the locked Qwen2.5 data-recipe config (E024 P0_mix_best3), holding the training
config fixed and varying only the **data recipe** (turn filter):
- Model: Qwen2.5-3B-Instruct; Gen ctx 2048/512.
- Reward: power, k=5, ll_min=-6, actor-as-RM (USE_ACTOR_AS_RM=True), nobaseline (SUBTRACT_BASELINE=False).
- KL=0.05 (low_var_kl), LR=5e-7, format_penalty=5, entropy_coeff=0.0.
- CoT prompt: COT_FREEFORM (tagged cot_eval recipe, inherited from dcfg_base).
- Data: `dcfg_mix_best_surprise` = casino(3000)+empathetic(4000)+dailydialog(4000) → answer_pp
  scored by Qwen2.5-3B-Instruct → keep 50% lowest answer_pp (most surprising). seed=42.

### Data generation (once, shared)
```
python build_dataset.py --config scripts/configs/dcfg_mix_best_surprise.yaml
```
Produces `data/dcfg_mix_best_surprise.parquet`. Config chain:
`dcfg_mix_best_surprise` → `dcfg_mix_best` → `dcfg_mix_best3` → `dcfg_base`.
`turn_filter.mode=surprise, keep_fraction=0.5, seed=42`; perplexity scorer =
Qwen2.5-3B-Instruct (batch_size=2).

### Launch command (planned)
```
EXP_ID=P0_filter_surprise DATA_NAME=dcfg_mix_best_surprise DATA_TRAIN=data/dcfg_mix_best_surprise.parquet \
  REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False \
  KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 MAX_PROMPT=2048 MAX_RESP=512 \
  bash experiments/train_behavior_qwen2.5.sh
```
Env: conda `tom` (transformers 4.51.3, vLLM 0.6.3, torch 2.4.0); WandB online (project TOM_EXP,
entity jkhouja-oxford). No OOM/dynamic-bsz overrides needed (Qwen2.5, standard 2048 ctx; E024
ran healthily with identical knobs).

## Debugging / issues
(none yet)

## Findings
(filled on completion)

## How to rerun
1. `python build_dataset.py --config scripts/configs/dcfg_mix_best_surprise.yaml` (if parquet missing)
2. the launch command above.

## Launch (r1) — 2026-07-15 23:21
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/ui2xm36n
- **Log:** logs/20260715/P0_filter_surprise-dcfg_mix_best_surprise-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log
- **Data:** data/dcfg_mix_best_surprise.parquet — 5500 rows (dailydialog 2436 + empathetic 2108 + casino 956); surprise filter kept 50% lowest answer_pp (Qwen2.5-3B scorer, resp-words mean 9.8).
- **Startup:** Total training steps=171 (5500/32, 1 epoch), test_freq=30, save_freq=50. All 8 GPUs ~36 GiB, no import/OOM errors. Rollout begun.
- Exact command as in "Launch command (planned)" above (EXP_ID=P0_filter_surprise). driver pid 1795673 / main_task 1803190.

## Findings (r1 — COMPLETE 2026-07-16)
**POSITIVE.** Eval steps [0,30,60,90,120,150,171]; metric = HM_tom (harmonic mean of ToM
benches, excl gsm8k+mmlu), Qwen2.5 family.
- **HM_tom:** baseline(step0)=0.4181 → last-3=0.4593 (**+9.8%**), last-5=0.4609 (+10.2%). Peak step171=0.4681.
- **AM_tom:** 0.5034 → 0.5189 last-3 (+3.1%).
- **Capability preserved/up:** gsm8k 0.663→0.674 (+1.7%), mmlu 0.470→0.620 (+31.9%).
- **Health:** clean — reward/mean ~37–39 (actor-RM power, near +40 ceiling), clip_ratio ~0,
  response_length stable ~90–102, kl_loss ~0.05–0.1, no OOM, no reward-floor. 171 steps / 1 epoch.
- **Trajectory:** HM rises fast (step30 0.4565, step60–90 ~0.463), small dip step120–150, recovers to peak at step171.
- **Top movers (last-3 vs base):** fantom_answerability_list +114%, hi_tom +43%, explore_tom +21%,
  tombench +12%, tomi +6%, fantom_info_list +6%, opentom_location_so +6%. Declines: dyntom_type_a −8%,
  dyntom_type_c −7%, fantom_belief_mc −3%, bigtom_forward_action −3%.

**Verdict:** Surprise/ToM-dependence filtering (train on 5500 = 50% least-predictable turns) gives a
solid HM_tom gain with capability preserved — quality selection works on its own. **Comparative
conclusion pending E027 (filter_off = full 11k best-mix) and E028 (randlen length-matched control):**
only after those can we say whether surprisal *selection* beats quantity/length-matched-random.

## How to rerun
1. `python build_dataset.py --config scripts/configs/dcfg_mix_best_surprise.yaml`
2. `EXP_ID=P0_filter_surprise DATA_NAME=dcfg_mix_best_surprise DATA_TRAIN=data/dcfg_mix_best_surprise.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 MAX_PROMPT=2048 MAX_RESP=512 bash experiments/train_behavior_qwen2.5.sh`
