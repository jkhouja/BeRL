# E104 — data-recipe-P0g_filter_off (Gemma-2-2B)

- **RUN_NAME_BASE:** data-recipe-P0g_filter_off-dcfg_mix_best_gemma
- **Exp #:** E104 | **Exp ID:** P0g_filter_off
- **Run name (full):** P0g_filter_off-dcfg_mix_best_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 (planned)
- **Owner_host:** h100-021-003
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/3ulzqqyx (run 3ulzqqyx)
- **Log:** logs/20260716/P0g_filter_off-dcfg_mix_best_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log

## Hypothesis / question
Phase-0 S3 turn-filtering ablation (Gemma-2 arm), mirror of E027. This is the **filter OFF**
control = train on the *full* best-mix corpus (all 11k turns, no surprisal selection). It is the
quantity baseline against which the Gemma surprise arm (E103) and random-length control (E105)
are compared: does selecting the most ToM-dependent turns (E103) beat using the whole corpus?
Filter OFF data = the S2 Gemma winner mix `dcfg_mix_best3_gemma` (craigslist+dailydialog+empathetic)
re-emitted as `dcfg_mix_best_gemma`.

## Implementation details (resolved knobs)
Gemma-2 P0g spec (frozen RM), holding the training config fixed, varying only the data recipe:
- Model: google/gemma-2-2b-it; Gen ctx 2048/512.
- Reward: power, k=5, ll_min=-4, **frozen** RM (USE_ACTOR_AS_RM=False), nobaseline (SUBTRACT_BASELINE=False).
- KL=0.05 (low_var_kl), LR=5e-7, format_penalty=5, entropy_coeff=0.001.
- CoT prompt: COT_FREEFORM (tag-free cot_eval_notags recipe for Gemma).
- Data: `dcfg_mix_best_gemma` = full mix (craigslist+dailydialog+empathetic), turn_filter=off → 11000 rows.

### Data generation (once, shared)
```
python build_dataset.py --config scripts/configs/dcfg_mix_best_gemma.yaml
```
Produces `data/dcfg_mix_best_gemma.parquet`. Config chain:
`dcfg_mix_best_gemma` → `dcfg_mix_best3_gemma` → `dcfg_base_notags`/`dcfg_base`. turn_filter.mode=off
(answer_pp still scored by shared Qwen2.5-3B for cross-arm comparability but not used for filtering).

### Launch command (planned)
```
EXP_ID=P0g_filter_off DATA_NAME=dcfg_mix_best_gemma MODEL_PATH=google/gemma-2-2b-it \
  DATA_TRAIN=data/dcfg_mix_best_gemma.parquet \
  REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False \
  KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 MAX_PROMPT=2048 MAX_RESP=512 \
  bash experiments/train_behavior_gemma.sh
```
Env: conda `tom`; WandB online (project TOM_EXP). Mix data is short-context (dialogue turns,
prompts ~200–600 tok), so no dynamic-bsz OOM guard expected (unlike E100 thoughttrace); add
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096` only if it OOMs.

## Debugging / issues
- 2026-07-16 00:37 data-gen complete: 11000 rows (full mix, filter off), answer_pp scored (491.8s). All GPUs free.
- 2026-07-16 00:43 launched r1 (WandB 3ulzqqyx). 343 steps/1 epoch, test_freq=30, frozen RM. No dynamic-bsz needed (short-context mix).
- 2026-07-16 01:16 **r1 BROKEN** — `critic/score/mean=max=min=-45.000` at every step (std=0), `pg_loss=0.000`, `grad_norm=0.004`: the frozen RM returned the *invalid sentinel* (valid_floor −40 − format_penalty 5) for ALL rollouts → zero GRPO advantage → no gradient. Actor generations were coherent; data/command/code verified byte-identical to the working reference run `P0g_mix_best3-dcfg_mix_best3_gemma` (which reached score ~0.005→0.8, grad_norm ~0.1). Diagnosis: transient frozen-RM NaN (nan_to_num→neginf=invalid_value). Killed r1, all GPUs freed.
- 2026-07-16 01:1x **relaunched as r2** (log ...-r2.log). Watching first steps for recurrence.
- 2026-07-16 01:4x **r2 ALSO floored** at step 1 (`score mean=max=min=-45.000`). Reproducible, not transient. Killed r2, GPUs freed.
- 2026-07-16 **ROOT-CAUSE INVESTIGATION → BLOCKED (shared bug):**
  - Survey of all historical frozen-RM Gemma runs: every *working* run uses an OLD parquet (mix_best3, single_*, mix_all — pre-S3). Both *floored* runs use NEW S3-pipeline parquets: **E103** `filter_surprise` (another agent, step1 −45, incomplete, Awaiting-input) and **E104** `filter_off` (this run). E026 (Qwen, new-pipeline parquet, **actor**-as-RM) trained fine.
  - Reward code (`verl/workers/fsdp_workers.py`) unchanged since 2026-07-06 → both working ref and floored runs use identical code. Not a code-version regression.
  - Parquet diff (new `dcfg_mix_best_gemma` vs working `dcfg_mix_best3_gemma`): RM-relevant fields (prompt, ground_truth, raw_user_prompt, order) **byte-identical**; the ONLY difference is the populated `answer_pp` column (new = floats; old = all None) plus metadata key-order.
  - Standalone gemma-2-2b forward on this node = clean (no NaN, eager+flash) → not a node/model NaN.
  - **Code asymmetry:** frozen-RM path reads `data.batch['answer_pp']` raw (`fsdp_workers.py:1264` and `:1468`); the actor-RM path guards `None` (`:774` `0.0 if _app is None else _app`). Populating answer_pp via the new S3 perplexity pipeline exposes a latent frozen-RM bug → all rollouts scored invalid (-45) → zero GRPO advantage.
  - **Verdict:** shared bug in frozen-RM reward path × new-pipeline parquets. Fix requires touching shared `verl/` reward code (or the shared build_dataset perplexity emission) that live runs depend on → escalated to user per golden rules (E103 already blocked on the same issue). Set E104 → Awaiting-input.

## Findings
BLOCKED pre-fix; **RESOLVED by shared fix `8769467`** (model-aware invalid check — tag-free Gemma2/Qwen3 rollouts no longer force-invalid on missing `</think>`). The real root cause was the invalid-response detector marking ALL tag-free (COT_FREEFORM) Gemma rollouts invalid → −45; the answer_pp correlation was a red herring. Relaunched **r3** (WandB 4d3po0yz, log ...20260717/...-r3.log) — reward healthy at step 1 (`score mean 0.556, max 22.24, min 0.0`, grad_norm 0.140, no −45 floor). Training in progress; final metrics TBD.

### Final result (r3, completed step 343)
**POSITIVE.** AM_tom (arithmetic mean of ToM benches, excl. gsm8k+mmlu): step0 **0.3153 → last-3 0.3747 (+18.8%)**, last-5 0.3789 (+20.2%), peak step270 0.3900. Capability preserved/improved: gsm8k 0.277→0.383, mmlu 0.390→0.410. Top movers (last3 vs step0): opentom_multihop_fo +0.190, opentom_location_so +0.123, explore_tom +0.121, bigtom_forward_action +0.109, opentom_multihop_so +0.101, simpletom_judgment +0.095. No meaningful regressions (worst −0.008, fantom_info_binary). Reward healthy throughout (score oscillated 0–22, grad_norm 0.05–0.18, no −45 floor post-fix).
This is the **filter_OFF quantity baseline** for the Phase-0 S3 Gemma turn-filtering ablation. Comparison verdict (does surprisal *selection* beat using the full corpus?) awaits E103 (surprise), E105 (randlen control), E107 (predictable).

## How to rerun
1. `python build_dataset.py --config scripts/configs/dcfg_mix_best_gemma.yaml`
2. the launch command above.
