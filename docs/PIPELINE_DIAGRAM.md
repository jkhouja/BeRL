```
BeRL/TomRL GRPO Training Pipeline
==================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DATA GENERATION                             │
│                                                                             │
│  HuggingFace Datasets                                                       │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────┐              │
│  │  DailyDialog  │  │ EmpathicDialogues │  │ Hi-ToM/ExploreToM│              │
│  └──────┬───────┘  └────────┬──────────┘  └────────┬─────────┘              │
│         │                   │                      │                        │
│         ▼                   ▼                      ▼                        │
│  convert_dailydialog.py    convert_empathetic_     convert_theory_          │
│                            dialogues.py            of_mind.py              │
│         │                   │                      │                        │
│         │  Split each conversation into            │                        │
│         │  (history, next_turn) pairs              │                        │
│         ▼                   ▼                      ▼                        │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  Parquet Columns:                                               │        │
│  │  ┌─────────┐ ┌───────────┐ ┌─────────────┐ ┌───────────────┐  │        │
│  │  │ prompt  │ │raw_prompt │ │ data_source │ │ reward_model  │  │        │
│  │  │(chat    │ │(sys+user  │ │(dailydialog/│ │{ground_truth, │  │        │
│  │  │template)│ │ dicts)    │ │ empathetic/ │ │ style}        │  │        │
│  │  │         │ │           │ │ hi_tom/     │ │               │  │        │
│  │  │         │ │           │ │ explore_tom)│ │               │  │        │
│  │  └─────────┘ └───────────┘ └─────────────┘ └───────────────┘  │        │
│  │  + metadata, response_words, ability, answer_pp                │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                     ┌──────────────┼──────────────┐                         │
│                     ▼              ▼              ▼                         │
│              Individual      Pipeline Config    Merge Script                │
│              .parquet        (YAML filters:     ───────────►                │
│              files           min_turns,         merged_dialogue_            │
│                              min_response_      datasets_filtered_          │
│                              words, sample)     eval_prompt.parquet         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: DATA LOADING                                │
│                    verl/utils/dataset/rl_dataset.py                          │
│                                                                             │
│  pd.read_parquet(train_files)                                               │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────┐                           │
│  │ prompt_is_text=False:                        │                           │
│  │   tokenizer.apply_chat_template(             │                           │
│  │     prompt,  ◄── list of {role, content}     │                           │
│  │     add_generation_prompt=True               │                           │
│  │   )                                          │                           │
│  │                                              │                           │
│  │ Tokenize + Left-pad to max_prompt_length     │                           │
│  └──────────────────┬──────────────────────────┘                           │
│                     │                                                       │
│                     ▼                                                       │
│  ┌──────────────────────────────────────────────┐                          │
│  │ Output per sample:                            │                          │
│  │   input_ids      [max_prompt_length]          │                          │
│  │   attention_mask  [max_prompt_length]          │                          │
│  │   position_ids    [max_prompt_length]          │                          │
│  │   + raw_prompt, data_source, reward_model     │                          │
│  └──────────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  STAGE 3: ROLLOUT / RESPONSE GENERATION                     │
│               verl/workers/rollout/vllm_rollout/vllm_rollout.py             │
│                                                                             │
│  ┌────────────────────┐                                                     │
│  │  vLLM Engine        │                                                     │
│  │  (tensor_parallel=2)│                                                     │
│  │                     │                                                     │
│  │  For each prompt,   │     sampling_params:                               │
│  │  generate N=16      │◄─── temperature=1.0                                │
│  │  candidate          │     top_k=-1, top_p=1.0                            │
│  │  responses           │                                                    │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  batch_size=32 × N=16 = 512 (prompt, response) pairs per step              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 4: LOG PROBABILITY COMPUTATION                          │
│                    verl/workers/fsdp_workers.py                              │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │      Actor Model            │    │    Reference Model (frozen)  │        │
│  │                             │    │                              │        │
│  │  Forward(prompt+response)   │    │  Forward(prompt+response)    │        │
│  │         │                   │    │         │                    │        │
│  │         ▼                   │    │         ▼                    │        │
│  │  old_log_probs              │    │  ref_log_probs               │        │
│  │  [bs, response_len]         │    │  [bs, response_len]          │        │
│  └─────────────────────────────┘    └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 5: REWARD COMPUTATION                             │
│                    verl/workers/fsdp_workers.py                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    TWO REWARD MODES                              │        │
│  │                                                                  │        │
│  │  ┌──────────────────────┐    ┌────────────────────────────┐     │        │
│  │  │  MODE A: LL-based    │    │  MODE B: Rule-based         │     │        │
│  │  │  (Frozen or Actor RM)│    │  (for direct ToM training)  │     │        │
│  │  │                      │    │                              │     │        │
│  │  │  Build target:       │    │  Extract <answer> tag from   │     │        │
│  │  │  prompt +             │    │  generated response          │     │        │
│  │  │  <think>response      │    │         │                    │     │        │
│  │  │  </think><answer>     │    │         ▼                    │     │        │
│  │  │  ground_truth         │    │  Format check: ±1            │     │        │
│  │  │  </answer>            │    │  Answer match: ±2            │     │        │
│  │  │       │               │    │  Total: [-1, +3]             │     │        │
│  │  │       ▼               │    │                              │     │        │
│  │  │  log P(GT|reasoning)  │    └────────────────────────────┘     │        │
│  │  │       │               │                                       │        │
│  │  │       ▼               │                                       │        │
│  │  │  Reward Transform:    │                                       │        │
│  │  │  ┌─────────────────┐  │                                       │        │
│  │  │  │ "power" mode:   │  │                                       │        │
│  │  │  │                 │  │                                       │        │
│  │  │  │ shifted = max(  │  │                                       │        │
│  │  │  │  ll - ll_min, 0)│  │   ll_min=-8: ~70% samples get        │        │
│  │  │  │                 │  │   non-zero reward (vs 21% at -5)      │        │
│  │  │  │ reward =        │  │                                       │        │
│  │  │  │  shifted ^ k    │  │   k=2: convex scaling amplifies       │        │
│  │  │  │  (k=2.0)        │  │   high-quality predictions            │        │
│  │  │  └─────────────────┘  │                                       │        │
│  │  │       │               │                                       │        │
│  │  │       ▼               │                                       │        │
│  │  │  Frozen RM: uses      │                                       │        │
│  │  │   base model weights  │                                       │        │
│  │  │  Actor-as-RM: uses    │                                       │        │
│  │  │   current actor       │                                       │        │
│  │  │   weights (evolving   │                                       │        │
│  │  │   curriculum effect)  │                                       │        │
│  │  └──────────────────────┘                                        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│              clamp(reward, MIN=-40, MAX=+40)                                │
│                          │                                                  │
│                          ▼                                                  │
│              token_level_scores [bs, response_len]                          │
│              (reward placed on last token before EOS)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 6: KL PENALTY & REWARD ADJUSTMENT                   │
│                   verl/trainer/ppo/ray_trainer.py                            │
│                                                                             │
│  KLD = old_log_probs - ref_log_probs    (per-token KL divergence)           │
│                                                                             │
│  token_level_rewards = token_level_scores - β × KLD                         │
│                                          │                                  │
│                                    β = kl_coef                              │
│                                    (0.05 default,                           │
│                                     0.01 for lower KL)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 7: GRPO ADVANTAGE ESTIMATION                       │
│                    verl/trainer/ppo/core_algos.py                            │
│                                                                             │
│  For each prompt (batch_size=32), there are N=16 candidate responses:       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │ Prompt_i ──► Response_1  ──► reward_1 = 24.3        │                    │
│  │          ──► Response_2  ──► reward_2 = 18.7        │                    │
│  │          ──► Response_3  ──► reward_3 = 31.5        │                    │
│  │          ──► ...                                     │                    │
│  │          ──► Response_16 ──► reward_16 = 22.1       │                    │
│  └─────────────────────────────────────────────────────┘                    │
│                          │                                                  │
│                          ▼                                                  │
│  1. Scalar reward: score_j = Σ token_level_rewards_j                        │
│                                                                             │
│  2. Group stats:  μ_i = mean(score_1..score_16)                             │
│                   σ_i = std(score_1..score_16)                               │
│                                                                             │
│  3. Normalize:    advantage_j = (score_j - μ_i) / (σ_i + ε)               │
│                                                                             │
│  4. Expand:       advantages_j = advantage_j repeated across                │
│                                  all response tokens                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │ Key insight: GRPO needs NO critic network.               │               │
│  │ The group of N responses provides its own baseline.      │               │
│  │ Better-than-average responses get positive advantage,    │               │
│  │ worse-than-average get negative.                         │               │
│  └──────────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 8: POLICY UPDATE (PPO)                           │
│                      verl/workers/actor/dp_actor.py                          │
│                                                                             │
│  For each PPO epoch (1 epoch):                                              │
│    Shuffle data into mini-batches (ppo_mini_batch_size=128)                 │
│    For each mini-batch:                                                     │
│      For each micro-batch (ppo_micro_batch_size=8):                         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │  1. FORWARD PASS                                       │                 │
│  │     logits = actor(prompt + response)                   │                 │
│  │     log_prob = log_softmax(logits)[response_tokens]     │                 │
│  │     entropy = -Σ p·log(p)                               │                 │
│  │                                                         │                 │
│  │  2. PPO CLIPPED LOSS                                    │                 │
│  │     ratio = exp(log_prob - old_log_prob)                 │                 │
│  │     loss_unclipped = -advantages × ratio                │                 │
│  │     loss_clipped = -advantages ×                        │                 │
│  │                    clamp(ratio, 1-ε, 1+ε)   ε=0.2      │                 │
│  │     pg_loss = max(loss_unclipped, loss_clipped)         │                 │
│  │                                                         │                 │
│  │  3. ENTROPY BONUS                                       │                 │
│  │     entropy_loss = mean(entropy)                        │                 │
│  │                                                         │                 │
│  │  4. KL REGULARIZATION (GRPO-specific)                   │                 │
│  │     kl_loss = mean(log_prob - ref_log_prob)             │                 │
│  │                                                         │                 │
│  │  5. TOTAL LOSS                                          │                 │
│  │     loss = pg_loss                                      │                 │
│  │          - 0.001 × entropy_loss                         │                 │
│  │          + 0.05  × kl_loss        (kl_loss_coef)        │                 │
│  │                                                         │                 │
│  │  6. BACKWARD + OPTIMIZE                                 │                 │
│  │     loss.backward()                                     │                 │
│  │     clip_grad_norm_(params, max_norm=1.0)               │                 │
│  │     optimizer.step(lr=5e-7)                             │                 │
│  └────────────────────────────────────────────────────────┘                 │
│                                                                             │
│  Actor weights updated ──► used for next rollout (and actor-as-RM)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                          ▼                     ▼
                    Next training          Every test_freq=10
                    step (loop             steps: VALIDATE
                    back to Stage 3)
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 9: VALIDATION                                    │
│                  verl/trainer/ppo/ray_trainer.py                             │
│                                                                             │
│  Eval dataset: ToM_test_HiExTi_hint_v3.parquet                              │
│  (8,060 samples: 5,994 tomi + 1,066 explore_tom + 1,000 hi_tom)            │
│                                                                             │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │  1. Generate responses (do_sample=False, greedy)        │                 │
│  │                                                         │                 │
│  │  2. Score via rule-based reward:                         │                 │
│  │     Extract <answer>...</answer> from response           │                 │
│  │     Compare to ground truth (exact match)               │                 │
│  │     Score = 3 if format correct + answer correct         │                 │
│  │                                                         │                 │
│  │  3. Aggregate by data_source:                           │                 │
│  │     val/test_score/tomi =                               │                 │
│  │       count(reward == 3) / 5994                         │                 │
│  │     val/test_score/explore_tom =                        │                 │
│  │       count(reward == 3) / 1066                         │                 │
│  │     val/test_score/hi_tom =                             │                 │
│  │       count(reward == 3) / 1000                         │                 │
│  │                                                         │                 │
│  │  4. Log to wandb + console                              │                 │
│  └────────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘


TRAINING LOOP SUMMARY
=====================

  ┌─────────┐     ┌──────────┐     ┌────────┐     ┌──────────┐
  │ Generate │────►│ Compute  │────►│ Compute│────►│  Update  │
  │ N=16     │     │ Rewards  │     │ GRPO   │     │  Actor   │
  │ responses│     │ (LL or   │     │ Advant.│     │  (PPO    │
  │ per      │     │  rule)   │     │ (group │     │  clipped │
  │ prompt   │     │          │     │  norm) │     │  loss)   │
  └─────────┘     └──────────┘     └────────┘     └──────────┘
       ▲                                                │
       │                                                │
       └────────────────────────────────────────────────┘
                    Repeat for 2 epochs
               (eval every 10 steps via rule-based scoring)


KEY HYPERPARAMETERS (17f best config)
=====================================

  Data:     6,214 filtered dialogue samples, cot_eval system prompt
  Rollout:  N=16 responses per prompt, temperature=1.0
  Reward:   Power-law LL, ll_min=-8, k=2, clip=(-40,+40), actor-as-RM
  GRPO:     KL=0.05, clip_ratio=0.2, entropy_coeff=0.001
  Optim:    LR=5e-7, grad_clip=1.0, batch=32, mini_batch=128
  Eval:     test_freq=10, greedy decoding, rule-based scoring
```
