# Narrative datasets for ToM (Q3b)

**Scope.** This document records the narrative (non-conversational) behavior-prediction corpora
built for the **Q3b** experiments (`project_planning/BeRL_paper_plan.md → Q3b`), the candidate
screenplay sources we evaluated, **why** we chose the ones we did, and the exact build recipe so
the parquets can be reproduced.

**Q3b question.** Does the BeRL behavior-prediction reward (`log P(next human production | context + CoT)`)
still induce ToM transfer when the training signal is **narrative continuation** — predict the real
next sentence/line of a human-written story or screenplay — instead of **two-party dialogue**? The
dialogue anchor is B1 (`dcfg_smoke_mix`, d_avg **+0.0215**). A positive result on narration broadens
the claim to "predict the human's next production" in general; a null bounds it to interactive text.

All narrative corpora are label-free, built on the shared converter base
`scripts/narrative_converter_base.py` via `build_dataset.py`, and scored with the **identical
behavior (ll_loss / power) reward** as the ST01 anchor — only the *training corpus* changes; evals
are the identical ToM `subsample300` suite.

---

## 2. Source background — what each dataset is

| Source | HF id | What it is | Style of text | License / origin |
| --- | --- | --- | --- | --- |
| **ROCStories** | [`shawon/rocstories-combined`](https://huggingface.co/datasets/shawon/rocstories-combined) | ~98k crowd-authored **5-sentence commonsense stories**, each with a title. Designed for the Story Cloze Task (commonsense causal/temporal reasoning about everyday events). | Short, clean, third-person everyday prose; strong causal/temporal structure. | Mostofa Rooshan mirror of Mostafazadeh et al. (2016) ROCStories. |
| **WritingPrompts** | [`vkpriya/GPT-WritingPrompts`](https://huggingface.co/datasets/vkpriya/GPT-WritingPrompts) (`human_wp_stories.json`) | Human-written short stories from Reddit **r/WritingPrompts**, each responding to a `[WP]` prompt. We use the **human** stories (the repo also ships GPT-generated counterparts for bias studies). | Long-form, creative, stylistically varied first/third-person fiction; informal (Reddit) tokenization, occasional profanity. | Fan & al. (2018) WritingPrompts, re-packaged by Vishakh Padmakumar et al. |
| **TinyStories** | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | ~2M **synthetic** very-short children's stories (GPT-3.5/4 generated) using only vocabulary a 3–4 year-old would understand. Built to study emergence of coherence in small LMs. | Very simple, short, third-person; small vocabulary; explicit named characters (e.g. "Timmy"). | Eldan & Li (2023), Microsoft. |
| **MovieSum** | [`rohitsaxena/MovieSum`](https://huggingface.co/datasets/rohitsaxena/MovieSum) | 2,200 **movie screenplays** (1800/200/200 split) paired with Wikipedia plot summaries, released for abstractive screenplay summarization. We use only the `script` field. | Structured screenplay text with an XML tag vocabulary (`<scene>`, `<stage_direction>`, `<scene_description>`, `<character>`, `<dialogue>`, `<parenthetical>`); **named speakers**. | Saxena & Keller (2024), "Select and Summarize". |

**What a training example looks like (from the built parquets):**

- **ROCStories** — `Title: Fixing a Computer Problem` / *"Abby was having trouble with her laptop…
  She tried everything…"* → gold next line: *"Finally she decided to restart it one last time."*
- **WritingPrompts** — `Writing prompt: At age 15 you told the gf…` / story so far → gold next line
  (raw Reddit tokenization, e.g. backtick quotes ``` `` ```).
- **TinyStories** — *"Once upon a time, there was a kind boy named Timmy… His bed, dresser, and desk
  had all disappeared!"* → gold next line: *"Timmy looked all around his room, but he couldn't find
  his furniture anywhere."*
- **MovieSum** — `Scene: EXT. LANI'S HOUSE… SCOTTIE: Her house is down there… MATT: You've been here
  before?` → predict what **SCOTTIE** says next: *"Yea - uh. She invites me to her birthday…"*

**Why this mix.** ROCStories + TinyStories give clean, well-structured everyday narration (strong
event/causal signal, small vocabulary); WritingPrompts adds long-form, stylistically diverse human
fiction. Together they form `narrative_mix` — the pure-narration lower bound. MovieSum is kept
separate as the screenplay arm because it uniquely has **named speakers**, letting us predict a
*named agent's* next line (the closest narrative analog to the dialogue anchor).

---

## 3. Interpretation ladder (why two arms)

Narrative sources differ in how much they resemble the dialogue anchor. We picked two points on a
"distance-from-dialogue" ladder to bracket the effect:

| Arm | Corpus | What is predicted | Named agents? | Position on ladder |
| --- | --- | --- | --- | --- |
| **Q3b-N1** | `dcfg_narrative_mix` | real next **sentence** of a prose story | ✗ (third-person narration) | **lower bound** — furthest from dialogue |
| **Q3b-M1** | `dcfg_moviesum` | real next **character line** in a scene | ✓ (character names as speakers) | **primary** — closest narrative analog to dialogue |

**Expected d_avg ordering if target ToM-dependence drives transfer:**
`dialogue (B1) ≥ M1 (screenplay) ≥ N1 (narration)`.

**Observed (raw d_avg, step0→final, N=1 exploratory screen):**
B1 **+0.0215** > N1 **+0.0185** > M1 **+0.0131**. Both narrative arms are positive (ToM lift survives
dropping dialogue structure), but N1 (pure narration) slightly *beat* M1 (screenplay) — the opposite
of the expected ladder. These are raw d_avg (N=1, possibly format-confounded); an honest
`d_cavg` / `d_cond_acc` reassessment is pending before drawing conclusions.

WandB: N1 = `jkhouja-oxford/TOM_EXP/runs/dbxjlwhh`; M1 = `jkhouja-oxford/TOM_EXP/runs/q3bm1moviesum`.

---

## 4. Screenplay source selection (M1)

We evaluated four candidate screenplay/movie corpora for the M1 (character-line prediction) arm.
The goal was a source with **explicit named speakers** (so each target is a *named agent's*
utterance — a genuine intent/ToM signal at narrative scale) that also parses cleanly at scale.

| Source | Approx. size | Character info | Parsing | Verdict |
| --- | --- | --- | --- | --- |
| **MovieSum** (`rohitsaxena/MovieSum`) | 2,200 screenplays (1800/200/200) | **Explicit** `<character>`/`<dialogue>` tag pairs; named speakers per line | Clean XML tag vocabulary (`<scene>` / `<stage_direction>` / `<scene_description>` / `<character>` / `<dialogue>` / `<parenthetical>`); split on `<scene>`, extract ordered character/dialogue pairs | ✅ **Chosen** |
| Movie Script DB | ~1k+ raw scripts | Present but **only as free-text ALL-CAPS cue lines** | Requires brittle heuristic regex (indentation/caps) to recover speaker turns; noisy | ✗ rejected — parsing fragility |
| ScriptBase | ~1k films | Present, aligned to summaries | Screenplay text is loosely structured; speaker attribution needs the same heuristic recovery as raw scripts | ✗ rejected — no clean speaker tags |
| MENSA (movie-dialogue / speaker-attribution corpora) | dialogue-level | Speaker labels present | Pre-segmented into dialogue only — **loses scene grounding** (stage directions / scene description) we use as the premise | ✗ rejected — no scene-level premise |

**Why MovieSum won.** It is the only candidate that ships an **explicit, machine-readable tag
vocabulary** with named speakers, so we get genuine multi-party exchanges *and* a scene premise
(setting grounding) without heuristic speaker recovery. Screenplays are the highest-ToM-value
narrative source — the closest narrative-scale analog to the dialogue anchor — because predicting a
*named* character's next line is a direct character-intent signal. The other three would have
required brittle regex speaker attribution (Movie Script DB / ScriptBase) or discard scene grounding
(MENSA).

---

## 5. Corpus 1 — `dcfg_narrative_mix` (N1, lower bound)

**6,000 rows**, three English prose narrative sources (predict the next prose sentence, no named
agents). Config: `scripts/configs/dcfg_narrative_mix.yaml` (`prompt_style: narrative`).

| Source | Rows | Converter | `premise_label` | Notable filters |
| --- | --- | --- | --- | --- |
| ROCStories | 2,500 | `scripts/convert_rocstories.py` | `Title` | `min_turns 3`, `min_context_sentences 2`, `min_response_words 3`, `max_response_words 60` |
| WritingPrompts | 2,000 | `scripts/convert_writingprompts.py` | `Writing prompt` | + `max_context_sentences 12`, `min_response_words 4` |
| TinyStories | 1,500 | `scripts/convert_tinystories.py` | (none) | + `max_context_sentences 8` |

Row counts in the built parquet confirmed: `{rocstories: 2500, writingprompts: 2000, tinystories: 1500}`.

---

## 6. Corpus 2 — `dcfg_moviesum` (M1, primary)

**3,000 rows** sampled from **101 unique films** (scene-level, named speakers). Config:
`scripts/configs/dcfg_moviesum.yaml` (`prompt_style: script`); converter `scripts/convert_moviesum.py`.

**How examples are built:**
1. Load `rohitsaxena/MovieSum`; **shuffle movie order** (seeded) to avoid alphabetical bias, then
   collect scenes up to `scene_cap = max(sample_size×2, 4000)` (6,000 scenes collected here).
2. **Split each script on `<scene>` boundaries** → many bounded exchanges per film (scene-coherent
   context, no 900-turn giant sequences).
3. Within a scene, extract ordered `<character>`/`<dialogue>` pairs → `SPEAKER: line` turns.
4. **Premise** = scene heading (`<stage_direction>`) + optional first `<scene_description>`
   (`include_scene_description: true`), capped at `max_premise_words 60` — used as setting grounding.
5. `NarrativeConverterBase` emits one target per character line from `min_context_sentences` on,
   with the standard row schema + behavior reward path.

**Filtering criteria (from the config + base converter):**
- `min_turns 3` and a scene must have `≥ max(min_turns, min_context_sentences+1)` lines to qualify.
- `min_context_sentences 2` … `max_context_sentences 12` (sliding context window).
- `min_response_words 3` … `max_response_words 60` (targets outside this band are skipped).

**Build stats (from `moviesum_build.log`):** 6,000 scenes → **61,323** candidate training examples →
**3,000** sampled (seeded). Sampled parquet spans **101 unique films** (e.g. *Juno_2007*,
*Sideways_2004*, *Heathers_1988*, *12 Monkeys_1995*).

---

## 7. Reproduce

```bash
# Build the parquets (generate once, shared across rows)
python build_dataset.py --config scripts/configs/dcfg_narrative_mix.yaml   # → data/dcfg_narrative_mix.parquet (6000)
python build_dataset.py --config scripts/configs/dcfg_moviesum.yaml        # → data/dcfg_moviesum.parquet   (3000)
```

Launch (exact ST01/B1 anchor recipe — actor-RM power k4 ll_min−6, kl0.05, lr5e-7, batch32, n16,
ec0, max_resp512, Qwen2.5-3B, `VAL_SUITE=subsample300`):

```bash
EXP_ID=Q3b-moviesum-behavior DATA_NAME=dcfg_moviesum DATA_TRAIN=data/dcfg_moviesum.parquet \
  REWARD_TYPE=power POWER_K=4 POWER_LL_MIN=-6 KL=0.05 LR=5e-7 TRAIN_BATCH=32 MAX_RESP=512 \
  ENTROPY_COEFF=0.0 VAL_SUITE=subsample300 bash experiments/train_behavior_qwen2.5.sh
```

Per-run reproducibility record: `experiments_logs/s2-Q3b-moviesum-behavior-dcfg_moviesum.md`.

---

## 8. Files

| Kind | Path |
| --- | --- |
| Shared converter base | `scripts/narrative_converter_base.py` |
| Prose converters | `scripts/convert_{rocstories,writingprompts,tinystories}.py` |
| Screenplay converter | `scripts/convert_moviesum.py` |
| Configs | `scripts/configs/dcfg_{narrative_mix,moviesum,rocstories,writingprompts,tinystories}.yaml` |
| Built parquets | `data/dcfg_narrative_mix.parquet` (6k), `data/dcfg_moviesum.parquet` (3k) |
| Experiment design | `project_planning/BeRL_paper_plan.md → Q3b` |
| Tracker rows | `project_planning/BeRL_experiments_tracker.md` (Q3b-N1, Q3b-M1) |
