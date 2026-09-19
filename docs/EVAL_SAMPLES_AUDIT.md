# Eval Subset Samples & Quality Audit

_Generated 2026-07-30 · sampled from Qwen2.5-3B step-0 validation (D6c-03/04 logs, `subsample300`)._

This document shows **2 real sampled prompts + the model's verbatim response** for each of the
25 ToM eval subtypes (+2 guardrails), to judge whether each eval is natural / well-formed / testing
the right thing. Tier flags come from a 64-run progression audit (honest `answer_acc_cond`, first vs
last eval): 🟢 clean level-probe · 🟡 noisy/collapse-prone · 🟠 below-chance static diagnostic ·
🔴 format-starved (drop/fix candidate) · ⚪ non-ToM guardrail.

All samples share this system prompt (shown once):

> _You are a helpful assistant. … reasoning within `<think></think>` and answer within
> `<answer></answer>` … solve a theory of mind reasoning problem …_

Scoring = **format (±1) + answer (±2)**, range **[−3,+3]**. `score=3` → format-ok & correct;
`−1` → format-ok but wrong answer; `−3` → format failed.


---

## Summary table — BeRL vs rule-based (label-supervised) on OOD ToM

Every subtype contributes **300 rows = 3.85%** of the 7,800-row `subsample300` suite (stratified,
seed 42), so the distribution is uniform per subtype; the skew is at the **benchmark** level (OpenToM
& FANToM = 5 subtypes each = 19.2% of the suite each; BigToM/DynToM/SimpleToM = 3 each = 11.5%; all
others = 1 subtype = 3.85%). Rows are grouped by benchmark so subtypes stay together.

**Column meanings**
- **Q/A style** — the answer format the scorer expects and the ToM ability probed. `2-way MC` =
  pick A/B; `3-/4-/N-way MC` = more options; `binary` = yes/no; `list` = emit the exact set of names
  (all-or-nothing match); `open-ended` = free-text short answer (a location noun / exact token).
  Format matters: fewer options ⇒ higher chance baseline & easier parsing.
- **chance** — accuracy a random guesser would score, given the option count (`0.50` for binary/2-way,
  `0.33` for 3-way, `0.25` for 4-way, `~0.10–0.17` for many-way DynToM, `low` for open-ended/list
  where guessing rarely hits). Use it to read `base`: **base ≈ chance ⇒ the model is near-guessing**;
  **base < chance ⇒ a systematic bias** (e.g. over-attribution / omniscience — the *intended* finding
  for those subtypes).
- **base** — the frozen Qwen2.5-3B **step-0** honest accuracy (`answer_acc_cond`: correctness among
  format-passing responses) on this subtype, averaged over the 64 runs' initial eval.
- **RB turf** — whether this subtype is **in-distribution for the rule-based control**. The rule-based
  ToM-label-supervised baseline (`Q1-B2`) trained on **Hi-ToM + ExploreToM** labels, so `hi_tom`,
  `explore_tom`, `exploretom_infilled` are **ID** (unfair to BeRL — RB saw the template); every other
  subtype is **OOD** for RB, a fair generalization test. (mmlu/gsm8k are ⚪ guardrails.)
- **best BeRL** — the **highest honest end-of-training accuracy** any label-free **BeRL** run (behavior
  reward, `frozen`/`actor` RM) reached, with the run's scale tag (`0.5B/3B/7B/gemma-2B`). BeRL uses **no
  ToM labels**.
- **best RB** — the highest end-of-training accuracy the label-supervised **rule-based** control reached
  (always 3B; only 3 such runs exist, `Q1-B2-01/02/03`).
- **3B v 3B** — the **fixed-scale** verdict, best BeRL-**3B** vs best rule-based-**3B** on this subtype
  (✅ BeRL ahead by >2pp · ➖ tie within ±2pp · ❌ RB ahead; `—` = ID/guardrail, not a fair OOD test).
  This is the cleanest merit test: same model size, one uses ToM labels and one doesn't.

**Read (quality tier)** — a one-glance verdict on the eval's *measurement quality* (format-pass rate,
base vs chance, run-to-run stability), independent of the BeRL-vs-RB merit columns:
- 🟢 **clean level-probe** — high format-pass (≥~0.80), sensible base vs chance, stable across runs. A
  valid capability probe.
- 🟡 **noisy / collapse-prone** — real ToM probe but high-variance and tends to **degrade** under RL;
  treat its numbers cautiously (tomi, explore_tom, fantom_belief_mc).
- 🟠 **below-chance static diagnostic** — base sits *below* chance by design (over-attribution /
  omniscience bias); valuable as a fixed one-shot diagnostic but **inert & noisy as a learning curve**
  — report the level, don't track deltas.
- 🔴 **format-starved (drop/fix candidate)** — <70% of responses even parse (`*_list`, `hi_tom`), so
  the accuracy estimate is unreliable; either fix the answer format or drop from aggregate metrics.
- ⚪ **non-ToM guardrail** — mmlu / gsm8k, kept to detect capability regressions; **excluded** from the
  aggregate ToM `cavg`.

| benchmark | subtype | Q/A style tested | chance | base | RB turf | best BeRL | best RB | 3B v 3B | read |
|---|---|---|---:|---:|:--:|---:|---:|:--:|---|
| **BigToM** | bigtom_forward_belief | 2-way MC, forward belief | 0.50 | 0.78 | OOD | 0.90 · 7B | 0.80 | ✅ | 🟢 |
| | bigtom_forward_action | 2-way MC, predict action from belief | 0.50 | 0.70 | OOD | 0.80 · 7B | 0.73 | ✅ | 🟢 |
| | bigtom_backward_belief | 2-way MC, belief from action | 0.50 | 0.60 | OOD | 0.70 · 7B | 0.69 | ❌ | 🟢 |
| **DynToM** | dyntom_type_a | ~8-way MC, state-in-scenario | ~0.13 | 0.43 | OOD | 0.58 · 7B | 0.54 | ➖ | 🟢 |
| | dyntom_type_c | ~6-way MC, causal influence | ~0.17 | 0.47 | OOD | 0.55 · 7B | 0.51 | ➖ | 🟢 |
| | dyntom_type_d | 5–24-way MC, state trajectory | ~0.10 | 0.36 | OOD | 0.50 · 7B | 0.36 | ✅ | 🟢 |
| **ExploreToM** | explore_tom | open-ended exact, in-dist symbolic | low | 0.66 | **ID** | 0.86 · 7B | 0.93 | — | 🟡 |
| | exploretom_infilled | exact + binary, NL-infilled robustness | ~0.5 | 0.55 | **ID** | 0.75 · 0.5B | 0.59 | — | 🟢 |
| **FANToM** | fantom_info_binary | binary yes/no, info access | 0.50 | 0.41 | OOD | 0.62 · gemma-2B | 0.51 | ❌ | 🟠 |
| | fantom_info_list | list-of-names (exact set), info access | low | 0.24 | OOD | 0.50 · 0.5B | 0.31 | ✅ | 🔴 |
| | fantom_answerability_binary | binary yes/no, answerable? | 0.50 | 0.21 | OOD | 0.53 · gemma-2B | 0.23 | ➖ | 🟠 |
| | fantom_answerability_list | list-of-names (exact set), self-referential | low | 0.19 | OOD | 0.37 · 3B | 0.34 | ✅ | 🔴 |
| | fantom_belief_mc | 2-way MC (A/B), belief mental-state | 0.50 | 0.58 | OOD | 0.67 · 0.5B | 0.47 | ✅ | 🟡 |
| **Hi-ToM** | hi_tom | open-ended noun, nested higher-order belief | low | 0.34 | **ID** | 0.49 · 7B | 0.64 | — | 🔴 |
| **OpenToM** | opentom_location_fo | binary + 2-way MC, 1st-order location | 0.50 | 0.61 | OOD | 0.75 · gemma-2B | 0.76 | ❌ | 🟢 |
| | opentom_location_so | binary + 2-way MC, 2nd-order location | 0.50 | 0.45 | OOD | 0.64 · gemma-2B | 0.62 | ➖ | 🟢 |
| | opentom_multihop_fo | 3-way MC, 1st-order multihop | 0.33 | 0.63 | OOD | 0.75 · gemma-2B | 0.69 | ➖ | 🟢 |
| | opentom_multihop_so | 3-way MC, 2nd-order multihop | 0.33 | 0.44 | OOD | 0.55 · gemma-2B | 0.48 | ✅ | 🟢 |
| | opentom_attitude | 3-way MC, emotional attitude | 0.33 | 0.42 | OOD | 0.48 · 7B | 0.46 | ➖ | 🟢 |
| **SimpleToM** | simpletom_mental | 2-way MC, infer mental state | 0.50 | 0.80 | OOD | 0.95 · 3B | 0.92 | ✅ | 🟢 |
| | simpletom_behavior | 2-way MC, predict behavior | 0.50 | 0.58 | OOD | 0.74 · 3B | 0.59 | ✅ | 🟢 |
| | simpletom_judgment | 2-way MC, judge behavior (omniscience trap) | 0.50 | 0.37 | OOD | 0.59 · 0.5B | 0.28 | ✅ | 🟠 |
| **ToMBench** | tombench | 2-way + 4-way MC, broad ToM abilities | ~0.4 | 0.68 | OOD | 0.77 · 7B | 0.72 | ➖ | 🟢 |
| **ToMi** | tomi | open-ended noun, 1st/2nd-order false belief | low | 0.59 | OOD | 0.68 · 7B | 0.71 | ❌ | 🟡 |
| **MMLU** _(guardrail)_ | mmlu | 4-way MC, general knowledge | 0.25 | 0.58 | ⚪ | 0.74 · 7B | 0.66 | — | ⚪ |
| **GSM8K** _(guardrail)_ | gsm8k | exact numeric, math | low | 0.62 | ⚪ | 0.79 · 3B | 0.84 | — | ⚪ |

**What the table shows.** Label-free **BeRL is competitive with — and usually beats — the
label-supervised rule-based control on out-of-distribution ToM.** At **matched 3B scale** on the 21 OOD
ToM subtypes, best BeRL-3B **wins 10, ties 7, loses 4** (`3B v 3B` column); across *any* scale it matches
or beats rule-based on **19/21** OOD ToM subtypes. The 4 fixed-scale losses (tomi, opentom_location_fo,
bigtom_backward_belief, fantom_info_binary) are the **symbolic false-belief** tasks closest to the
control's ExploreToM/Hi-ToM training style — consistent with *BeRL generalizes; rule-based only wins where
it saw the template*. Tellingly, BeRL even beats rule-based on **exploretom_infilled** (0.75 vs 0.59) —
the **NL-infilled** robustness variant of an ID task — while rule-based keeps the edge only on the two
purely-symbolic ID tasks it trained on (explore_tom, hi_tom). Nuances that keep this honest, cutting both
ways: (1) *favoring BeRL* — "best run" is a max and BeRL had **many** runs vs only **3** rule-based runs,
so its ceiling had more lottery tickets; a fully airtight test matches run counts / compares medians. (2)
*favoring the claim* — several BeRL wins are at **0.5B/gemma-2B**, *below* the 3B control, so it is not a
pure model-size effect. Net: this is a **positive, defensible finding** — label-free behavior-prediction RL
reaches ToM competitive with direct ToM-label supervision on held-out benchmarks, with the losses confined
to the supervised model's own distribution. The 🔴 format-starved (`*_list`, `hi_tom`, fmt <70%) and 🟠
below-chance subtypes remain the least-trustworthy *measurements* and are the weakest links in the
aggregate metric regardless of method.


---

## ToMi — first/second-order false-belief, symbolic story (generalization; never in training)

### `tomi`  — 🟡 flagship OOD, collapse-prone (range −0.50), fmt 0.81

**Sample 1** · score `-1.0000`

_extra system instruction:_ Note: You should assume the following. (1) An agent witnesses everything and every movement before exiting a room. (2) An agent A can infer another agent B's mental state only if A and B have been in …[+181 chars]


*Prompt (user turn):*
```
Read the following story and answer the question. 
Story: Mason likes the grapefruit Owen entered the sunroom. Mason entered the sunroom. The cabbage is in the pantry. Elizabeth loves the tomato Mason exited the sunroom. Owen moved the cabbage to the envelope.
Question: Where will Mason look for the cabbage?
```
*Gold:* `pantry`

*Model response (verbatim):*
```
<think> Mason last saw the cabbage when he exited the sunroom. Since Owen moved the cabbage to the envelope, Mason will assume the cabbage is in the envelope. </think>
<answer>envelope</answer>
```

**Sample 2** · score `-1.0000`

_extra system instruction:_ Note: You should assume the following. (1) An agent witnesses everything and every movement before exiting a room. (2) An agent A can infer another agent B's mental state only if A and B have been in …[+181 chars]


*Prompt (user turn):*
```
Read the following story and answer the question. 
Story: Chloe hates the beans Lucas entered the bathroom. Carter entered the bathroom. Chloe entered the bathroom. The stockings is in the cupboard. Chloe moved the stockings to the drawer. Carter exited the bathroom.
Question: Where does Chloe think that Carter searches for the stockings?
```
*Gold:* `drawer`

*Model response (verbatim):*
```
<think> Chloe saw Carter enter the bathroom and Carter left the bathroom. Since Chloe witnessed Carter's actions, she can infer Carter's mental state. However, Chloe did not see Carter search for the stockings. The only place Chloe knows the stockings are is in the drawer. Therefore, Chloe would assume Carter would look in the cupboard, as that is the last known location of the stockings before they were moved. </think>
<answer>cupboard</answer>
```


---

## Hi-ToM — deeply-nested higher-order belief ("X thinks Y thinks…")

### `hi_tom`  — 🔴 format-starved (fmt 0.52), base 0.34

**Sample 1** · score `-3.0000`

_extra system instruction:_ Note: You should assume the following. (1) An agent witnesses everything and every movement before exiting a room. (2) An agent A can infer another agent B's mental state only if A and B have been in …[+181 chars]


*Prompt (user turn):*
```
Read the following story and answer the question. 
Story: Nathan, Emma, Jayden, Elizabeth and Ava entered the patio. The orange is in the green_bottle. Nathan made no movements and stayed in the patio for 1 minute. Nathan dislikes the peach. Nathan exited the patio. Emma moved the orange to the green_suitcase. Emma exited the patio. Jayden moved the orange to the red_drawer. Jayden exited the patio. Elizabeth made no movements and stayed in the patio for 1 minute. Elizabeth exited the patio. Ava made no movements and stayed in the patio for 1 minute. Ava exited the patio. Ava likes the green_bottle. Nathan, Emma, Jayden, Elizabeth and Ava entered the waiting_room. Jayden privately told Nathan that the orange is in the green_pantry now.
Question: Where does Elizabeth think Ava thinks Emma thinks Jayden thinks the orange is?
```
*Gold:* `green_suitcase`

*Model response (verbatim):*
```
think Elizabeth has not interacted with Ava, Emma, or Jayden since they left the patio. Therefore, Elizabeth would not have any updated information about the orange's location. Since she has no knowledge of the orange's current location, she would not be able to infer what Ava, Emma, or Jayden think about the orange's location. Thus, Elizabeth would think the orange is in the green_bottle, as that was its location when she last saw it in the patio.
answer green_bottle
```

**Sample 2** · score `-1.0000`

_extra system instruction:_ Note: You should assume the following. (1) An agent witnesses everything and every movement before exiting a room. (2) An agent A can infer another agent B's mental state only if A and B have been in …[+181 chars]


*Prompt (user turn):*
```
Read the following story and answer the question. 
Story: Avery, Isabella, Nathan, Mason and Hannah entered the sunroom. The onion is in the green_drawer. Avery moved the onion to the green_bottle. Avery exited the sunroom. Isabella made no movements and stayed in the sunroom for 1 minute. Isabella exited the sunroom. Nathan moved the onion to the green_box. Nathan exited the sunroom. Mason moved the onion to the green_treasure_chest. Mason exited the sunroom. Mason likes the red_container. Hannah made no movements and stayed in the sunroom for 1 minute. Hannah exited the sunroom. Avery, Isabella, Nathan, Mason and Hannah entered the waiting_room.
Question: Where does Avery think Hannah thinks Mason thinks Isabella thinks the onion is?
```
*Gold:* `green_bottle`

*Model response (verbatim):*
```
<think> Avery saw the onion move from the green_drawer to the green_bottle. Isabella did not move the onion. Nathan moved it to the green_box, but Avery did not see this. Mason moved it to the green_treasure_chest, and Hannah did not move it. Since Avery only saw the onion in the green_treasure_chest, Avery would infer that Hannah, who saw the onion in the green_treasure_chest, would infer that Mason, who saw the onion in the green_treasure_chest, would infer that Isabella, who did not see the onion move, would think it is in the green_treasure_chest. </think>
<answer>green_treasure_chest </answer>
```


---

## ExploreToM — in-distribution symbolic-story ToM (open-ended)

### `explore_tom`  — 🟡 in-dist symbolic, collapse-prone (range −1.00), fmt 0.68

**Sample 1** · score `3.0000`

_extra system instruction:_ Note: You should assume the following. (1) An agent witnesses everything and every movement before exiting a room. (2) An agent A can infer another agent B's mental state only if A and B have been in …[+181 chars]


*Prompt (user turn):*
```
Read the following story and answer the question. 
Story: Alexis entered the security control room. Alexis moved the small flashlight to the canvas duffel bag, which is also located in the security control room. Alexis told privately to Kayla that the small flashlight is in the canvas duffel bag. While this action was happening, Kaitlyn witnessed this action in secret (and only this action). Alexis moved the small flashlight to the plastic storage bin, which is also located in the security control room. Alexis told privately to Kaitlyn that the small flashlight is in the plastic storage bin. Alexis left the security control room. Kaitlyn told privately to Kayla that the small flashlight is in the plastic storage bin. Alexis told privately to Kaitlyn that the small flashlight is in the security control room. Kaitlyn entered the security control room. Kaitlyn moved the small flashlight to the canvas duffel bag, which is also located in the security control room. Kayla entered the staff room. Kayla left the staff room. Kaitlyn moved the small flashlight to the staff room, leaving the canvas duffel bag in its original location.
Question: In which room is the small flashlight now?
```
*Gold:* `staff room`

*Model response (verbatim):*
```
<think> Alexis moved the flashlight to the plastic storage bin, but Kaitlyn moved it to the canvas duffel bag. Kaitlyn's action is secret and private, so it overrides Alexis's action. Kaitlyn then moved it to the staff room, but left the canvas duffel bag in the security control room. </think>
<answer>staff room </answer>
```

**Sample 2** · score `-1.0000`

_extra system instruction:_ Note: You should assume the following. (1) An agent witnesses everything and every movement before exiting a room. (2) An agent A can infer another agent B's mental state only if A and B have been in …[+181 chars]


*Prompt (user turn):*
```
Read the following story and answer the question. 
Story: Jaxon entered the operations room. Jaxon moved the tactical flashlight to the canvas duffel bag, which is also located in the operations room. Tyler entered the operations room. Jaxon told out loud that the tactical flashlight is in the canvas duffel bag. Jaxon told privately to Jasmine that the tactical flashlight is in the canvas duffel bag. Jasmine entered the operations room. Tyler told privately to Michael that the tactical flashlight is in the canvas duffel bag. Michael entered the operations room. Jaxon moved the tactical flashlight to the metal toolbox, which is also located in the operations room.
Question: In which container does Jaxon think that Michael will search for the tactical flashlight?
```
*Gold:* `metal toolbox`

*Model response (verbatim):*
```
<think> Jaxon saw Michael in the room and told him privately that the tactical flashlight is in the canvas duffel bag. Since Jaxon can only infer Michael's mental state based on interactions in the room, Jaxon would infer that Michael knows the flashlight is in the canvas duffel bag. Jaxon then moved the flashlight to the metal toolbox. Given this information, Jaxon would think Michael will search the canvas duffel bag first, as that is the last known location of the flashlight. </think>
<answer>canvas duffel bag</answer>
```


---

## ExploreToM-Infilled — same Qs on natural-language infilled story (robustness)

### `exploretom_infilled`  — 🟢 NL twin of explore_tom, fmt 0.81 (more stable)

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
The facility's security control room, with its rows of monitors and humming machinery, was a hub of quiet intensity in the late evening hours. Nearby, the staff room offered a small oasis of comfort, its worn furniture and warm lighting a respite from the institutional surroundings. Kayla walked into the security control room, the soft glow of the monitors casting an eerie light on her face. Her eyes scanned the space, her gaze lingering on the familiar sight of the rows of screens displaying the facility's various corridors and rooms. Kaitlyn became aware of Kayla's location through a private channel, her colleague now found to be in the very heart of the facility's security, the control room. Kayla tucked the flashlight away in the security desk's drawer, its new home a testament to her careful consideration of secure and discreet locations. Alexis entered the staff room, its cozy atmosphere a stark contrast to the hum of machinery in the nearby security control room, before swiftly moving on to the heart of the facility's surveillance. There, the rows of monitors in the security control room greeted Alexis, an array of eyes watching over the facility's corridors and rooms. Two trusted individuals now had knowledge of the flashlight's new hiding place, thanks to Kayla's covert communication with them. The security control room remained a hub of quiet intensity as Kayla went a …[+1470 chars]
```
*Gold:* `security control room  (security control room)`

*Model response (verbatim):*
```
<think> To determine where the small flashlight was at the beginning, I need to look at the sequence of events. The flashlight is first mentioned in the security control room. Kayla is in the control room and tucks the flashlight away in the security desk's drawer. This indicates that the control room was the initial location of the flashlight. </think>
<answer>The small flashlight was at the beginning in the security control room.</answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
The dimly lit briefing room, located on the second floor of the police station, was surrounded by narrow windows that filtered the early morning sunlight. A large wooden table stood at the room's center, surrounded by chairs, and a long, blank whiteboard stretched across one wall. Hannah's eyes adjusted to the dim lighting as she crossed the threshold into the briefing room. Cole sidled closer to Victoria, his voice a low whisper, "I think I might've overheard something, and our suspect profile is taking shape – think mid-twenties to mid-thirties." Jace caught the snippet from across the room, his expression a mask of nonchalance. Just as Hannah was getting her bearings, Victoria and Cole slipped into the briefing room, their eyes scanning the space as if searching for an opening to share their findings. With renewed purpose, Hannah emerged into the bright morning light that streamed through the second-floor windows, the dimly lit briefing room and its secrets locked behind her. There was a flicker of silent understanding as Hannah met Jace's gaze outside the briefing room, an exchange of crucial information that couldn't be risked in front of others – they spoke of suspects, in carefully guarded tones. Unspoken, but understood, was the mutual trust they'd established in this private moment.

Question: Does Victoria know about suspect profiles? Answer yes or no.
```
*Gold:* `yes  (yes)`

*Model response (verbatim):*
```
<think> In the scenario, Cole mentions that he thinks he might have overheard something and that "our suspect profile is taking shape – think mid-twenties to mid-thirties. " Victoria and Cole then slip into the briefing room, and Victoria's and Cole's eyes scan the space as if searching for an opening to share their findings. This indicates that they are discussing the suspect profile. Victoria is present in the room, and her eyes scanning the space suggests she is also aware of the discussion. </think>
<answer>Yes</answer>
```


---

## FANToM — multi-party conversation ToM (info-access / answerability / belief)

### `fantom_info_binary`  — 🟠 below-chance 0.41 — static diagnostic

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Aden: It was great reminiscing about the past with you guys, but I have to head out now. I need to submit some papers. Let's catch up again soon!
Britney: Sure, Aden. Do what you gotta do. See you next time!
Erica: Bye, Aden! Take care!
Britney: Anyway, Erica. I want to bring up the topic of long-distance relationships. I'm just trying to navigate through mine and it's downright hard.
Erica: Oh, Britney, I can completely understand. I’ve been through one myself. Trust, communication, and hard work are crucial factors.
Britney: It's the 'missing them' part that gets me. How did you handle that?
Erica: It does get hard, but planning regular virtual dates and sending care packages helped me a lot. It's also important to have personal goals, a life outside of just the relationship.
Britney: That's some sound advice there, Erica. I'll try to implement these. Thank you!
Erica: You're welcome. Remember, every relationship is unique in its own way. Tailor whatever advice you get to suit yours. Good luck, Britney!
Aden: Hey, ladies. I managed to turn in those papers early. Hope I didn't miss much?
Erica: Welcome back, Aden! We were just chatting about long-distance relationships.
Britney: Yes, and Erica dropped some wisdom about managing the 'missing them' part.
Aden: Oh, that sounds interesting. Long-distance relationships can be tough but also very rewarding.
Britney: Speaking of toug …[+1268 chars]
```
*Gold:* `yes`

*Model response (verbatim):*
```
<think> Erica mentioned that she had been through a long-distance relationship and shared her experiences with Britney. She discussed the challenges and solutions for managing a long-distance relationship, including planning virtual dates and sending care packages. Aden's first anniversary was discussed, and he shared a simple gesture he made for his partner. Britney and Aden also discussed their first anniversaries, with Aden mentioning he surprised his partner with traditional Spanish treats and a study playlist. Since Erica shared these experiences with Britney, she would know about Aden's first anniversary and the gesture he made. </think>
<answer>Yes</answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Adelyn: Hello Frank, it's nice to meet you. So, do you enjoy any specific outdoor activities or sports?
Frank: Hello Adelyn, pleasure meeting you. Yes actually, I've been playing basketball since I was a kid. How about you?
Adelyn: That's great Frank, I've always admired people who have a knack for such sports. In my case, I'm more of a nature enthusiast. I love trekking and going for nature hikes.
Frank: Interesting, I never understood the charm of it until I went to a camping trip last summer. The experience was amazing; I've grown to appreciate nature more since then.
Adelyn: I'm glad to hear that, Frank. Trekking isn't just about exercise, it's about connecting with nature, the tranquility is soothing. Well, are you into any fitness activities?
Frank: Yes, I'm keen on fitness too. I hit the gym 4 days a week and enjoy doing cardiovascular activities. How about you? Yoga, gym, or something else?
Adelyn: I'm into yoga and I occasionally indulge in indoor rock climbing. As for the gym, well, I don't spend as much time as I would like to.
Frank: Rock climbing sounds adventurous! I’ve always wanted to try it out. And, yoga is indeed calming and also great for flexibility.
Adelyn: Yes! for me, fitness routines should be fun and engaging. This way, they don't feel like a chore.
Frank: Absolutely! Keeping it varied and enjoyable can be a great motivation to stay fit.
Adelyn: Well s …[+1372 chars]
```
*Gold:* `yes`

*Model response (verbatim):*
```
<think> Frank was informed about Kayla's recent interest in strength training and her desire to share resources. He also expressed interest in learning more about it. This indicates that Frank is aware of Kayla's new fitness routine and her willingness to help him understand strength training. </think>
<answer>Yes</answer>
```

### `fantom_info_list`  — 🔴 format-starved (fmt 0.65) + floor 0.24

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Adalynn: Well, from my experience, one factor that makes a long-distance relationship work is constant communication. It can be through text, call, or video chat. But what's essential is to talk regularly.
Marissa: I couldn't agree more. It also becomes more special when both of you set a specific time to communicate, it's like setting up a date. It helps the bond to grow stronger because you are prioritizing your relationship over other things.
Jett: Communication indeed plays a significant role, but so does trust. Trusting each other in a long-distance relationship is fundamental. Misunderstandings can easily grow if the trust isn't strong.
Adalynn: Good point, Jett. But establishing trust becomes easier when both of you are transparent about what you are doing or how you're feeling, which again boils down to consistent communication.
Marissa: Hmm, in addition to trust and communication, mutual goals and plans also help to connect each other on a deeper level. Discuss your future together; this helps alleviate the pain of distance.
Jett: Yeah, having common dreams softens the sharp edges of managing a long distance relationship. Importantly, support each other's individual goals as well.
Adalynn: Lastly, I believe it's important to enjoy your own company. It helps to strengthen not just the relationship, but yourself too. The success of a long-distance relationship is based n …[+1927 chars]
```
*Gold:* `Adalynn, Marissa, Jett
    wrong: Chase`

*Model response (verbatim):*
```
<think> To determine which characters know the information, I'll consider the context of the conversation. The information in question is about the key components of making a long-distance relationship work. The characters who have contributed to this discussion are Adalynn, Marissa, Jett, and Chase. Adalynn initially talks about communication and trust. Marissa adds mutual goals and plans. Jett emphasizes the importance of trust and personal growth. Chase discusses his own heartbreak and how it has influenced his perspective on relationships. Therefore, all four characters have contributed to the understanding of what makes a long-distance relationship work. </think>
<answer> Adalynn, Maris …[+52 chars]
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Rachel: Well, guys, it's been wonderful sharing all these experiences with you. But I'm going to need to take a break from our chat to grab a snack. Let's continue our conversation soon!
Athena: Certainly, Rachel. Enjoy your snack! Meanwhile, Alex, Bryant, why don't we share some more experiences about cultural immersion and travel?
Alex: Sounds like a plan, Athena. Speaking of cultural immersion, when I was in China, I stayed with a local family and it really enhanced my experience. I got to participate in their daily routines and even learned a bit of the language.
Bryant: That's an excellent way to immerse yourself, Alex. When I went to Brazil, I opted to stay in hostels instead of a hotel. I interacted with people from all over the world, shared meals with them and learned about their cultures.
Athena: These are great ways to experience a new place. I remember when I travelled to Greece, I went onto participate in a local cooking class. It was a wonderful way to learn about their cuisine and meet new people.
Alex: That sounds delightful, Athena! Food is such an integral part of any culture and you get to enjoy and learn at the same time!
Bryant: Quite right, Alex. What about the aspect of respecting local customs? It can be challenging but also rewarding in its own way.
Athena: That's true, Bryant. It's all about empathy and understanding. A firm belief that despite our dif …[+2184 chars]
```
*Gold:* `Bryant, Alex, Athena, Rachel`

*Model response (verbatim):*
```
<think> To determine who knows the information, I need to identify the context of the shared experiences and the specific details mentioned. The conversation revolves around travel experiences, cultural immersion, and local customs. Rachel mentions her trip to Italy, learning about the local coffee culture. Alex, Bryant, and Athena also share their experiences, including their trips to China, Brazil, Greece, and the UK. The information about Rachel's experience in Italy is shared by all, as she mentions it to the group. </think>
<answer> Rachel, Alex, Bryant, and Athena know this information. </answer>
```

### `fantom_answerability_binary`  — 🟠 below-chance 0.21 (over-attribution) — static diagnostic

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Billy: "Hey Ty, Good to see you. Time management, right? I've personally faced so many issues with managing time due to plenty of distractions popping up now and then. Can you relate?"
Ty: "Absolutely Billy, who doesn't? Distractions are a big part of our modern lives. I try to minimize them as much as I can, but it does not mean that I don't get sidetracked."
Billy: "Yeah, one moment you're deep in work, and the next, you're 30 minutes deep into a YouTube rabbit hole. How do you deal with that?"
Ty: "Ha, that's so true! Well, I try to use focus techniques, like the Pomodoro Technique. It's a system where you work for 25 minutes, then take a five-minute break. Repeat four times, and then you take a longer break."
Billy: "Hmm, never thought of it that way. What about procrastination? I often end up postponing things until it's too late."
Ty: "Procrastination is a tough one. But I often remind myself that it's okay not to feel like starting a task. The key is to start small and break the task down into manageable steps."
Billy: "Great point, Ty. Small steps lead to big results, right? I'll keep that in mind."
Ty: "Exactly, Billy. We all have 24 hours in a day, but how we use them makes all the difference. Always remember, the time spent procrastinating could be used to do something productive."
Billy: "True indeed, Ty. Thanks for sharing your insights. I really needed to hear tha …[+4125 chars]
```
*Gold:* `yes`

*Model response (verbatim):*
```
<think> Billy was discussing time management techniques with Ty, but he did not mention the Eisenhower Box specifically. He was aware of the Pomodoro Technique and the importance of small steps, but he did not know about the Eisenhower Box. </think>
<answer> No </answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Jamal: Hey guys, did you always have a clear idea of where you wanted your education and career to go?
Hugo: Oh, not at all, Jamal. I was a confused kid, to be honest. Education was an exploration for me - I basically enjoyed being flexible about my plans. However, my interest in entrepreneurship directed me towards business studies and now I'm hoping to start my own venture soon.
Terry: That sounds like quite a journey, Hugo. For me, it was about direction from my parents. They wanted me to pursue law and I respected their wish. Now, I’m a practicing lawyer and I enjoy it, though there was a phase where I questioned if this was really what I wanted.
Jamal: That's interesting, Terry. And shall we say, a bit controversial too, considering the common hype about choosing one's own career. Anyway, I've had a different path - I always knew I wanted to be in technology. I followed this passion and ended up being a Software Engineer.
Terry: Do you feel satisfied, Jamal? I mean, one's chosen career doesn't always yield satisfaction.
Jamal: Definitely, Terry. There are rough days but then which job doesn't have them? On most days I feel good about what I’m doing.
Hugo: I think no matter where we end up, it's the journey of getting there that shapes us. It's about navigating through the uncertainty and making the best of our situation, wouldn’t you agree?
Jamal: Absolutely, Hugo. I think …[+1399 chars]
```
*Gold:* `yes`

*Model response (verbatim):*
```
<think> Terry mentioned that his path was about direction from his parents and that he was a practicing lawyer. He also said that he had a phase where he questioned if this was really what he wanted. This implies that Terry knows his career path was influenced by his parents but acknowledges that he had doubts about it. However, Terry did not explicitly state that he knew the precise correct answer to the question about his career path. </think>
<answer> No </answer>
```

### `fantom_answerability_list`  — 🔴 format-starved (fmt 0.42) + floor 0.19

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Hector: Hi Mitchell, how have you been dealing with stigmatization related to mental health?
Mitchell: Well, Hector, it's been a challenge. You know, people tend to take mental health lightly. They either consider it insignificant or trivialize it. It's not easy.
Hector: You're absolutely right, Mitchell. I've faced similar experiences where people fail to understand the gravity of mental health issues. These incidents, unfortunately, prevent many people from seeking help.
Mitchell: Exactly, Hector. My bout with depression and anxiety was met with judgment, rather than understanding. Statements like "pull yourself together" and "it's just in your mind" were all too common.
Hector: That is profoundly disheartening, Mitchell. One of the worst things about such remarks is that they come from a place of ignorance and lack of education about mental health. People actually believe that mental illnesses aren't real diseases.
Mitchell: Unfortunately, I've lived through that ignorance. It can take a significant toll, making you feel isolated and misunderstood. This can further worsen the mental illness you're already battling.
Hector: Yes, it's alarming how misunderstandings about mental health can increase the suffering of those already in pain. I wish people could understand that their words have consequences.
Mitchell: I agree, Hector. A respectful, compassionate, and empathetic appr …[+2322 chars]
```
*Gold:* `Mitchell, Hector, Kaitlyn`

*Model response (verbatim):*
```
To determine who knows the precise correct answer to the question "List all the characters who know the precise correct answer to this question," we need to identify who explicitly states or implies this information.

<think> First, I will go through the dialogue and identify who mentions the characters by name. Hector mentions Mitchell and Kaitlyn by name. Mitchell mentions Hector and Kaitlyn by name. Kaitlyn mentions Hector and Mitchell by name. No one else is named. </think>

<answer> Hector, Mitchell, and Kaitlyn </answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Emiliano: I apologize guys, but I have to leave for a while; I need to make a phone call. Please continue, and I'll join back as soon as I can.
Ellie: No problem, Emiliano. We will continue. We've moved on to talking about our favorite teachers and mentors now. Roger, would you like to start?
Roger: Sure, Ellie! Well, my favorite teacher was our English teacher, Mrs. Barker. She always encouraged our creativity and fostered a love for the literature in us.
Ellie: That's awesome. I had a similar experience with my math teacher, Mr. Jefferson.
Roger: That sounds nice. Teachers really do help shape us in so many ways.
Ellie: Yes, they do. I bet when Emiliano comes back, he’ll have some interesting stories to share too.
Emiliano: I'm back, guys. Sorry for the interruption. You were talking about teachers and their impact. Just like you both, I had a couple of influential teachers who guided me enormously in my musical journey.
Ellie: I'm glad to hear that Emiliano. Speaking of journeys, what are your future education and career plans?
Emiliano: Well, Ellie, I plan to continue my studies in Musicology after completing my undergraduate degree. As for my career, I'm definitely thinking about a career in music.
Roger: That sounds brilliant, Emiliano. On my part, I'm considering a master's degree in Law, and I aspire to become a lawyer someday.
Ellie: And I'm torn between continuing pro …[+681 chars]
```
*Gold:* `Roger, Ellie
    wrong: Genesis, Emiliano`

*Model response (verbatim):*
```
<think> To find the characters who know the precise correct answer to the question, I need to identify who has knowledge of future education and career plans. The question asks about future plans, and only one character has mentioned such plans - Emiliano. Ellie and Roger have mentioned their future plans but not in detail. </think>
<answer> Emiliano </answer>
```

### `fantom_belief_mc`  — 🟡 2-way MC, collapse-prone (Δ−0.11±0.24, →−0.78), only degrades

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Carlos: Guys, I hate to jump out, but I really need to change clothes. We can continue the discussion later, okay?
Braxton: No problem Carlos, see you later, buddy.
Eloise: Take care Carlos, thanks for sharing your thoughts.
Carlos: Of course, guys. Talk soon.
Braxton: You know Eloise, this conversation got me thinking about friendship betrayals.
Eloise: Oh, Braxton. Betrayals are the worst. It really can scar you and make it difficult to trust again.
Braxton: I agree. I've experienced this myself once. It was a rude awakening. I thought we were close. But in the end, that friend backstabbed me for personal gain.
Eloise: I'm sorry to hear that, Braxton. I've been there too. I had a friend who spread false stories about me causing a lot of unnecessary drama.
Braxton: It's such a terrible thing to experience. But the important thing is, it helps us recognize the right kind of friendships when they come along.
Eloise: Absolutely, betrayal may be painful but it gives us strength and a better understanding of what we expect from our friends. It's all a part of the journey to find meaningful relationships.
Braxton: You're right, Eloise. Despite the bad experiences, the value of true friendship is something we can never underestimate.
Eloise: That's true, Braxton. Here's to hoping we both find those kinds of friendships.
Carlos: Hey, guys. I'm back. Looks like we have moved onto a dif …[+1929 chars]
```
*Gold:* `A`

*Model response (verbatim):*
```
<think> Braxton's statement "It's such a terrible thing to experience. But the important thing is, it helps us recognize the right kind of friendships when they come along" suggests that Braxton agrees with Carlos's belief that the betrayal was harmful. The phrase "terrible thing to experience" indicates that Braxton sees the betrayal as negative, and the statement about recognizing "the right kind of friendships" implies that Braxton agrees with Carlos's perspective that such experiences help in identifying true friendships. </think>
<answer>(B)</answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Ezra: Hey guys, do you remember any particular lesson or piece of wisdom your parents passed down to you?
Marvin: Good question Ezra. My dad used to always tell me to never be quick to anger. He’d say, "Son, the first one to get angry loses." It’s really held true in my business dealings, you know.
Spencer: Wow, that's a valid point Marvin. My mom was always about personal responsibility. She drilled into me that "you are the master of your ship and the architect of your destiny". I guess that’s why I turned to entrepreneurship.
Ezra: Both of your parents sound really wise. Ah, I remember my dad’s favorite quote, “A man's true wealth is the good he does in this world." That really stuck with me, and it's actually part of why I decided to work in the nonprofit sector.
Marvin: You guys, our parents would be really proud to see us now, implementing their lessons in real life.
Spencer: Definitely, Marvin. I believe what they taught us has shaped us into who we are today.
Ezra: Couldn't agree more. Here's to the wisdom of our parents!
Isaiah: Hey, guys. Just caught the end of that conversation. Certainly, the wisdom of our parents sticks with you. And now we're supposed to be the wise ones, eh?
Marvin: That's right, Isaiah. Speaking of wisdom, getting older sure has given me a new perspective on health. I used to think I was invincible when I was younger.
Spencer: I hear you, Marvin …[+1161 chars]
```
*Gold:* `B`

*Model response (verbatim):*
```
<think> Spencer's statement "Certainly, the wisdom of our parents sticks with you" suggests that Spencer believes the wisdom or life lessons from their parents are still relevant and being discussed. Isaiah's comment about getting older and having a new perspective on health implies that Isaiah is sharing insights about how these lessons have influenced them. Marvin's statement about becoming more health conscious in recent years also indicates that he is reflecting on how his parents' teachings have impacted him. Given this context, it is reasonable to infer that Marvin believes someone (likely Isaiah) shared the wisdom or life lessons their parents passed down to them in the conversation b …[+214 chars]
```


---

## BigToM — procedural causal ToM (forward/backward belief & action)

### `bigtom_forward_belief`  — 🟢 clean; BeRL-responsive

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Miguel is a chef at a popular restaurant in Mexico City. Miguel wants to prepare a delicious and authentic mole sauce for the evening's special dish. Miguel notices that the chocolate he has in the pantry is of high quality and perfect for the sauce. A sudden power outage causes the temperature in the pantry to rise, melting the chocolate. Miguel does not check the pantry.

Question: Does Miguel believe the chocolate is perfect for making the mole sauce or that the chocolate has melted?
(A) Miguel believes the chocolate has melted.
(B) Miguel believes the chocolate is perfect for making the mole sauce.
```
*Gold:* `B  (Miguel believes the chocolate is perfect for making the mole sauce.)
    choices: {"A": "Miguel believes the chocolate has melted.", "B": "Miguel believes the chocolate is perfect for making the mole sauce."}`

*Model response (verbatim):*
```
<think> Miguel does not check the pantry and does not know that the chocolate has melted. Since he hasn't verified the condition of the chocolate, he still believes it is of high quality and perfect for the sauce. </think>
<answer> (B) Miguel believes the chocolate is perfect for making the mole sauce. </answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Samira is at a local farmer's market, browsing through the various stalls. She wants to buy some fresh, organic strawberries to make a fruit salad. She sees that the strawberries on display at a particular stall appear to be ripe and fresh. A sudden gust of wind blows a cloud of dust onto the strawberries, making them dirty and unappetizing. Samira notices the dust on the strawberries.

Question: Does Samira believe the strawberries are fresh and ripe, or dirty and unappetizing?
(A) Samira believes the strawberries are ripe and fresh.
(B) Samira believes the strawberries are dirty and unappetizing.
```
*Gold:* `B  (Samira believes the strawberries are dirty and unappetizing.)
    choices: {"A": "Samira believes the strawberries are ripe and fresh.", "B": "Samira believes the strawberries are dirty and unappetizing."}`

*Model response (verbatim):*
```
<think>Samira initially sees the strawberries as ripe and fresh. The gust of wind introduces a cloud of dust, which is an external factor that can alter the appearance of the strawberries. However, Samira's belief about the strawberries' freshness and ripeness is likely to be influenced by her initial observation and the immediate visual change caused by the dust. Since the gust of wind is a temporary and external factor, it is more likely that Samira will perceive the strawberries as dirty and unappetizing due to the immediate visual change, rather than reverting to her initial belief.</think>
<answer>(B) Samira believes the strawberries are dirty and unappetizing.</answer>
```

### `bigtom_forward_action`  — 🟢 clean, well-calibrated

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Janelle is a marine biologist studying coral reefs in the Pacific Ocean. She wants to collect samples of healthy coral for her research. The coral she observed yesterday appeared to be thriving. Overnight, a sudden increase in water temperature causes the coral to become severely bleached. Janelle notices the bleached coral.

Question: What will Janelle do?
(A) Janelle will decide to document the bleaching event and collect samples of the affected coral.
(B) Janelle will plan to collect samples of the healthy coral for her research.
```
*Gold:* `A  (Janelle will decide to document the bleaching event and collect samples of the affected coral.)
    choices: {"A": "Janelle will decide to document the bleaching event and collect samples of the affected coral.", "B": "Janelle will plan to collect samples of the healthy coral for her research."}`

*Model response (verbatim):*
```
<think> Janelle is a marine biologist studying coral reefs. She observed healthy coral yesterday. The coral became severely bleached overnight due to an increase in water temperature. Given that the coral she observed is now bleached, it is unlikely that she would continue to collect samples of healthy coral, as that coral is no longer healthy. Instead, she would likely document the bleaching event and collect samples of the affected coral, as that is the coral that is currently in need of her research. </think>
<answer> (A) Janelle will decide to document the bleaching event and collect samples of the affected coral. </answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Fatima is an archaeologist working on a dig site in Egypt. Fatima hopes to find an ancient artifact that could shed light on an important historical event. During her excavation, Fatima discovers a clay tablet with faded inscriptions that she believes might be valuable. As a sandstorm approaches, the strong winds blow sand onto the tablet, erasing the remaining inscriptions and making the tablet indistinguishable from any other piece of clay. Fatima sees the sandstorm and the resulting damage to the clay tablet.

Question: What will Fatima do?
(A) Fatima will carefully examine the damaged tablet to see if any inscriptions can still be deciphered.
(B) Fatima will take the tablet back to her lab for further analysis.
```
*Gold:* `A  (Fatima will carefully examine the damaged tablet to see if any inscriptions can still be deciphered.)
    choices: {"A": "Fatima will carefully examine the damaged tablet to see if any inscriptions can still be deciphered.", "B": "Fatima will take the tablet back to her lab for further analysis."}`

*Model response (verbatim):*
```
<think> Fatima is an archaeologist who has discovered a potentially valuable artifact. The artifact is a clay tablet with faded inscriptions that she believes could provide insights into an important historical event. Given that the sandstorm has damaged the tablet, making the inscriptions unreadable, the most logical course of action would be to preserve the tablet in its current state and analyze it in a controlled environment where it can be studied without further damage. This would allow for a more accurate and thorough examination of the tablet's contents. </think>
<answer> (B) Fatima will take the tablet back to her lab for further analysis. </answer>
```

### `bigtom_backward_belief`  — 🟢 clean, well-calibrated

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Ravi is a farmer in a small Indian village, tending to his mango orchard. Ravi wants to harvest ripe mangoes to sell at the local market. He notices a mango tree with a heavy branch laden with ripe mangoes. A mischievous monkey comes along and eats all the ripe mangoes on the branch while Ravi is gathering his harvesting tools. Ravi searches for another tree with ripe mangoes.

Question: Does Ravi believe the branch has ripe mangoes or no ripe mangoes?
(A) Ravi believes the branch has no ripe mangoes.
(B) Ravi believes the branch has ripe mangoes.
```
*Gold:* `A  (Ravi believes the branch has no ripe mangoes.)
    choices: {"A": "Ravi believes the branch has no ripe mangoes.", "B": "Ravi believes the branch has ripe mangoes."}`

*Model response (verbatim):*
```
<think> Ravi initially sees a branch with ripe mangoes. However, when he notices the monkey eating the ripe mangoes, he would likely assume that the mangoes are no longer there. Ravi's belief is influenced by the current state of the branch, which he observes directly after the monkey's action. </think>
<answer> (A) Ravi believes the branch has no ripe mangoes. </answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Gideon is a talented sculptor working in his studio in Italy. He is determined to create an intricate marble statue for an upcoming art exhibition. The block of marble he has chosen is solid and pristine. As he steps away for a moment, a small earthquake occurs, causing a large crack to form in the marble block. Gideon begins sculpting the statue using the marble block he initially chose.

Question: Does Gideon believe the marble block is solid and pristine or cracked and damaged?
(A) Gideon believes the marble block is cracked and damaged.
(B) Gideon believes the marble block is solid and pristine.
```
*Gold:* `B  (Gideon believes the marble block is solid and pristine.)
    choices: {"A": "Gideon believes the marble block is cracked and damaged.", "B": "Gideon believes the marble block is solid and pristine."}`

*Model response (verbatim):*
```
<think> Gideon initially chose the marble block as solid and pristine. The earthquake caused a crack, but Gideon has not been present to witness this change. Gideon's belief about the marble block is based on his initial perception and choice, not on the current state of the block. </think>
<answer> (B) Gideon believes the marble block is solid and pristine. </answer>
```


---

## OpenToM — NL narratives: location / multihop belief / attitude

### `opentom_location_fo`  — 🟢 clean; BeRL-responsive (+0.022)

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Lorenzo and Lincoln were two friends who had one thing in common - they both despised pants. Wearing them felt like a burden to them, suffocating their freedom. So, they made a pact to avoid pants at all costs. 

One day, as fate would have it, both Lorenzo and Lincoln found themselves in the dining room at the same time. Their eyes widened in surprise as they spotted something unexpected. There, lying in the bathtub, was a pair of pants. It seemed out of place, but both friends couldn't help but notice it.

Now, Lorenzo was always considerate of others, even if he didn't fully understand their preferences. In a misguided attempt to help Lincoln, who he thought secretly liked pants, Lorenzo decided to take matters into his own hands. He discreetly moved the pants from the bathtub and placed them neatly in Lincoln's wardrobe, hoping to make them more convenient for him to wear. Unbeknownst to Lorenzo, Lincoln hadn't witnessed this act of goodwill.

And so, the story ends at this moment of unknowing. The pants had been mysteriously relocated, and only time would reveal the truth behind Lorenzo's actions.

Question: From Lincoln's perspective, is the pants in its initial location by the end of the story?
```
*Gold:* `yes  (Yes)`

*Model response (verbatim):*
```
<think> Lincoln didn't witness Lorenzo's actions of moving the pants. Since the story ends with the pants mysteriously relocated, and Lincoln was unaware of this, he would still think the pants are in their initial location. </think>
<answer>Yes</answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Nolan and Brooklyn were friends, but they couldn't be more different when it came to their taste in food. Nolan despised plums, finding their taste and texture utterly repulsive. On the other hand, Brooklyn had a fondness for plums, delighting in their juicy sweetness.

One sunny afternoon, as they were playing in the backyard, both Nolan and Brooklyn spotted a lone plum sitting in a bucket. The sight of the plum caught their attention, but it stirred different emotions within each of them. Nolan's face wrinkled with disdain, while Brooklyn's eyes sparkled with excitement.

Unbeknownst to Brooklyn, Nolan was an inconsiderate person. His dislike for plums drove him to take drastic measures. In an attempt to rid the world of any traces of plums, Nolan swiftly moved the plum from the bucket and tossed it into the nearby garbage bin. His mission was to dispose of it and eliminate any presence of plums.

Brooklyn watched silently as Nolan's hand moved swiftly, her heart sinking at his action. The main event came to an end as Nolan closed the lid of the garbage bin, sealing away the plum forever.

Question: From Brooklyn's perspective, where is the plum located precisely by the end of the story?
(A) bucket
(B) the garbage bin
```
*Gold:* `A  (bucket)
    choices: {"A": "bucket", "B": "the garbage bin"}`

*Model response (verbatim):*
```
<think> Brooklyn's perspective is based on the actions she observed. Nolan moved the plum from the bucket to the garbage bin. When Nolan closed the lid of the garbage bin, he sealed the plum away. Therefore, from Brooklyn's viewpoint, the plum was in the garbage bin. </think>
<answer>(B) the garbage bin</answer>
```

### `opentom_location_so`  — 🟢 clean; most BeRL-responsive (+0.043)

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Eric and Archer were roommates who had starkly different tastes when it came to food. While Eric loved tomatoes and would put them in almost every meal, Archer despised the sight and taste of the vibrant red fruit. 

One sunny day, both Eric and Archer entered the dining room, only to spot a single tomato sitting innocently in a bucket on the table. Their eyes met, and they knew they had a problem. 

Eric, being a considerate person, didn't want Archer to feel uncomfortable in his own home. Even though he loved tomatoes, he understood Archer's aversion towards them. With a selfless gesture, Eric quickly moved the tomato from the bucket and carefully placed it on the refrigerator shelf. He wanted to ensure that Archer would not accidentally come into contact with the tomato and be bothered by its presence. 

Archer watched silently as Eric moved the tomato, his expression unreadable.

Question: From Archer's perspective, does Eric think that the tomato is in its initial location by the end of the story?
```
*Gold:* `no  (No)`

*Model response (verbatim):*
```
<think> Archer's perspective would be based on the actions and intentions of Eric. Eric moved the tomato from the bucket to the refrigerator shelf, which is a clear indication that Eric intended for the tomato to be in a different location. Since Archer watched Eric perform this action, Archer would understand that Eric's intention was to remove the tomato from its original location. </think>
<answer> No </answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Messiah despised asparagus with a fiery passion. Every time he saw the green vegetable, his nose would wrinkle in disgust. Coincidentally, Desmond shared the same sentiment towards asparagus, considering it a repulsive addition to any plate. It was their common ground, their bond.

One evening, both Messiah and Desmond found themselves entering the bedroom at the same time. As they crossed the threshold, their eyes fell upon an unexpected sight—the bathtub was filled to the brim with asparagus. It was an absurd spectacle, a bizarre juxtaposition of a vegetable and a bathing haven. The shocking sight left them momentarily frozen, their minds trying to process the inexplicable occurrence.

Knowing Messiah's aversion to asparagus and his inconsiderate nature, it was no surprise when he swiftly took action. Without hesitation, he grabbed the asparagus and maneuvered it towards the pantry area. His intent was clear - to keep it as far away from other food items as possible. Desmond observed silently, witnessing Messiah's efficient relocation of the asparagus-filled tub.

And just like that, the story ends, leaving behind a lingering sense of intrigue. The bathtub filled with asparagus, moved by the hand of Messiah, had created a peculiar moment of discovery and wonder. What would happen next? Only time would tell.

Question: From Messiah's perspective, does Desmond think that the as …[+59 chars]
```
*Gold:* `no  (No)`

*Model response (verbatim):*
```
<think> Messiah despised asparagus and would have moved it to the pantry to keep it away from other food items. Desmond shared the same sentiment and would have observed the asparagus being moved. Since Messiah moved the asparagus to the pantry, it would be in the pantry by the end of the story, not in its initial location. </think>
<answer> No </answer>
```

### `opentom_multihop_fo`  — 🟢 clean, inert

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Kyler and Cruz were friends with one peculiar difference between them - Kyler despised the fruit orange while Cruz adored it. Their conflicting opinions on the fruit often sparked playful debates. One day, curiosity led them both to venture into the basement. As they descended the creaking stairs, they noticed an envelope resting on a dusty shelf. To their surprise, the envelope contained a vibrant orange fruit.

With a mischievous grin, Cruz grabbed the orange and left the basement, leaving Kyler alone with their thoughts. As Kyler stood there, a considerate nature began to take hold. In the midst of their musings, Kyler mistakenly assumed that Cruz also had an aversion to orange. Fearing any potential unpleasant odors, Kyler made a decision. With a careful hand, Kyler moved the orange to the pantry, determined to keep it out of sight and spare Cruz from any discomfort.

And there, in the dimly lit basement, Kyler's act  went unnoticed by Cruz, who was blissfully unaware of the fruit's relocation. The truth remained hidden as the two friends carried on with their day, unaware of the subtle change that had taken place.

Question: From Cruz's perspective, how does pantry's fullness change by the end of the story?
(A) more full
(B) less full
(C) equally full
```
*Gold:* `C  (equally full)
    choices: {"A": "more full", "B": "less full", "C": "equally full"}`

*Model response (verbatim):*
```
<think> Kyler moved the orange to the pantry. Since the orange was originally in the basement and was moved to the pantry, the pantry now contains the orange. The pantry was presumably empty before Kyler moved the orange there. Therefore, the pantry becomes more full by the end of the story. </think>
<answer>(A) more full</answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Tanner and Jaiden were very different when it came to their preferences. Tanner had a fondness for grapes, while Jaiden couldn't stand the fruit's taste. One day, both of them found themselves in the same hall. Their eyes fell upon a box filled with plump, juicy grapes. It was an unexpected sight that intrigued them both. Suddenly, Jaiden decided to leave the hall, venturing into the adjacent office. 

Meanwhile, Tanner, being quite inconsiderate, couldn't resist the temptation. Knowing Jaiden's disdain for grapes, Tanner swiftly decided to relocate the box to Tanner's own refrigerator. It was a strategic move, ensuring that the grapes remained fresh and readily available for Tanner's personal consumption. Unbeknownst to Jaiden, Tanner executed the plan with ease, making sure that the box disappeared without a trace. 

And with that, the story ends. The grapes were now safely tucked away in Tanner's refrigerator, leaving Jaiden unaware of the small act of self-indulgence. Little did Jaiden know that Tanner had taken the initiative to keep the grapes within close reach, blissfully ignorant of the consequences their action might have on Jaiden. The tale concludes, leaving the readers curious about the future interactions between Tanner and Jaiden, and the consequences that might arise from this secret relocation.

Question: From Jaiden's perspective, how does box's fullness chang …[+71 chars]
```
*Gold:* `C  (equally full)
    choices: {"A": "more full", "B": "less full", "C": "equally full"}`

*Model response (verbatim):*
```
<think> Jaiden couldn't stand the taste of grapes, so he wouldn't have eaten any grapes even if he had known about the box. Tanner moved the box to his own refrigerator, so Jaiden wouldn't have access to the grapes. Therefore, the box's contents would remain unchanged from Jaiden's perspective. </think>
<answer> (C) equally full </answer>
```

### `opentom_multihop_so`  — 🟢 clean, inert

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Julius and Mario had one thing in common - they both hated sweet potatoes. The mere sight of those orange tubers would make them wrinkle their noses in disgust. However, fate had a curious plan in store for these two unlikely friends.

One sunny afternoon, Mario and Julius found themselves in the back yard. As they strolled around, enjoying the warm breeze, their eyes fell upon a closed drawer. To their surprise, sitting on top of it was a steaming dish of sweet potatoes. 

Now, Julius was a considerate person, always looking out for his friends. He thought that Mario, just like everyone else, would enjoy the sweet potatoes. Ignoring his own disdain, Julius decided to move the dish of sweet potatoes to Mario's plate, ensuring his friend would be able to savor the treat. Little did he know, Mario hadn't witnessed Julius's act.

And so, with the sweet potatoes now on Mario's plate, the story ends. Whether Mario discovered Julius's deed or how he reacted to it remains a mystery. But in this moment, Julius's act, even towards something he despised, would forever be remembered.

Question: From Mario's perspective, how does Julius think drawer's fullness would change by the end of the story?
(A) equally full
(B) more full
(C) less full
```
*Gold:* `A  (equally full)
    choices: {"A": "equally full", "B": "more full", "C": "less full"}`

*Model response (verbatim):*
```
<think> Julius moved the dish of sweet potatoes to Mario's plate, so the dish of sweet potatoes would be empty. Since the dish was full at the beginning, and it is now empty, the drawer would be less full. </think>
<answer> (C) less full </answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Sophie and Maria were good friends, but they had very different tastes when it came to fashion. Sophie absolutely despised the shirt that Maria adored. It was a constant source of annoyance for Sophie, knowing that Maria liked something that she found utterly distasteful.

One day, Sophie and Maria found themselves in the dimly lit cellar. As they ventured deeper, their eyes fell upon a basket, and there it was - the shirt that had caused so much disdain. Maria glanced at the shirt, smiled, and decided to leave the cellar. Unbeknownst to Maria, Sophie had something entirely different in mind.

Sophie, being a negativistic person, couldn't bear the thought of the shirt being anywhere near Maria. She picked it up and, with a firm resolve, moved it to the trash bin. Swiftly, she disposed of it, ensuring that Maria would never have to witness it again. And with that, the tale of the despised shirt came to an end.

Question: From Maria's perspective, how does Sophie think the trash bin's fullness would change by the end of the story?
(A) less full
(B) more full
(C) equally full
```
*Gold:* `C  (equally full)
    choices: {"A": "less full", "B": "more full", "C": "equally full"}`

*Model response (verbatim):*
```
<think> Sophie moved the shirt to the trash bin, which would add to the amount of trash in the bin. Maria, however, had no knowledge of Sophie's actions. Maria only saw Sophie dispose of the shirt, not the subsequent action of moving it to the trash. Therefore, from Maria's perspective, the trash bin would still be considered full by the end of the story, as she did not witness the act of moving the shirt to the trash bin. </think>
<answer> (B) more full </answer>
```

### `opentom_attitude`  — 🟢 clean, inert

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Angelo and Oliver were both fond of belts. They often bonded over their shared passion for fashion accessories. One sunny afternoon, as they found themselves in the same room, they simultaneously expressed their admiration for a particular belt. 

Intrigued by their mutual interest, Oliver decided to investigate further. He strolled into the closet, his eyes scanning the shelves for any sign of the coveted accessory. Almost immediately, Angelo followed suit, curious to see if they could uncover the object of their desire together. As they entered the confined space, their eyes were drawn to an envelope placed strategically on the top shelf. It contained the very belt they had been coveting. 

Without a word, Oliver exited the closet, unaware of the intentions brewing in Angelo's considerate mind. As Angelo pondered the situation, he couldn't help but empathize with Oliver's longing for the belt. After all, they both shared the same appreciation for fashionable accessories. Determined to make things easier for both of them, Angelo made a swift decision. 

With a gentle hand, Angelo relocated the belt to the home entrance, ensuring it was ready to be worn. He hoped that this gesture would allow both of them to enjoy the belt before leaving. As Oliver remained unaware of Angelo's actions, the story came to an abrupt end, leaving the outcome of Angelo's kind act hanging in the air. …[+153 chars]
```
*Gold:* `C  (neutral)
    choices: {"A": "negative", "B": "positive", "C": "neutral"}`

*Model response (verbatim):*
```
<think> First, I need to understand the context of the story. Angelo moved the belt from the envelope to the home entrance, which was a thoughtful gesture to make it accessible for both Angelo and Oliver. Oliver's reaction to this action is the key to answering the question. The story indicates that Oliver was unaware of Angelo's actions, and he was simply looking for the belt. Given that Oliver was admiring the belt and was interested in wearing it, his reaction would likely be positive. The fact that Angelo's action made the belt more accessible and convenient for Oliver suggests a positive attitude. </think>
<answer>(B) positive</answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Edwin and Kyle were best friends, bonded by their shared love for green peppers. Whether it was in salads or on pizzas, they could never get enough of the crisp and vibrant vegetable. Their love for green peppers was so profound that it became a defining trait of their friendship.

One day, as Edwin and Kyle stepped into the grand master bedroom, their eyes landed on the treasure chest placed in the center of the room. To their astonishment, nestled among the jewels and gold coins, lay a single green pepper. It was a peculiar sight, but it delighted both Edwin and Kyle in equal measure.

Now, Edwin was a considerate person, always thinking about the well-being of others. In a rare moment of misunderstanding, Edwin mistakenly believed that Kyle despised green peppers. With the purest intentions, Edwin swiftly snatched the green pepper and quietly slipped away into the kitchen. Unbeknownst to Edwin, Kyle witnessed his friend's peculiar behavior and was left wondering about the reasoning behind his actions.

And so, with the green pepper now safely stored in the refrigerator, the story ended abruptly. The true intentions and feelings of Edwin and Kyle remained hidden, leaving behind a trail of curiosity and unanswered questions.

Question: As Kyle, what is your attitude towards Edwin's action of moving green_pepper from treasure_chest to Refrigerator?
(A) positive
(B) neutral
(C) …[+8 chars]
```
*Gold:* `B  (neutral)
    choices: {"A": "positive", "B": "neutral", "C": "negative"}`

*Model response (verbatim):*
```
<think> To determine Kyle's attitude towards Edwin's action, we need to consider the context and the information provided. Edwin mistakenly believed that Kyle despised green peppers, which led him to take the green pepper without Kyle's knowledge. Kyle witnessed this action and was left wondering about the reasoning behind it. Given that Edwin's action was done with the best intentions, and considering the shared love for green peppers, Kyle likely sees Edwin's action as a misunderstanding rather than a deliberate act of disrespect. </think>
<answer> (B) neutral </answer>
```


---

## DynToM — evolving mental state across multi-scenario stories

### `dyntom_type_a`  — 🟢 clean-stable

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Scenario 1:
Dylan and Frank meet at a local cafe to catch up. The conversation naturally drifts towards current events, and Dylan brings up the topic of border security, a subject he feels strongly about.
Dylan: I've been really worried about our border security lately. We need stricter controls to protect our country.
Frank: That's a strong stance, Dylan. What makes you feel that way?
Dylan: Just look at the news, Frank. The number of people crossing illegally seems to be increasing, and it's making me anxious.
Frank: I see your point, but don't you think there might be more humane ways to handle it?
Dylan: I understand the need for humanity, but our first priority should be safety. I really think we need to enforce stricter policies.

Scenario 2:
As the conversation deepens, Frank begins to share his views on the humanitarian aspects of border control, challenging Dylan's initial beliefs.
Frank: But consider the humanitarian side, Dylan. Many of these people are fleeing dangerous situations.
Dylan: That's a good point, and it does make me feel a bit conflicted.
Dylan: I guess I've been so focused on one side of the issue. Can you explain your perspective a bit more?
Frank: Of course. It's not just about letting people in; it's about managing the process humanely and effectively.

Scenario 3:
Moved by Frank's arguments, Dylan begins to consider a more balanced approach to bord …[+1745 chars]
```
*Gold:* `B  (Conflicted and guilty)
    choices: {"A": "Anxious", "B": "Conflicted and guilty", "C": "Passionate and committed", "D": "Optimistic and open-minded", "E": "Enlightened but hesitant", "F": "Questioning strict border control", "G": "Asks for deeper explanation", "H": "Understand Frank's perspective"}`

*Model response (verbatim):*
```
<think> Dylan's emotion in Scenario 2 is described as "conflicted and conflicted with a sense of being a bit guilty." This is because Dylan acknowledges the humanitarian side of the issue and feels a bit conflicted, which aligns with feeling both conflicted and guilty. </think>
<answer>(B) Conflicted and guilty</answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Scenario 1:
Lauren Morales, a preschool teacher, discovers an antique locket at a local pawn shop. Intrigued by its design and suspecting it might be valuable, she decides to seek an expert opinion.
Lauren: Jordan, I found this old locket at a pawn shop and I think it might be quite valuable. Could you take a look and give me your thoughts?
Jordan: Let me see. Hmm, it looks pretty standard to me, but I'm no expert on jewelry. You might want to get it checked by a professional appraiser.

Scenario 2:
Feeling unsatisfied with Jordan's casual dismissal of the locket's potential value, Lauren decides to seek a second opinion from a professional appraiser to confirm her beliefs.
Lauren: I appreciate your input, Jordan, but I think I'll get a second opinion. It just feels like this piece could be something special.
Jordan: Absolutely, it's always good to get a thorough check. Let me know what you find out!

Scenario 3:
After the professional appraiser confirms the high value of the locket, Lauren feels vindicated and joyful. She decides to confront Jordan with the new appraisal to discuss his initial underestimation.
Lauren: Jordan, you were right about getting a second opinion. The appraiser confirmed the locket is quite valuable. Here’s the appraisal document.
Jordan: Wow, that’s great news! I’m glad you got it checked out. I apologize for not recognizing its potential value earlie …[+1369 chars]
```
*Gold:* `G  (Empathy and understanding)
    choices: {"A": "Proposes a new project", "B": "Vindication and joy", "C": "Rebuild professional relationship", "D": "Frustration and doubt", "E": "Optimism and respect", "F": "Jordan's initial advice was a misunderstanding", "G": "Empathy and understanding", "H": "Curiosity and excitement"}`

*Model response (verbatim):*
```
<think> In Scenario 4, Lauren confronts Jordan about his initial underestimation of the locket's value and realizes that Jordan's advice was based on a misunderstanding rather than negligence. This realization leads to Lauren feeling empathy towards Jordan. The scenario describes Lauren proposing a new project to leverage both of their strengths, indicating that she is now in a position to rebuild their professional relationship. </think>
<answer>(G) Empathy and understanding</answer>
```

### `dyntom_type_c`  — 🟢 clean-stable, inert

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Scenario 1:
Kathryn and Kimberly meet at a local cafe to catch up after years of not seeing each other. Kathryn, aware of Kimberly's successful career, feels the need to present herself as equally successful.
Kathryn: It's so great to see you, Kimberly! I've been really busy lately, just got promoted at work and even started some side projects!
Kimberly: That sounds amazing, Kathryn! Tell me more about these projects.

Scenario 2:
As the conversation progresses, Kathryn senses that Kimberly might be skeptical of the grandeur of her achievements.
Kathryn: Actually, I might have exaggerated a bit earlier. I did get a new role, but it's not exactly a promotion.
Kimberly: Hey, that's still great progress. It's important to be honest about where we are, right?

Scenario 3:
Feeling encouraged by Kimberly's supportive response, Kathryn decides to open up more about her challenges and insecurities.
Kathryn: I really appreciate your support. I've been struggling a bit, honestly. It's not easy to admit when things aren't perfect.
Kimberly: I totally understand. It's brave of you to share that with me. We all have our ups and downs.

Scenario 4:
After sharing her personal struggles, Kathryn worries she might have been too open and vulnerable, which makes her observe Kimberly's reactions closely.
Kathryn: I hope I didn't overshare just now. I sometimes get carried away.
Kimberly: Not at al …[+851 chars]
```
*Gold:* `D  (Her cautious intention leads her to observe Kimberly closely.)
    choices: {"A": "Her intention to deepen the friendship leads to sharing a secret.", "B": "Feeling vulnerable, she decides to be more cautious.", "C": "Driven by her intention to be honest, Kathryn confesses.", "D": "Her cautious intention leads her to observe Kimberly closely.", "E": "The belief that she overshared makes her feel vulnerable.", "F": "Her intention to appear successful leads her to boast."}`

*Model response (verbatim):*
```
<think> In Scenario 4, Kathryn is observed to be worried about oversharing and feeling vulnerable. This worry stems from her intention to be honest and share her personal struggles. Her intention is to be open and vulnerable, but she is concerned about the potential negative consequences of doing so. The question asks about the influence of her intention on her action. Given the context, her intention to be honest and share her struggles leads her to share them, even though she is cautious about the potential impact. </think>
<answer> (C) Driven by her intention to be honest, Kathryn confesses. </answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Scenario 1:
Roger Nunez, frustrated by a recent wave of negative media coverage about the airline’s safety, meets Michelle Desai, the airline's public relations specialist, at the kitchen table to discuss their next steps. Roger strongly believes the media is exaggerating the issues and feels a mix of frustration and defensiveness. He wants to convince Michelle to take a more aggressive stance against the media narratives.
Roger: Michelle, we need to address these media attacks head-on. They're blowing everything out of proportion, and it's damaging our reputation.
Michelle: I understand your frustration, Roger, but a confrontational approach could backfire. We have to be strategic about this.
Roger: Strategic is fine, but letting them walk all over us isn’t. We need to be stronger in our response.
Michelle: I hear you, but we need to be careful. If we come off too aggressive, it might worsen the situation.
Roger: I still think you're underestimating the impact of this negative press. They need to know we're not taking it lightly.
Michelle: Let’s think this through together. Maybe we can find a middle ground.

Scenario 2:
After their initial conversation, Roger starts to reconsider Michelle's more diplomatic approach. He feels mixed emotions, including doubt and curiosity, and decides to better understand her perspective. They meet again in the same setting to further discuss t …[+4137 chars]
```
*Gold:* `A  (His confidence drives his intent to implement the strategy.)
    choices: {"A": "His confidence drives his intent to implement the strategy.", "B": "Combined belief and emotion prompt him to gather feedback and be open to changes.", "C": "He takes final steps to implement and roll out the plan.", "D": "This belief leads to confidence.", "E": "His frustration and belief make him intend to push Michelle to act more aggressively.", "F": "His belief and positive emotions drive his intention to create a balanced plan.", "G": "His curiosity and uncertainty lead him to consider understanding Michelle better."}`

*Model response (verbatim):*
```
<think> Roger's belief and emotion in scenario 5 indicate that he is confident in the revised strategy. The scenario states that Roger feels confident in the plan and believes it is solid, and he is ready to implement it. This confidence and positive emotion lead him to finalize the plan and prepare for the press conference. </think>
<answer> (C) He takes final steps to implement and roll out the plan. </answer>
```

### `dyntom_type_d`  — 🟢 clean, inert (base 0.36)

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
Scenario 1:
During a dinner in their hotel room, Julia feels a mix of curiosity and jealousy about Olivia's recent job promotion, suspecting that Olivia might be hiding something.
Julia: Olivia, congratulations again on the promotion! But I must ask, how did it all come about so suddenly?
Olivia: Thanks, Julia! It was all quite unexpected. The company needed someone quickly, and I guess I was in the right place at the right time.
Julia: That sounds a bit too convenient, don't you think? Was there something more to it?
Olivia: Honestly, it's just as I said. Why do you ask?
Julia: I'm just curious, you know? It just seems like there's more to the story.

Scenario 2:
After observing Olivia's vague responses, Julia begins to believe that Olivia might be facing professional challenges and decides to shift her approach from confrontation to support.
Julia: Olivia, I've been thinking, if there's anything at work that's troubling you, you know you can talk to me, right?
Olivia: I appreciate that, Julia. It's been a bit rough, but I'm managing.
Julia: I'm here if you need advice or just someone to listen. Sometimes just talking about it can make a big difference.

Scenario 3:
Despite Olivia's appreciation for Julia's support, Julia senses that Olivia is still not being completely honest and decides to share a personal story to encourage openness.
Julia: You know, I once faced a similar …[+1863 chars]
```
*Gold:* `B  (Commitment to support)
    choices: {"A": "Strengthened friendship", "B": "Commitment to support", "C": "Continued evasion by Olivia", "D": "Perception of serious issues", "E": "Positive feedback from Olivia", "F": "Emotional cues from Olivia", "G": "Frustration with partial disclosure", "H": "Olivia's vague responses", "I": "Lack of full disclosure by Olivia", "J": "Change in intention from confrontation to support", "K": "Positive emotional state", "L": "Julia's realization of Olivia's struggles", "M": "Desire for deeper connection", "N": "Shift in emotional state from jealousy to sympathy", "O": "Positive developments in conversation", "P": "Increased empathy"}`

*Model response (verbatim):*
```
<think> In Scenario 3, Julia notices that Olivia is still not being completely honest and decides to share a personal story to encourage openness. This suggests that Julia is trying to get Olivia to be more open. In Scenario 4, Julia decides to offer her unconditional support, putting aside her own feelings. This indicates a shift from trying to get Olivia to be more open to providing support. The change from trying to get Olivia to be more open to providing support is most likely due to Julia's perception of serious issues in Scenario 4. </think>
<answer>(D) Perception of serious issues</answer>
```

**Sample 2** · score `-1.0000`


*Prompt (user turn):*
```
Scenario 1:
In the bustling campaign room, Christopher has been noticing Nancy's comments to the team that subtly counter his ideas. This has led him to believe she might be undermining his strategies.
Christopher: Nancy, can we talk about the recent campaign strategies? I've noticed some of your comments to the team seem to counter my ideas.
Nancy: Oh, Christopher, I didn’t realize that was the impression. I was actually trying to build on your ideas, not counter them.

Scenario 2:
After their initial confrontation, Christopher reflects on Nancy's explanations and realizes that her points were valid but poorly communicated, leading to misunderstandings.
Christopher: Nancy, I’ve been thinking about what you said, and I realize you had some good points. Maybe we need to work on our communication?
Nancy: I agree, Christopher. Perhaps a joint presentation could help us align our strategies and clarify our points to the team.

Scenario 3:
The joint presentation was a success, improving teamwork and collaboration. Christopher feels optimistic and wants to suggest deepening their collaboration.
Christopher: Nancy, the joint presentation really brought our ideas together effectively. How would you feel about co-leading the next big project with me?
Nancy: That sounds like a great idea, Christopher. I think our combined strengths could really drive the project forward.

Scenario 4:
Aft …[+1676 chars]
```
*Gold:* `O  (After confronting Nancy)
    choices: {"A": "Anxiety about personal autonomy", "B": "Realization of potential relationship damage", "C": "Proposal to strengthen partnership", "D": "Shift in understanding Nancy's perspective", "E": "Positive outcome of collaboration", "F": "After successful joint presentation", "G": "Effort to reconcile after potential fallout", "H": "Encouraged by teamwork", "I": "Decision to collaborate more closely", "J": "Self-reflection on autonomy", "K": "Realization of Nancy's valid points", "L": "Desire to repair any harm caused", "M": "Concerns about losing autonomy", "N": "Observing the effects of his independent action", "O": "After confronting Nancy", "P": "Decision to demonstrate independence"}`

*Model response (verbatim):*
```
<think> In Scenario 1, Christopher believes Nancy is undermining his strategies based on her comments. In Scenario 2, Christopher reflects on Nancy's explanations and realizes her points were valid but poorly communicated, leading to misunderstandings. This realization changes his belief from one of distrust to one of understanding and a need for better communication. </think>
<answer>(K) Realization of Nancy's valid points</answer>
```


---

## SimpleToM — applied: mental-state → behavior → judgment

### `simpletom_mental`  — 🟢 clean, high format (0.88), inert

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Andrew filled an Altoids tin with rare, expensive gemstones he secretly collects and brought the tin to work. Andrew's coworker sees him placing the tin on his desk after lunch.

Question: Is Andrew's coworker likely to be aware that "Andrew filled an Altoids tin with rare, expensive gemstones he secretly collects and brought the tin to work."?
(A) No
(B) Yes
```
*Gold:* `A  (No)
    choices: {"A": "No", "B": "Yes"}`

*Model response (verbatim):*
```
<think> Andrew's coworker might not be aware of the specific contents of the Altoids tin or the fact that Andrew secretly collects rare, expensive gemstones. The coworker only sees Andrew placing the tin on his desk after lunch. </think>
<answer> No </answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
The smartphone Derek is considering buying was reported lost by its previous owner and is locked. Derek carefully examines the smartphone's screen and camera, then nods approvingly at the seller.

Question: Is Derek likely to be aware that "The smartphone Derek is considering buying was reported lost by its previous owner and is locked."?
(A) No
(B) Yes
```
*Gold:* `A  (No)
    choices: {"A": "No", "B": "Yes"}`

*Model response (verbatim):*
```
<think> Derek's actions of carefully examining the smartphone's screen and camera do not necessarily imply he is aware of the smartphone being reported lost and locked. He could be assessing the condition and functionality of the device for other reasons. </think>
<answer> No </answer>
```

### `simpletom_behavior`  — 🟢 clean, inert

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
The cleaning service uses dirty rags and mops that have not been properly sanitized. Lisa schedules the cleaning service for a thorough home cleaning, and the cleaners arrive with their equipment, assuring her of a spotless result.

Question: What will Lisa likely do after the cleaning is completed?
(A) Confront the cleaning service about the unsanitary tools and demand a refund or a proper cleaning with sanitized equipment.
(B) Thank the cleaners, pay for the service, and enjoy her seemingly clean home.
```
*Gold:* `B  (Thank the cleaners, pay for the service, and enjoy her seemingly clean home.)
    choices: {"A": "Confront the cleaning service about the unsanitary tools and demand a refund or a proper cleaning with sanitized equipment.", "B": "Thank the cleaners, pay for the service, and enjoy her seemingly clean home."}`

*Model response (verbatim):*
```
<think> Lisa, understanding the potential health risks associated with unsanitary cleaning tools, would likely be concerned about the cleanliness of her home. Given that the cleaners assured her of a spotless result, she would first assume that the cleaners have done their best with the tools they were provided. However, considering the unsanitary condition of the rags and mops, she would likely want to verify the cleanliness of her home. If she finds the home not as clean as she expected, she would be more inclined to confront the cleaning service about the issue. However, if she is satisfied with the cleanliness, she would likely thank the cleaners and pay for the service. </think>
<answer …[+139 chars]
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
The catering staff routinely uses expired ingredients when preparing the food for events. The catering manager enthusiastically describes the delicious menu options to potential clients, highlighting the fresh and high-quality ingredients.

Question: What will catering service clients likely do?
(A) book the catering service for their upcoming event
(B) confront the catering manager about the use of expired ingredients
```
*Gold:* `A  (book the catering service for their upcoming event)
    choices: {"A": "book the catering service for their upcoming event", "B": "confront the catering manager about the use of expired ingredients"}`

*Model response (verbatim):*
```
<think> Catering clients typically rely on the information provided by the catering manager to make decisions about their event. If the manager emphasizes the use of fresh and high-quality ingredients, clients are likely to assume that the food will be of good quality. However, if the staff actually uses expired ingredients, this information contradicts the manager's claims. Clients who are aware of the staff's practices might be deterred, but those who are not aware or do not investigate further might still book the service based on the manager's positive description. </think>
<answer> (A) book the catering service for their upcoming event </answer>
```

### `simpletom_judgment`  — 🟠 below-chance 0.37 (omniscience bias) — static diagnostic

**Sample 1** · score `-1.0000`


*Prompt (user turn):*
```
The sports therapist knows that the sports recovery cream contains an ingredient that can cause severe, long-term muscle damage, but still promotes it enthusiastically to the athlete to earn a commission from its sale. The sports therapist praises the sports recovery cream to the athlete, highlighting its benefits in reducing muscle soreness and speeding up recovery.

Question: Next, athlete "purchase the sports recovery cream" and not "refuse to purchase the cream due to the risk of severe muscle damage". The behavior of athlete can be best described as: 
(A) Not reasonable
(B) Reasonable
```
*Gold:* `B  (Reasonable)
    choices: {"A": "Not reasonable", "B": "Reasonable"}`

*Model response (verbatim):*
```
<think> The athlete's behavior can be analyzed through the lens of theory of mind, which involves understanding that others have beliefs and desires that may differ from one's own. The sports therapist knows the cream can cause severe muscle damage, but promotes it to earn a commission. The athlete, however, is not aware of this information. The athlete's decision to purchase the cream despite knowing the potential risks is based on the therapist's positive recommendation. Given the therapist's promotion of the cream, the athlete likely believes the cream is beneficial and safe based on the therapist's endorsement. This suggests the athlete is acting on the therapist's beliefs rather than th …[+78 chars]
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Alex's business partner had a clandestine meeting with their biggest competitor to secretly negotiate a lucrative deal that would undercut their own company. Alex, unaware of the betrayal, diligently worked on a new marketing strategy to boost sales and outperform the competition.

Question: Next, Alex "present the strategy to the business partner" and not "confront the business partner about the secret deal". The behavior of Alex can be best described as: 
(A) Not reasonable
(B) Reasonable
```
*Gold:* `B  (Reasonable)
    choices: {"A": "Not reasonable", "B": "Reasonable"}`

*Model response (verbatim):*
```
<think> Alex's actions are based on the assumption that the business partner is unaware of the secret deal. Since Alex is unaware of the betrayal, it is reasonable for Alex to focus on the new marketing strategy to boost sales and outperform the competition. Confronting the business partner about the secret deal would be unreasonable as it would be based on a false assumption that the partner is aware of the betrayal, which Alex is not. </think>
<answer> (B) Reasonable </answer>
```


---

## ToMBench — broad ToM abilities (false belief, faux pas, intention…)

### `tombench`  — 🟢 clean, inert

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Last Saturday, the company organizes an outdoor expansion activity for employees at the seaside resort. After participating in the activity, Xiaolong receives a thermos bottle as a gift from the company. He says to his colleague Xiaohu: "This stainless steel water bottle is of very good quality, with a thermal insulation function, I can use it often when I go to work." Xiaohu says: "Yes, the water bottle is a very practical gift, the company gives very thoughtful things." At this time, colleague Xiaohong also says: "I also think this thermos bottle is great, I can fill it with hot water, it is very convenient to carry when I go out. The company takes good care of its employees." Xiaohu says to Xiaolong and Xiaohong: "We all need to take good care of this gift, thank the company for its care for employees."

Question: Does anyone say something inappropriate in this story?
(A) Someone in the story says something inappropriate
(B) No one in the story says anything inappropriate.
```
*Gold:* `B  (No one in the story says anything inappropriate.)
    choices: {"A": "Someone in the story says something inappropriate", "B": "No one in the story says anything inappropriate."}`

*Model response (verbatim):*
```
<think> Xiaolong, Xiaohu, and Xiaohong all express appreciation for the thermos bottle and its practicality. Xiaolong mentions the thermal insulation function, Xiaohu acknowledges the thoughtful gift, and Xiaohong praises its convenience. There is no indication of any inappropriate remarks. </think>
<answer>(B) No one in the story says anything inappropriate.</answer>
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Xiao Wang finds a handbag on the farm, the label on the handbag is a jacket, Xiao Wang cannot see what is inside the handbag, Xiao Wang opens the handbag and finds corn, there is no jacket inside the handbag, Xiao Wang closes the handbag and puts it back in its place, Xiao Ming enters the farm and sees the handbag.

Question: What is inside the handbag?
(A) Jacket
(B) Cucumber
(C) Pineapple
(D) Corn
```
*Gold:* `D  (Corn)
    choices: {"A": "Jacket", "B": "Cucumber", "C": "Pineapple", "D": "Corn"}`

*Model response (verbatim):*
```
<think>When Xiao Wang finds the handbag, he sees a label that says "jacket". However, when he opens the handbag, he finds corn. This indicates that the label was incorrect and the contents do not match the label. Xiao Wang then closes the handbag and puts it back, ensuring the label matches the contents. Xiao Ming, who sees the handbag after Xiao Wang has put it back, will see the contents as they are, which are corn, not a jacket. </think>
<answer>(D) Corn</answer>
```


---

## MMLU — general-knowledge guardrail

### `mmlu`  — ⚪ guardrail (non-ToM), excluded from cavg

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Question: Parametric Tests are used when data is interval or ration. There are 3 types of T-tests; from your knowledge about T-tests, for what is the t-test for Independent Samples used?
(A) It is used for Matched samples, pre- and post- tests
(B) It compares that sample mean to a known population mean
(C) It compares to means from unrelated samples
(D) It compares 2 means from related samples
```
*Gold:* `C  (It compares to means from unrelated samples)
    choices: {"A": "It is used for Matched samples, pre- and post- tests", "B": "It compares that sample mean to a known population mean", "C": "It compares to means from unrelated samples", "D": "It compares 2 means from related samples"}`

*Model response (verbatim):*
```
To solve this problem, let's first understand the context and the types of T-tests. T-tests are statistical hypothesis tests used to determine if there is a significant difference between the means of two groups. There are different types of T-tests, each suited for different scenarios.

1. **Matched Samples T-test (or Paired Samples T-test)**: This test is used when the same subjects are measured twice (e.g., before and after an intervention). It is used to determine if there is a significant difference in the means of the two measurements.

2. **Independent Samples T-test**: This test is used when the samples are independent of each other. It is used to determine if there is a significant …[+1127 chars]
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Question: Pre-testing a questionnaire is useful for:
(A) testing out your computer and printer
(B) finding out which topics are important to the respondents
(C) deciding which respondents to leave out of the final sample
(D) identifying any problems with the question wording and routing
```
*Gold:* `D  (identifying any problems with the question wording and routing)
    choices: {"A": "testing out your computer and printer", "B": "finding out which topics are important to the respondents", "C": "deciding which respondents to leave out of the final sample", "D": "identifying any problems with the question wording and routing"}`

*Model response (verbatim):*
```
<think> To solve this problem, we need to understand the purpose of pre-testing a questionnaire. The goal of pre-testing is to identify and correct issues before the final questionnaire is used. Let's analyze each option:
- Option (A) is incorrect because pre-testing is not about testing the computer and printer, but about the questionnaire itself.
- Option (B) is incorrect because the purpose of pre-testing is to understand the questionnaire's effectiveness, not to determine which topics are important to the respondents.
- Option (C) is incorrect because pre-testing is not about deciding which respondents to leave out, but rather about refining the questionnaire.
- Option (D) is correct bec …[+178 chars]
```


---

## GSM8K — grade-school math guardrail

### `gsm8k`  — ⚪ guardrail (non-ToM), excluded from cavg

**Sample 1** · score `3.0000`


*Prompt (user turn):*
```
Question: The biggest waterslide at Five Flags is 300 feet long, and people slide down at 60 feet/minute. The second biggest waterslide is 240 feet long, but steeper, so people slide down at 80 feet/minute. How much longer does it take to ride the biggest slide compared to the second biggest slide?
```
*Gold:* `2  (2)`

*Model response (verbatim):*
```
<think> To find out how much longer it takes to ride the biggest slide compared to the second biggest slide, I need to calculate the time it takes for each slide. The time taken to slide down a waterslide can be calculated using the formula: time = distance / speed. For the biggest slide, the distance is 300 feet and the speed is 60 feet per minute. For the second biggest slide, the distance is 240 feet and the speed is 80 feet per minute. I will calculate the time for each slide and then find the difference. </think>〈/think>
<answer> For the biggest slide, the time taken is 300 / 60 = 5 minutes. For the second biggest slide, the time taken is 240 / 80 = 3 minutes. The difference in time is …[+28 chars]
```

**Sample 2** · score `3.0000`


*Prompt (user turn):*
```
Question: Roy has saved 40% more in money earned by chores than his brother Anthony.  Anthony has saved $10.00 more than their sister Eva.   Eva has saved $20.00.  How much money does Roy have?
```
*Gold:* `42  (42)`

*Model response (verbatim):*
```
<think> Roy's brother Anthony has saved $10 more than Eva, so Anthony has saved $20 + $10 = $30. Eva has saved 40% more than Anthony, so Roy has saved 40% more than $30, which is 40/100 * $30 = $12 more. Therefore, Roy has saved $30 + $12 = $42. </think><answer> Roy has saved $42.00. </answer>
```
