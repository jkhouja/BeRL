## Env (one-time)
```bash
pip install "datasets>=2.18" convokit pandas matplotlib ipywidgets nbclient nbformat
```

## CRITICAL gotcha (HF `datasets` ≥3 / 5.0)
Script-based datasets fail with "Dataset scripts are no longer supported." Fix: load the
auto-converted parquet via `revision="refs/convert/parquet"`, or use a community parquet mirror.

## Sources & loaders (all `split="train"`)
| Dataset | How to load | Notes |
|---|---|---|
| DailyDialog | `load_dataset("roskoN/dailydialog", revision="refs/convert/parquet")` | cols: utterances, acts, emotions; 2-spkr alternating |
| EmpatheticDialogues | `load_dataset("Estwld/empathetic_dialogues_llm")` | `conversations`(role/content)+emotion+situation |
| CaSiNo | `load_dataset("kchawla123/casino")` | `chat_logs`,`participant_info`=hidden value priorities |
| CraigslistBargain | `load_dataset("stanfordnlp/craigslist_bargains", revision="refs/convert/parquet")` | utterance[],agent_turn[],hidden target prices |
| DealOrNoDeal | `load_dataset("mikelewis0/deal_or_no_dialog", revision="refs/convert/parquet")` | `dialogue` "YOU:/THEM:" + `<eos>`; some rows None→skip |
| CGA (Conversations Gone Awry) | `convokit download("conversations-gone-awry-corpus")` | long forum comments (~66 w/utt), 3+ speakers |
| Switchboard (SwDA) | `convokit download("switchboard-corpus")` | ~106 turns/conv; dialog-act tags in utt meta |
| Diplomacy (deception) | `convokit download("diplomacy-corpus")` | intent_lie/perceived_lie labels; long games |
| PersuasionForGood | `convokit download("persuasionforgood-corpus")` | donation persuasion; ER/EE roles |
| ThoughtTrace | `load_dataset("SCAI-JHU/ThoughtTrace")` | human–AI; `messages[].type`=role, `messages[].reasons`=user thoughts; +survey demographics |

ConvoKit pattern: `Corpus(download(id))` → `iter_conversations()` → `iter_utterances()` (`.speaker.id`, `.text`, `conv.meta`).

## Skipped
- CANDOR — registration + multi-GB; skip unless explicitly needed.
- CICERO — covered by ConvoKit diplomacy-corpus.

## Normalized schema (all datasets)
`{dataset, dialog_id, num_turns, num_speakers, speakers, utterances:[{speaker,text[,thought]}], meta}`
Saved: `data/<n>.parquet` (utterances/speakers/meta JSON-encoded), `samples/<n>.json` (5 dialogs), `stats/<n>.json`, `stats/SUMMARY.csv`.

## Stats (train)
| dataset | dialogs | utts | avg_turns | avg_words/utt |
|---|--:|--:|--:|--:|
| dailydialog | 11,118 | 87,170 | 7.8 | 13.6 |
| empathetic_dialogues | 19,533 | 84,168 | 4.3 | 13.4 |
| casino | 1,030 | 14,297 | 13.9 | 16.2 |
| craigslist_bargain | 5,247 | 47,986 | 9.2 | 10.8 |
| dealornodeal | 10,095 | 60,528 | 6.0 | 8.5 |
| conversations_gone_awry | 4,188 | 30,021 | 7.2 | 65.9 |
| switchboard | 1,155 | 122,646 | 106.2 | 16.7 |
| diplomacy | 246 | 17,289 | 70.3 | 20.4 |
| persuasionforgood | 1,017 | 20,932 | 20.6 | 16.8 |
| thoughttrace | 2,155 | 17,058 | 7.9 | 231.5 |

## Processing tips for BeRL (perplexity-of-next-utterance reward)
- **High ToM-signal** (next utterance needs inferring hidden state): CaSiNo, CraigslistBargain, DealOrNoDeal, PersuasionForGood, Diplomacy. Low-signal smalltalk: DailyDialog → predictable without ToM (shortcut risk).
- **Long utts**: CGA (~66 w) and ThoughtTrace (~231 w) → cap/trim or budget context. Switchboard/Diplomacy are very long-turn → multi-turn/context-length pressure.
- **ThoughtTrace** uniquely has gold user `thought` — enables behavior-vs-thought ablations; human–AI not human–human.
- **No HF token** needed but set `HF_TOKEN` to dodge rate limits. ConvoKit caches to `~/.convokit/`.
- Keep held-out splits; report an OOD non-dialogue ToM eval to detect genre overfitting.

