#!/usr/bin/env python3
"""Generate a machine-readable experiment sidecar from the tracker markdown table.

The tracker (`project_planning/BeRL_experiments_tracker.md`) is the human-facing source of
truth, but its markdown table is fragile to parse in bash. This script extracts each row's
knobs into `project_planning/experiments.tsv` (tab-separated, one row per Exp ID) which the
dispatcher `experiments/run_experiment.sh <EXP_ID>` reads to set env knobs.

Keep the sidecar in sync: re-run this whenever tracker knob cells change:
    python scripts/tracker_to_sidecar.py

Only knob columns needed by the launchers are emitted; free-text columns (Question, Data
params, Results, Notes) stay in the markdown.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACKER = REPO / "project_planning" / "BeRL_experiments_tracker.md"
SIDECAR = REPO / "project_planning" / "experiments.tsv"

SIDECAR_COLS = [
    "exp_id", "model_family", "model_path", "task", "reward_type",
    "power_k", "power_ll_min", "rm_mode", "kl", "lr",
    "max_prompt", "max_resp", "data_name", "cot_var", "run_name_base",
]

# Model/Size -> HF path.
MODEL_PATHS = {
    ("qwen2.5", "0.5b"): "Qwen/Qwen2.5-0.5B-Instruct",
    ("qwen2.5", "1.5b"): "Qwen/Qwen2.5-1.5B-Instruct",
    ("qwen2.5", "3b"): "Qwen/Qwen2.5-3B-Instruct",
    ("qwen2.5", "7b"): "Qwen/Qwen2.5-7B-Instruct",
    ("qwen3", "1.7b"): "Qwen/Qwen3-1.7B",
    ("qwen3", "4b"): "Qwen/Qwen3-4B",
    ("qwen3", "8b"): "Qwen/Qwen3-8B",
    ("gemma", "2b"): "google/gemma-2-2b-it",
    ("gemma", "9b"): "google/gemma-2-9b-it",
}


def norm_family(model: str) -> str:
    m = model.strip().lower()
    if "qwen3" in m:
        return "qwen3"
    if "qwen2.5" in m or "qwen2" in m:
        return "qwen2.5"
    if "gemma" in m:
        return "gemma"
    return ""


def norm_size(size: str) -> str:
    return size.strip().lower().replace(" ", "")


def map_task_reward(loss: str) -> tuple[str, str]:
    """Return (task, reward_type)."""
    l = loss.strip().lower()
    if "rule_based_tom" in l:
        return "tom_rulebased", ""
    if l.startswith("sft") or l == "sft":
        return "sft", ""
    if "log_prob" in l:
        return "behavior", "log_prob"
    if "neg_perplexity" in l or "neg_ppl" in l:
        return "behavior", "neg_perplexity"
    if "behavior+rule" in l:
        return "behavior", "power"  # mixed reward: behavior side defaults to power
    if "power" in l:
        return "behavior", "power"
    return "", ""


def parse_powers(powers: str) -> tuple[str, str]:
    """Extract (power_k, power_ll_min) from a free-text 'Loss powers' cell."""
    if not powers or powers.strip() in {"-", "—", ""}:
        return "", ""
    k = ll = ""
    mk = re.search(r"k\s*=?\s*(-?\d+(?:\.\d+)?)", powers, re.I)
    ml = re.search(r"(?:ll[_ ]?min|llmin)\s*=?\s*(-?\d+(?:\.\d+)?)", powers, re.I)
    if mk:
        k = mk.group(1)
    if ml:
        ll = ml.group(1)
    return k, ll


def parse_ctx(ctx: str) -> tuple[str, str]:
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", ctx or "")
    return (m.group(1), m.group(2)) if m else ("", "")


def norm_rm(rm: str) -> str:
    r = (rm or "").strip().lower()
    if r.startswith("actor"):
        return "actor"
    if r.startswith("frozen"):
        return "frozen"
    return ""


def clean(cell: str) -> str:
    c = cell.strip()
    return "" if c in {"-", "—", "TBD", "?"} else c


def main() -> int:
    if not TRACKER.exists():
        print(f"tracker not found: {TRACKER}", file=sys.stderr)
        return 1
    lines = TRACKER.read_text().splitlines()
    header_idx = next((i for i, ln in enumerate(lines)
                       if ln.startswith("| Exp #") and "Exp ID" in ln), None)
    if header_idx is None:
        print("could not find tracker table header", file=sys.stderr)
        return 1
    header = [h.strip() for h in lines[header_idx].strip("|").split("|")]
    col = {name: idx for idx, name in enumerate(header)}

    def g(cells, name):
        i = col.get(name)
        return clean(cells[i]) if i is not None and i < len(cells) else ""

    rows = []
    for ln in lines[header_idx + 2:]:
        if not ln.strip().startswith("|"):
            if ln.strip() and not ln.startswith(" "):
                break  # left the table
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        exp_id = g(cells, "Exp ID")
        if not exp_id or exp_id.lower() == "exp id":
            continue
        family = norm_family(g(cells, "Model"))
        size = norm_size(g(cells, "Size"))
        model_path = MODEL_PATHS.get((family, size), "")
        task, reward_type = map_task_reward(g(cells, "Loss/reward type"))
        pk, ll = parse_powers(g(cells, "Loss powers"))
        mp, mr = parse_ctx(g(cells, "Gen ctx (prompt/resp)"))
        rows.append({
            "exp_id": exp_id,
            "model_family": family,
            "model_path": model_path,
            "task": task,
            "reward_type": reward_type,
            "power_k": pk,
            "power_ll_min": ll,
            "rm_mode": norm_rm(g(cells, "RM mode")),
            "kl": g(cells, "KL"),
            "lr": g(cells, "LR"),
            "max_prompt": mp,
            "max_resp": mr,
            "data_name": g(cells, "Data config name"),
            "cot_var": g(cells, "CoT prompt var"),
            "run_name_base": g(cells, "Run name"),
        })

    with SIDECAR.open("w") as f:
        f.write("\t".join(SIDECAR_COLS) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in SIDECAR_COLS) + "\n")
    print(f"wrote {len(rows)} rows -> {SIDECAR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
