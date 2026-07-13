#!/usr/bin/env python3
"""Canonical config-selection scorer for BeRL Phase-stability / GRPO runs.

Parses a training log and reports the **standard** metrics that every experiment
row must record, so numbers are comparable across agents:

  * ToM HM        - harmonic mean over the 24 Theory-of-Mind benchmarks
                    (excludes gsm8k and mmlu). Primary config-selection score.
  * ToM avg       - arithmetic mean over the same 24 ToM benchmarks (reported
                    alongside HM; HM stays primary, avg is complementary).
  * gsm8k         - reported SEPARATELY as a math-reasoning regression eval.
  * mmlu          - reported SEPARATELY as a general-knowledge regression eval.
  * parseable     - answer-parse rate = 1 - reward/format_error_ratio (tracked,
                    NOT folded into HM).
  * max_resp      - max_response_length used by the run (NOT a tracker column;
                    varies per launch, so recorded here for comparability).
  * health        - KL / entropy / response-length snapshot for collapse checks.

Aggregation (fixed convention):
  For a window of the last N eval iterations, each benchmark score is first
  AVERAGED over those N iterations ("avg-then-HM"), then the harmonic mean is
  taken across benchmarks. We report N=5 (primary) and N=3. The ToM arithmetic
  mean is reported alongside the HM. The per-iteration HM trajectory is also
  printed so dips/recoveries are visible.

Usage:
  python scripts/score_run.py <training.log> [--last 5] [--json]

Never edit the ToM set here without team sign-off - consistency is the point.
"""
import argparse
import json
import re
import sys
from statistics import harmonic_mean, mean

# --- Benchmark taxonomy (val/test_score/<name>_sub300) ------------------------
# Non-ToM benchmarks are scored and reported SEPARATELY, never inside the HM.
CAPABILITY_EVALS = ("gsm8k", "mmlu")  # gsm8k=math reasoning, mmlu=general knowledge

# Harmonic mean is undefined at 0 and explodes near 0; floor guards that while
# still letting a collapsed benchmark tank the HM (which is the desired signal).
HM_EPS = 1e-3

VAL_RE = re.compile(r"val/test_score/(\w+?)_sub300:([0-9.]+)")
STEP_RE = re.compile(r"step:(\d+) -")
HEALTH_RE = re.compile(r"step:(\d+) - global_seqlen")
# max_response_length is NOT a tracker column and varies per launch (family
# default / manual MAX_RESP override), so surface it here to keep the canonical
# Results string comparable across agents. See docs/CHANGE_HISTORY.md.
# Matches both the hydra CLI echo (max_response_length=512) and the OmegaConf
# config dump verl prints at startup ('max_response_length': 512).
MAX_RESP_RE = re.compile(r"max_response_length['\"]?\s*[:=]\s*(\d+)")


def hmean(scores):
    return harmonic_mean([max(v, HM_EPS) for v in scores])


def parse_eval_iters(path):
    """Return sorted list of (step, {benchmark: score}) for every eval iteration."""
    iters = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if not m:
                continue
            scores = {k: float(v) for k, v in VAL_RE.findall(line)}
            if not scores:
                continue
            step = int(m.group(1))
            # keep the richest record for a step (eval merged onto train lines)
            if step not in iters or len(scores) > len(iters[step]):
                iters[step] = scores
    return sorted(iters.items())


def parse_health(path):
    """Return {step: {kl, entropy, resp_len, reward, fmt_err}} from train lines."""
    keys = {
        "reward": r"reward/mean:(-?[0-9.]+)",
        "resp_len": r"response_length/mean:(-?[0-9.]+)",
        "entropy": r"actor/entropy_loss:(-?[0-9.]+)",
        "kl": r"actor/kl_loss:(-?[0-9.]+)",
        "fmt_err": r"reward/format_error_ratio:(-?[0-9.]+)",
    }
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = HEALTH_RE.search(line)
            if not m:
                continue
            rec = {}
            for name, pat in keys.items():
                mm = re.search(pat, line)
                if mm:
                    rec[name] = float(mm.group(1))
            out[int(m.group(1))] = rec
    return out


def parse_max_resp(path):
    """Return the max_response_length used by the run (int) or None.

    Not a tracker column and varies per launch, so we extract it from the log's
    hydra command echo so every canon Results string can record it.
    """
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = MAX_RESP_RE.search(line)
            if m:
                return int(m.group(1))
    return None


def window_avg(iters, benchmarks, last_n):
    """Average each benchmark over the last N eval iters; return {bench: avg}."""
    window = iters[-last_n:]
    present = set.intersection(*[set(s) for _, s in window]) if window else set()
    keys = [b for b in benchmarks if b in present]
    return {b: sum(s[b] for _, s in window) / len(window) for b in keys}


def score(path, last_n=5):
    iters = parse_eval_iters(path)
    if not iters:
        raise SystemExit(f"No val/test_score lines found in {path}")

    all_benches = sorted({b for _, s in iters for b in s})
    tom = [b for b in all_benches if b not in CAPABILITY_EVALS]

    # per-iteration ToM HM trajectory (excludes capability evals)
    traj = [(step, hmean([s[b] for b in tom if b in s])) for step, s in iters]

    def tom_hm(n):
        avg = window_avg(iters, tom, n)
        return hmean(avg.values()) if avg else float("nan")

    def tom_avg(n):
        avg = window_avg(iters, tom, n)
        return mean(avg.values()) if avg else float("nan")

    def cap_avg(name, n):
        vals = [s[name] for _, s in iters[-n:] if name in s]
        return sum(vals) / len(vals) if vals else None

    step0 = iters[0][1]
    step0_tom = [step0[b] for b in tom if b in step0]
    result = {
        "log": path,
        "n_eval_iters": len(iters),
        "first_step": iters[0][0],
        "last_step": iters[-1][0],
        "max_response_length": parse_max_resp(path),
        "n_tom_benchmarks": len(tom),
        "tom_hm_last5": round(tom_hm(5), 4),
        "tom_hm_last3": round(tom_hm(3), 4),
        "tom_hm_step0": round(traj[0][1], 4),
        "tom_avg_last5": round(tom_avg(5), 4),
        "tom_avg_last3": round(tom_avg(3), 4),
        "tom_avg_step0": round(mean(step0_tom), 4) if step0_tom else None,
        "tom_hm_trajectory": [(s, round(h, 3)) for s, h in traj],
    }
    for name in CAPABILITY_EVALS:
        base = step0.get(name)
        last = cap_avg(name, last_n)
        result[f"{name}_last{last_n}"] = round(last, 4) if last is not None else None
        result[f"{name}_step0"] = round(base, 4) if base is not None else None
        if base is not None and last is not None:
            result[f"{name}_delta"] = round(last - base, 4)

    # health snapshot at final eval-aligned train steps
    health = parse_health(path)
    if health:
        hsteps = sorted(health)
        last = health[hsteps[-1]]
        result["health_final"] = {k: round(v, 3) for k, v in last.items()}
        result["parseable_final"] = round(1.0 - last.get("fmt_err", 0.0), 4)
    return result


def fmt_human(r):
    lines = []
    lines.append(f"log: {r['log']}")
    lines.append(
        f"eval iters: {r['n_eval_iters']} (step {r['first_step']}..{r['last_step']}); "
        f"ToM benchmarks: {r['n_tom_benchmarks']} (excl gsm8k, mmlu)"
    )
    lines.append(
        f"ToM HM(last5)={r['tom_hm_last5']}  HM(last3)={r['tom_hm_last3']}  "
        f"(baseline step0={r['tom_hm_step0']})"
    )
    lines.append(
        f"ToM avg(last5)={r['tom_avg_last5']}  avg(last3)={r['tom_avg_last3']}  "
        f"(baseline step0={r['tom_avg_step0']})"
    )
    for name in CAPABILITY_EVALS:
        k = next((x for x in r if x.startswith(name + "_last")), None)
        if k and r[k] is not None:
            d = r.get(f"{name}_delta")
            dtxt = f", delta vs step0={d:+.3f}" if d is not None else ""
            lines.append(f"{name} (separate): {r[k]} (step0={r.get(name+'_step0')}{dtxt})")
    if "health_final" in r:
        h = r["health_final"]
        mr = r.get("max_response_length")
        mrtxt = f" max_resp={mr}" if mr is not None else ""
        lines.append(
            f"health(final): kl={h.get('kl')} entropy={h.get('entropy')} "
            f"resp_len={h.get('resp_len')} reward={h.get('reward')} "
            f"parseable={r.get('parseable_final')}{mrtxt}"
        )
    traj = " ".join(f"{s}:{h}" for s, h in r["tom_hm_trajectory"])
    lines.append(f"ToM HM trajectory: {traj}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to the training .log")
    ap.add_argument("--last", type=int, default=5, help="window for capability-eval averaging (default 5)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    r = score(args.log, last_n=args.last)
    print(json.dumps(r, indent=2) if args.json else fmt_human(r))


if __name__ == "__main__":
    main()
