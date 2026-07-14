#!/usr/bin/env python3
"""Comprehensive, eval-grounded re-assessment of all Completed BeRL runs.

Ranks every Completed phase-stability run on **authoritative eval accuracy**
(`val/test_score/<bench>_sub300`, which is accuracy in [0,1]) instead of training
reward / rollout response-length, which the PS129 investigation showed are
misleading (a run can hack the *training* behavior objective — rollout resp_len
collapses, KL blows up — while its *eval* CoTs stay coherent and ToM accuracy
genuinely rises; and conversely a short-output policy can win MC benches by luck).

Design (see notebooks/BeRL_run_reassessment.ipynb for the narrative):

  * Primary score  = ToM HM over the 24 ToM benchmarks (excl gsm8k, mmlu),
                     avg-then-HM over the last-N eval iters (matches score_run.py).
  * Ranking metric = dHM = HM(last3) - HM(step0), computed vs each run's OWN
                     step-0 baseline and ranked WITHIN model family (Qwen2.5 /
                     Qwen3 / Gemma-2 have very different baselines).
  * Stability      = std of the per-iter ToM HM over the last-5 evals (prefer a
                     sustained plateau over a lucky one-step spike).
  * Regression     = dgsm8k, dmmlu reported separately (never folded into HM).
  * Eval-quality gate (degeneracy) = from `[val sample | step | score]` blocks,
                     the fraction of eval samples that are format-parseable and
                     the fraction fully correct, over the last evals. Catches the
                     "empty-CoT wins binary MC by luck" failure that test_score
                     alone cannot (short outputs -> low eval_correct but maybe
                     ~0.5 MC accuracy).
  * Training-stability annotation = final KL (flag if pinned near the ~10 cap),
                     min rollout resp_len (collapse), resp_len re-explosion. These
                     ANNOTATE, they do not auto-disqualify (PS129 lesson).

Usage:
  python scripts/reassess_runs.py --out analysis/reassess.csv            # all Completed
  python scripts/reassess_runs.py --logs a.log b.log --json              # ad-hoc
  python scripts/reassess_runs.py --top 15                               # print leaderboard
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from statistics import harmonic_mean, mean, pstdev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER = os.path.join(REPO, "project_planning", "BeRL_experiments_tracker.md")
CACHE_DIR = os.path.join(REPO, "analysis", "cache")

CAPABILITY_EVALS = ("gsm8k", "mmlu")  # reported separately, never inside the HM
HM_EPS = 1e-3

VAL_RE = re.compile(r"val/test_score/(\w+?)_sub300:([0-9.]+)")
STEP_RE = re.compile(r"step:(\d+) -")
HEALTH_RE = re.compile(r"step:(\d+) - global_seqlen")
VSAMPLE_RE = re.compile(r"\[val sample \| step=(\d+) \| source=(\w+) \| score=(-?[0-9.]+)\]")
USE_ACTOR_RE = re.compile(r"use_actor_as_rm['\"]?\s*[:=]\s*(True|False)")
HKEYS = {
    "reward": r"reward/mean:(-?[0-9.]+)",
    "resp_len": r"response_length/mean:(-?[0-9.]+)",
    "entropy": r"actor/entropy_loss:(-?[0-9.]+)",
    "kl": r"actor/kl_loss:(-?[0-9.]+)",
    "fmt_err": r"reward/format_error_ratio:(-?[0-9.]+)",
}

# lines worth keeping when we build the compact per-log cache
RG_PATTERN = (
    r"val/test_score/[a-z0-9_]+_sub300:[0-9.]+"
    r"|step:[0-9]+ - global_seqlen"
    r"|\[val sample \| step="
    r"|use_actor_as_rm"
)


def hmean(scores):
    return harmonic_mean([max(v, HM_EPS) for v in scores]) if scores else float("nan")


# --------------------------------------------------------------------------- #
# Extraction (rg-accelerated with a pure-python fallback), cached to JSON.
# --------------------------------------------------------------------------- #
def _iter_relevant_lines(path):
    """Yield only lines we care about, using rg if available else python."""
    try:
        p = subprocess.Popen(
            ["rg", "-aN", RG_PATTERN, path],
            stdout=subprocess.PIPE, text=True, errors="ignore",
        )
        for line in p.stdout:
            yield line
        p.wait()
    except FileNotFoundError:
        needles = ("val/test_score/", "global_seqlen", "[val sample | step=", "use_actor_as_rm")
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if any(n in line for n in needles):
                    yield line


def extract(path):
    """Parse a log into a compact dict of everything the metrics need."""
    iters = {}          # step -> {bench: acc}
    health = {}         # step -> {kl,resp_len,reward,fmt_err,entropy}
    vsamples = {}       # step -> list[score]
    use_actor = None
    for line in _iter_relevant_lines(path):
        if "use_actor_as_rm" in line and use_actor is None:
            m = USE_ACTOR_RE.search(line)
            if m:
                use_actor = (m.group(1) == "True")
        vs = VSAMPLE_RE.search(line)
        if vs:
            vsamples.setdefault(int(vs.group(1)), []).append(float(vs.group(3)))
            continue
        sm = STEP_RE.search(line)
        if not sm:
            continue
        step = int(sm.group(1))
        scores = {k: float(v) for k, v in VAL_RE.findall(line)}
        if scores:
            if step not in iters or len(scores) > len(iters[step]):
                iters[step] = scores
        if HEALTH_RE.search(line):
            rec = {}
            for name, pat in HKEYS.items():
                mm = re.search(pat, line)
                if mm:
                    rec[name] = float(mm.group(1))
            if rec:
                health[step] = rec
    return {
        "log": path,
        "use_actor_as_rm": use_actor,
        "iters": sorted(iters.items()),
        "health": sorted(health.items()),
        "vsamples": sorted(vsamples.items()),
    }


def load_cached(path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = os.path.basename(path)[:-4]
    cache = os.path.join(CACHE_DIR, stem + ".json")
    try:
        if os.path.getmtime(cache) >= os.path.getmtime(path):
            with open(cache) as fh:
                return json.load(fh)
    except OSError:
        pass
    data = extract(path)
    with open(cache, "w") as fh:
        json.dump(data, fh)
    return data


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def window_avg(iters, benches, n):
    window = iters[-n:]
    if not window:
        return {}
    present = set.intersection(*[set(s) for _, s in window])
    keys = [b for b in benches if b in present]
    return {b: sum(s[b] for _, s in window) / len(window) for b in keys}


def compute_metrics(data):
    iters = [(s, d) for s, d in data["iters"]]
    if len(iters) < 2:
        return None
    all_b = sorted({b for _, s in iters for b in s})
    tom = [b for b in all_b if b not in CAPABILITY_EVALS]
    traj = [(s, hmean([d[b] for b in tom if b in d])) for s, d in iters]

    def tom_hm(n):
        a = window_avg(iters, tom, n)
        return hmean(list(a.values())) if a else float("nan")

    def tom_avg(n):
        a = window_avg(iters, tom, n)
        return mean(a.values()) if a else float("nan")

    step0 = iters[0][1]
    hm0 = traj[0][1]
    avg0 = mean([step0[b] for b in tom if b in step0])
    late = [h for _, h in traj[-5:]]
    m = {
        "n_iters": len(iters),
        "first_step": iters[0][0],
        "last_step": iters[-1][0],
        "n_tom": len(tom),
        "hm_step0": round(hm0, 4),
        "hm_last3": round(tom_hm(3), 4),
        "hm_last5": round(tom_hm(5), 4),
        "hm_best": round(max(h for _, h in traj), 4),
        "d_hm": round(tom_hm(3) - hm0, 4),
        "hm_late_std": round(pstdev(late), 4) if len(late) > 1 else 0.0,
        "avg_step0": round(avg0, 4),
        "avg_last3": round(tom_avg(3), 4),
        "d_avg": round(tom_avg(3) - avg0, 4),
    }
    for name in CAPABILITY_EVALS:
        base = step0.get(name)
        vals = [s[name] for _, s in iters[-3:] if name in s]
        last = sum(vals) / len(vals) if vals else None
        m[f"{name}_step0"] = round(base, 4) if base is not None else None
        m[f"{name}_last3"] = round(last, 4) if last is not None else None
        if base is not None and last is not None:
            m[f"d_{name}"] = round(last - base, 4)

    # eval-quality gate from [val sample] scores (format-pass = score>=0, correct = max)
    vs = data.get("vsamples", [])
    if vs:
        last_steps = vs[-3:]
        allsc = [x for _, lst in last_steps for x in lst]
        if allsc:
            mx = max(allsc)
            m["eval_parseable"] = round(sum(1 for x in allsc if x >= 0) / len(allsc), 3)
            m["eval_correct"] = round(sum(1 for x in allsc if x >= mx - 1e-6) / len(allsc), 3)
        base0 = dict(vs).get(iters[0][0]) or (vs[0][1] if vs else [])
        if base0:
            mx0 = max(base0)
            m["eval_correct_step0"] = round(sum(1 for x in base0 if x >= mx0 - 1e-6) / len(base0), 3)

    # training-stability annotation from health
    health = [(s, d) for s, d in data["health"]]
    if health:
        hf = health[-1][1]
        rl = [d.get("resp_len") for _, d in health if "resp_len" in d]
        kl = [d.get("kl") for _, d in health if "kl" in d]
        m["kl_final"] = round(hf.get("kl", float("nan")), 3)
        m["kl_max"] = round(max(kl), 3) if kl else None
        m["resp_len_final"] = round(hf.get("resp_len", float("nan")), 1)
        m["resp_len_min"] = round(min(rl), 1) if rl else None
        m["reward_final"] = round(hf.get("reward", float("nan")), 3)
    return m


# --------------------------------------------------------------------------- #
# Knob parsing from run-name / EXP_ID + config-dump use_actor_as_rm
# --------------------------------------------------------------------------- #
def parse_knobs(stem, use_actor):
    s = stem
    def find(pat, default=None, cast=str):
        m = re.search(pat, s)
        return cast(m.group(1)) if m else default
    model = ("Qwen2.5" if "Qwen2.5" in s or "qwen2.5" in s else
             "Qwen3" if "qwen3" in s or "Qwen3" in s else
             "Gemma-2" if "gemma" in s.lower() else "?")
    if "log_prob" in s or "-lp-" in s or "_lp-" in s or s.endswith("_lp"):
        reward = "log_prob"
    elif "neg_perplex" in s or "negppl" in s:
        reward = "neg_perplexity"
    elif "power" in s:
        reward = "power"
    else:
        reward = "?"
    return {
        "model": model,
        "reward": reward,
        "power_k": find(r"[-_]k(\d+(?:\.\d+)?)", None, float),
        "ll_min": find(r"llm(?:in)?[-_]?(-?\d+(?:\.\d+)?)", None, float),
        "lr": find(r"lr([0-9.]+e-?\d+)", None),
        "kl": find(r"kl([0-9.]+)", None, float),
        "ec": find(r"_ec([0-9.]+)", None, float),
        "fp": find(r"_fp([0-9.]+)", None, float),
        "baseline": "nobaseline" not in s,
        "rm_mode": ("actor" if use_actor else "frozen") if use_actor is not None else "?",
    }


# --------------------------------------------------------------------------- #
# Tracker resolution
# --------------------------------------------------------------------------- #
def parse_tracker():
    rows = [l for l in open(TRACKER, encoding="utf-8") if l.strip().startswith("|")]
    hdr = None
    recs = []
    for l in rows:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if hdr is None:
            if "Status" in cells:
                hdr = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if len(cells) != len(hdr):
            continue
        recs.append(dict(zip(hdr, cells)))
    return recs


def resolve_logs(recs, status="completed"):
    disk = glob.glob(os.path.join(REPO, "logs", "**", "*.log"), recursive=True)
    stem_to_path = {os.path.basename(p)[:-4]: p for p in disk}
    out = []
    for r in recs:
        if r["Status"].lower() != status:
            continue
        run = r["Run name"].strip()
        cands = [p for stem, p in stem_to_path.items() if stem.startswith(run)]
        if not cands:
            continue
        out.append((r, cands))
    return out


def pick_best_attempt(cands):
    """Among retry attempts, pick the log with the most eval iters."""
    best, best_n = None, -1
    for p in cands:
        try:
            d = load_cached(p)
        except Exception:
            continue
        n = len(d.get("iters", []))
        if n > best_n:
            best, best_n = p, n
    return best


# --------------------------------------------------------------------------- #
def assess_all(status="completed"):
    recs = parse_tracker()
    resolved = resolve_logs(recs, status)
    out = []
    for i, (r, cands) in enumerate(resolved):
        path = pick_best_attempt(cands) if len(cands) > 1 else cands[0]
        data = load_cached(path)
        m = compute_metrics(data)
        stem = os.path.basename(path)[:-4]
        knobs = parse_knobs(stem, data.get("use_actor_as_rm"))
        row = {"exp_id": r.get("Exp ID", ""), "run": r["Run name"],
               "log": os.path.relpath(path, REPO), **knobs}
        if m:
            row.update(m)
        else:
            row["n_iters"] = 0
        out.append(row)
        print(f"[{i+1}/{len(resolved)}] {stem[:70]}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", help="score specific logs instead of the tracker")
    ap.add_argument("--out", help="write full results CSV")
    ap.add_argument("--top", type=int, default=15, help="print top-N per family")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.logs:
        rows = []
        for p in args.logs:
            d = load_cached(p)
            m = compute_metrics(d) or {}
            rows.append({"log": p, **parse_knobs(os.path.basename(p)[:-4],
                                                 d.get("use_actor_as_rm")), **m})
    else:
        rows = assess_all()

    if args.json:
        print(json.dumps(rows, indent=2))
    if args.out:
        import csv
        cols = sorted({k for r in rows for k in r})
        head = [c for c in ["exp_id", "run", "model", "reward", "power_k", "ll_min",
                            "rm_mode", "kl", "lr", "ec", "fp", "baseline",
                            "d_hm", "d_avg", "hm_step0", "hm_last3", "hm_best",
                            "hm_late_std", "eval_correct", "eval_parseable",
                            "d_gsm8k", "d_mmlu", "kl_final", "resp_len_min",
                            "n_iters", "log"] if c in cols]
        head += [c for c in cols if c not in head]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)

    # leaderboard per family
    scored = [r for r in rows if r.get("n_iters", 0) >= 2 and "d_hm" in r]
    for fam in ("Qwen2.5", "Qwen3", "Gemma-2"):
        fr = sorted([r for r in scored if r["model"] == fam],
                    key=lambda r: r["d_hm"], reverse=True)
        print(f"\n===== {fam}: top {args.top} by dHM (HM last3 - step0) =====")
        print(f"{'dHM':>7} {'dAvg':>7} {'hm0':>6} {'hm3':>6} {'std':>6} "
              f"{'ecorr':>5} {'dgsm':>6} {'reward':>7} {'kl_f':>6} {'rl_min':>6}  run")
        for r in fr[:args.top]:
            print(f"{r['d_hm']:>7.3f} {r.get('d_avg',0):>7.3f} {r['hm_step0']:>6.3f} "
                  f"{r['hm_last3']:>6.3f} {r.get('hm_late_std',0):>6.3f} "
                  f"{r.get('eval_correct','-'):>5} {r.get('d_gsm8k',0):>6.3f} "
                  f"{r.get('reward_final',0):>7.2f} {r.get('kl_final',0):>6.2f} "
                  f"{str(r.get('resp_len_min','-')):>6}  {r['run'][:52]}")


if __name__ == "__main__":
    main()
