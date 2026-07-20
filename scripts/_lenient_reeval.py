"""Ad-hoc: re-eval base vs step-100 rule-based checkpoint, scoring STRICT vs VANILLA.

STRICT   = existing scorer (requires <think></think> structure; content skipped on format fail).
VANILLA  = extract final answer ignoring ALL tag structure, then apply the SAME correctness
           checkers. Answers the question: does ToM *accuracy* regress, or only the format score?
"""
import os, re, sys, json, io, contextlib
import pandas as pd
from collections import defaultdict
from statistics import harmonic_mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.reward_score import explore_tom, tom_mc, fantom
from verl.utils.reward_score.explore_tom import check_answer_correctness, normalize_answer
from verl.utils.reward_score.tom_mc import _check_mc_or_text
from verl.utils.reward_score.fantom import _check_binary, _check_list
from verl.utils.reward_score.response_parser import get_parser
from verl.trainer.main_ppo import _select_rm_score_fn

N_PER = int(os.environ.get("N_PER", "40"))
EVAL = "data/cleaned_tom/eval_subsample_300.parquet"
CKPT = "checkpoints/TOM_EXP/adhoc-rulebase-tom-ToM_train_HiEx_hint-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1/actor/global_step_100"
BASE = "Qwen/Qwen2.5-3B-Instruct"
CAP = ("gsm8k", "mmlu")

def parse_gt(rm):
    gt = rm.get("ground_truth", rm) if isinstance(rm, dict) else rm
    if isinstance(gt, str):
        try:
            g = json.loads(gt)
            if isinstance(g, dict):
                return g
        except Exception:
            pass
        return {"answer": gt, "answer_text": gt, "question_type": "exact", "choices": {}}
    return gt if isinstance(gt, dict) else {"answer": str(gt), "question_type": "exact", "choices": {}}

def vanilla_extract(resp):
    t = resp
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    for tok in ("<|im_end|>", "<|endoftext|>", "<think>"):
        t = t.replace(tok, "")
    m = re.findall(r"<answer>(.*?)</answer>", t, re.DOTALL)
    if m:
        return m[-1].strip()
    m2 = re.search(r"<answer>(.*)", t, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return t.strip()

def vanilla_correct(pred, g):
    if not pred:
        return False
    qt = g.get("question_type", "mc")
    gt_answer = g.get("answer", "")
    answer_text = g.get("answer_text", "")
    wrong = g.get("wrong_answer", "")
    choices = g.get("choices", {}) or {}
    try:
        if qt == "binary":
            return bool(_check_binary(pred, gt_answer))
        if qt == "list":
            return bool(_check_list(pred, gt_answer, wrong))
        if qt == "exact":
            ok, _ = check_answer_correctness(pred, gt_answer)
            return bool(ok)
        return bool(_check_mc_or_text(pred, gt_answer, answer_text, choices))
    except Exception:
        return False

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    df = pd.read_parquet(EVAL)
    # subsample N_PER per data_source
    parts = []
    for ds, grp in df.groupby("data_source"):
        parts.append(grp.head(N_PER))
    sub = pd.concat(parts).reset_index(drop=True)
    print(f"eval subset: {len(sub)} prompts, {sub['data_source'].nunique()} sources, N_PER={N_PER}")

    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = []
    for _, r in sub.iterrows():
        msgs = [dict(role=m["role"], content=m["content"]) for m in r["prompt"]]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(temperature=0.0, max_tokens=512)
    results = {}
    for tag, path in (("step0_base", BASE), ("step100_rule", CKPT)):
        print(f"\n=== generating {tag} ({path}) ===", flush=True)
        llm = LLM(model=path, tensor_parallel_size=4, gpu_memory_utilization=0.85,
                  dtype="bfloat16", max_model_len=3072, enforce_eager=True)
        outs = llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
        results[tag] = texts
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    parser = get_parser("Qwen/Qwen2.5-3B-Instruct")
    # score
    for tag in ("step0_base", "step100_rule"):
        texts = results[tag]
        strict = defaultdict(lambda: [0, 0])   # source -> [correct, total]
        vanilla = defaultdict(lambda: [0, 0])
        thinktag = defaultdict(lambda: [0, 0])
        prompt_texts = prompts
        for i, r in sub.iterrows():
            ds = r["data_source"]
            g = parse_gt(r["reward_model"])
            resp = texts[i]
            gt_arg = r["reward_model"]["ground_truth"] if isinstance(r["reward_model"], dict) else r["reward_model"]
            full = prompt_texts[i] + resp  # faithful: parser needs assistant marker
            fn, supports_parser = _select_rm_score_fn(ds)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    if supports_parser:
                        s = fn(solution_str=full, ground_truth=gt_arg, parser=parser)
                    else:
                        s = fn(solution_str=full, ground_truth=gt_arg)
                except Exception:
                    s = -3.0
            strict_ok = s >= 2.0  # answer reward achieved (only granted when format passed)
            strict[ds][1] += 1; strict[ds][0] += int(strict_ok)
            has_think = ("<think>" in resp and "</think>" in resp)
            thinktag[ds][1] += 1; thinktag[ds][0] += int(has_think)
            pred = vanilla_extract(resp)
            vok = vanilla_correct(pred, g)
            vanilla[ds][1] += 1; vanilla[ds][0] += int(vok)
        # aggregate: per-source accuracy, then ToM HM over 24 (excl gsm8k/mmlu)
        def acc(d, ds): return d[ds][0] / d[ds][1] if d[ds][1] else 0.0
        tom_sources = [ds for ds in strict if ds not in CAP]
        strict_accs = [max(acc(strict, ds), 1e-3) for ds in tom_sources]
        van_accs = [max(acc(vanilla, ds), 1e-3) for ds in tom_sources]
        print(f"\n########## {tag} ##########")
        print(f"{'source':32s} {'strict':>7s} {'vanilla':>8s} {'think%':>7s}")
        for ds in sorted(strict):
            print(f"{ds:32s} {acc(strict,ds):7.3f} {acc(vanilla,ds):8.3f} {acc(thinktag,ds):7.2f}")
        print(f"--- ToM(24) strict:  HM={harmonic_mean(strict_accs):.3f}  avg={sum(strict_accs)/len(strict_accs):.3f}")
        print(f"--- ToM(24) vanilla: HM={harmonic_mean(van_accs):.3f}  avg={sum(van_accs)/len(van_accs):.3f}")
        for cap in CAP:
            if cap in strict:
                print(f"--- {cap}: strict={acc(strict,cap):.3f} vanilla={acc(vanilla,cap):.3f}")

if __name__ == "__main__":
    main()
