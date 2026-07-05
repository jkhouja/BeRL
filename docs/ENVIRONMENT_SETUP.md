# Environment Setup

This guide explains how to set up a working Python environment for BeRL/TomRL
(GRPO training of small LLMs for Theory of Mind via dialogue data) on a new
machine.

## TL;DR

There is **one canonical environment: the conda env named `tom`**. All experiment
scripts (`experiments/*.sh`) activate it with `conda activate tom`. It supports
**Qwen2.5, Qwen3, and Gemma-2** (Qwen3 needs one extra vllm patch — see below).

```bash
conda create -n tom python=3.10
conda activate tom

# Core stack (CUDA 12.1 build of torch)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 ray
pip install flash-attn --no-build-isolation

# Install verl (this repo) in editable mode
pip install -e .

# Newer transformers is REQUIRED for Qwen3 / Gemma2 support (see note below)
pip install "transformers==4.51.3"

# Extras used by training/eval/plotting
pip install wandb IPython matplotlib
```

> **Running Qwen3?** You also need to patch the installed vllm (it predates
> Qwen3). See [Qwen3 requires an extra vllm patch](#️-qwen3-requires-an-extra-vllm-patch)
> below. **Qwen2.5 and Gemma2 do not need it.**

After installing, sanity-check the env:

```bash
conda activate tom
python -c "import verl, torch, transformers, vllm; \
print('verl', verl.__file__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__, 'vllm', vllm.__version__)"
```

`verl.__file__` should point at `<repo>/verl/__init__.py` (editable install from
this checkout), and `torch.cuda.is_available()` should be `True` on a GPU machine.

## ⚠️ Important: transformers version vs `requirements.txt`

`requirements.txt` pins `transformers<4.48` and `setup.py` installs from it, so a
plain `pip install -e .` will give you an **old transformers that cannot load
Qwen3 or Gemma2**. The canonical `tom` env runs **transformers 4.51.3**, which
supports Qwen2.5, Qwen3, and Gemma-2.

Always upgrade transformers after `pip install -e .`:

```bash
pip install "transformers==4.51.3"
```

## ⚠️ Qwen3 requires an extra vllm patch

vllm 0.6.3 predates Qwen3 and has **no `Qwen3ForCausalLM` in its `ModelRegistry`**.
A fresh `pip install vllm==0.6.3` will therefore **crash at model-build time** for
Qwen3 models, before any weight loading:

```
ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported for now.
```

The repo code (`verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py`,
`verl/models/transformers/qwen3.py`, the response parser, etc.) handles the
weight-sync / attention / parsing half of Qwen3 support, but the **build-time
architecture remap and QK-norm layer creation must be patched into the installed
vllm package itself**. Those patches are *not* installed by pip and are *not* part
of `verl`; they live in `patches/`:

```bash
conda activate tom
bash patches/apply_vllm_qwen3.sh
```

This patches the installed vllm site-packages to (1) remap
`Qwen3ForCausalLM → Qwen2ForCausalLM` in the registry, and (2) add per-head
QK-norm (`q_norm`/`k_norm`) plus configurable `attention_bias` to vllm's Qwen2
model, activated when `config.model_type == 'qwen3'`. The script is idempotent
and verifies the remap afterward.

**Qwen2.5 and Gemma2 do NOT need this patch** — only Qwen3. Run it once per
environment, after installing `vllm==0.6.3`. The exact diff is captured in
`patches/vllm_0_6_3_qwen3.patch` so it is reproducible on any machine.

## Verified package versions (canonical `tom` env)

These are the versions known to work together for current experiments:

| Package | Version |
|---------|---------|
| python | 3.10 |
| torch | 2.4.0+cu121 |
| vllm | 0.6.3 |
| flash-attn | 2.8.3 |
| transformers | 4.51.3 |
| tensordict | 0.10.0 |
| ray | 2.44.1 |
| accelerate | 1.11.0 |
| datasets | 4.3.0 |
| hydra-core | 1.3.2 |
| wandb | 0.22.2 |

A full frozen pip list from a reference machine is available in
`pip_env_virgil.txt` at the repo root if you need exact transitive pins.

## Running experiments

Once `tom` is set up:

```bash
conda activate tom
bash experiments/qwen2.5_smoke_test.sh   # quick smoke test
# or any other script under experiments/
```

Each script calls `conda activate tom` itself, so as long as conda is initialized
in your shell the scripts are self-contained.

## Hardware notes

- Training configs assume NVIDIA A100 80G GPUs (e.g. `experiments/tom_grpo_power_reward.sh`
  targets 4×A100 80G). torch is the CUDA 12.1 build, so the host needs a compatible NVIDIA driver.
- `flash-attn` is built from source (`--no-build-isolation`) and needs a working
  CUDA toolchain at install time.
