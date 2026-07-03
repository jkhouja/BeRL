#!/usr/bin/env bash
#
# Apply Qwen3 support patches to the installed vllm 0.6.3 package.
#
# vllm 0.6.3 predates Qwen3 and has no Qwen3ForCausalLM in its ModelRegistry,
# so a fresh `pip install vllm==0.6.3` will CRASH at model-build time for Qwen3
# models with:
#   ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported for now.
#
# This script patches the *installed* vllm site-packages (NOT this repo) to:
#   1. Remap Qwen3ForCausalLM -> vllm's Qwen2ForCausalLM class (registry.py).
#   2. Add per-head QK-norm (q_norm/k_norm) + configurable attention_bias to the
#      Qwen2 model, activated when config.model_type == 'qwen3' (qwen2.py).
#
# The complementary weight-loading / parsing logic already lives in this repo
# (verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py, etc.), but the
# build-time remap and QK-norm layer creation MUST be patched into vllm itself.
#
# Qwen2.5 and Gemma2 do NOT need this patch. Run once per environment after
# installing vllm==0.6.3. The script is idempotent.
#
# Usage:
#   conda activate tom
#   bash patches/apply_vllm_qwen3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/vllm_0_6_3_qwen3.patch"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file not found: $PATCH_FILE" >&2
    exit 1
fi

# Locate the installed vllm package directory.
VLLM_DIR="$(python -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"
if [[ -z "$VLLM_DIR" || ! -d "$VLLM_DIR" ]]; then
    echo "ERROR: could not locate installed vllm package. Is vllm installed in this env?" >&2
    exit 1
fi

# Patch paths are 'a/vllm/...', so apply with -p1 from the site-packages root
# (the parent of the vllm/ directory).
SITE_ROOT="$(dirname "$VLLM_DIR")"
echo "vllm package : $VLLM_DIR"
echo "apply root   : $SITE_ROOT"

cd "$SITE_ROOT"

# Idempotency: if the patch is already applied, a forward dry-run reports it as
# already-applied (reverse would succeed). Detect and skip in that case.
if patch -p1 -R --dry-run --force < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "Patch already applied — nothing to do."
    exit 0
fi

if ! patch -p1 --dry-run --force < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "ERROR: patch does not apply cleanly to this vllm install." >&2
    echo "       Expected stock vllm==0.6.3. Check the installed version." >&2
    exit 1
fi

patch -p1 < "$PATCH_FILE"
echo "Qwen3 vllm patch applied successfully."

# Verify the registry remap took effect.
python - <<'PY'
from vllm.model_executor.models import ModelRegistry
try:
    cls, name = ModelRegistry.resolve_model_cls(["Qwen3ForCausalLM"])
    print(f"Verified: Qwen3ForCausalLM -> {cls.__name__}")
except Exception as e:
    raise SystemExit(f"Verification FAILED: {e}")
PY
