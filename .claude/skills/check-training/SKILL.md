---
name: check-training
description: Check the status of a running or completed training experiment. Use when the user asks to check training, check status, how's the run going, or check on the experiment.
allowed-tools: Bash Read Grep Glob
---

## Instructions

1. Find the most recent log file(s):
   ```!
   ls -lt logs/$(date +%Y%m%d)/ 2>/dev/null | head -5
   ```

2. Extract eval scores from the log by grepping for `val/test_score` lines
3. Parse out step number, tomi, explore_tom, and hi_tom scores
4. Present results as a markdown table
5. Note the trend: improving, flat, or degrading
6. If running, also check GPU utilization with `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
7. Compare against baselines:
   - **3B baseline**: tomi=63.3%, explore=47.0%, hi_tom=19.3%
   - **3B 17f peak**: tomi=69.4%, explore=66.9%, hi_tom=29.8%
   - **7B baseline**: tomi=67.8%, explore=73.0%, hi_tom=42.9%
