#!/usr/bin/env python3
"""
Parquet Dataset Viewer — Interactive HTML viewer for inspecting training data.

Usage:
    python view_dataset.py data/merged_dialogue_datasets_16k.parquet
    python view_dataset.py data/tom_all_11k.parquet --port 8080
    python view_dataset.py data/tom_train.parquet --sample 100
"""

import argparse
import json
import html
import random
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import tempfile
import webbrowser
import os

import pandas as pd
import numpy as np


def make_serializable(obj):
    """Convert numpy/non-serializable types to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


def build_html(df: pd.DataFrame, filename: str) -> str:
    """Build a self-contained HTML viewer for the dataset."""

    # Prepare rows as JSON
    rows = []
    for idx, row in df.iterrows():
        r = {}
        for col in df.columns:
            r[col] = make_serializable(row[col])
        r['_index'] = int(idx)
        rows.append(r)

    data_json = json.dumps(rows)
    columns = list(df.columns)

    # Get unique data sources for filter
    data_sources = sorted(df['data_source'].unique().tolist()) if 'data_source' in df.columns else []

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Viewer — {html.escape(filename)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1a1a2e; color: #fff; padding: 16px 24px; position: sticky; top: 0; z-index: 100; }}
.header h1 {{ font-size: 18px; font-weight: 600; }}
.header .meta {{ font-size: 13px; color: #aaa; margin-top: 4px; }}
.controls {{ background: #fff; border-bottom: 1px solid #ddd; padding: 12px 24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; position: sticky; top: 60px; z-index: 99; }}
.controls label {{ font-size: 13px; font-weight: 500; color: #666; }}
.controls select, .controls input {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }}
.controls input[type=text] {{ width: 250px; }}
.controls input[type=number] {{ width: 70px; }}
.controls button {{ padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 500; }}
.btn-primary {{ background: #4361ee; color: #fff; }}
.btn-primary:hover {{ background: #3a56d4; }}
.btn-secondary {{ background: #e9ecef; color: #333; }}
.btn-secondary:hover {{ background: #dde1e5; }}
.stats {{ font-size: 13px; color: #666; margin-left: auto; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 16px 24px; }}
.card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px; overflow: hidden; }}
.card-header {{ background: #f8f9fa; padding: 10px 16px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }}
.card-header:hover {{ background: #f0f1f3; }}
.card-header .idx {{ font-weight: 700; color: #4361ee; font-size: 14px; }}
.card-header .badges {{ display: flex; gap: 6px; }}
.badge {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
.badge-source {{ background: #e3f2fd; color: #1565c0; }}
.badge-ability {{ background: #f3e5f5; color: #7b1fa2; }}
.badge-turn {{ background: #e8f5e9; color: #2e7d32; }}
.badge-words {{ background: #fff3e0; color: #e65100; }}
.card-body {{ padding: 16px; display: none; }}
.card-body.open {{ display: block; }}
.field {{ margin-bottom: 14px; }}
.field-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #999; font-weight: 600; margin-bottom: 4px; }}
.field-value {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.field-value.mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; background: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #eee; }}
.chat-msg {{ margin-bottom: 8px; padding: 8px 12px; border-radius: 6px; }}
.chat-msg.system {{ background: #fff3e0; border-left: 3px solid #ff9800; }}
.chat-msg.user {{ background: #e3f2fd; border-left: 3px solid #2196f3; }}
.chat-msg.assistant {{ background: #e8f5e9; border-left: 3px solid #4caf50; }}
.chat-role {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: #666; margin-bottom: 2px; }}
.chat-content {{ font-size: 13px; line-height: 1.5; white-space: pre-wrap; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.no-results {{ text-align: center; padding: 60px; color: #999; font-size: 16px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Dataset Viewer</h1>
  <div class="meta">{html.escape(filename)} — {len(df)} rows, {len(columns)} columns</div>
</div>

<div class="controls">
  <label>Source:</label>
  <select id="filterSource">
    <option value="">All</option>
    {"".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in data_sources)}
  </select>

  <label>Search:</label>
  <input type="text" id="searchBox" placeholder="Search prompts, answers...">

  <label>Show:</label>
  <input type="number" id="pageSize" value="50" min="1" max="5000">

  <button class="btn-primary" onclick="applyFilters()">Filter</button>
  <button class="btn-secondary" onclick="randomSample()">Random Sample</button>
  <button class="btn-secondary" onclick="expandAll()">Expand All</button>
  <button class="btn-secondary" onclick="collapseAll()">Collapse All</button>

  <span class="stats" id="stats"></span>
</div>

<div class="container" id="container"></div>

<script>
const DATA = {data_json};
let filtered = DATA;

function renderChat(prompt) {{
  if (!prompt || !Array.isArray(prompt)) return '<em>N/A</em>';
  return prompt.map(m => {{
    const role = m.role || 'unknown';
    const content = escapeHtml(m.content || '');
    return `<div class="chat-msg ${{role}}"><div class="chat-role">${{role}}</div><div class="chat-content">${{content}}</div></div>`;
  }}).join('');
}}

function escapeHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

function renderCard(row, i) {{
  const meta = row.metadata || {{}};
  const turn = meta.turn !== undefined ? meta.turn : '—';
  const gt = row.reward_model ? (row.reward_model.ground_truth || '') : '';

  return `
    <div class="card">
      <div class="card-header" onclick="this.nextElementSibling.classList.toggle('open')">
        <span class="idx">#${{row._index}}</span>
        <div class="badges">
          ${{row.data_source ? `<span class="badge badge-source">${{escapeHtml(row.data_source)}}</span>` : ''}}
          ${{row.ability ? `<span class="badge badge-ability">${{escapeHtml(row.ability)}}</span>` : ''}}
          <span class="badge badge-turn">turn ${{turn}}</span>
          ${{row.response_words ? `<span class="badge badge-words">${{row.response_words}} words</span>` : ''}}
        </div>
      </div>
      <div class="card-body">
        <div class="field">
          <div class="field-label">Prompt (Chat Format)</div>
          <div class="field-value">${{renderChat(row.prompt || row.raw_prompt)}}</div>
        </div>
        <div class="two-col">
          <div class="field">
            <div class="field-label">Ground Truth Answer</div>
            <div class="field-value mono">${{escapeHtml(gt)}}</div>
          </div>
          <div class="field">
            <div class="field-label">Answer PP</div>
            <div class="field-value mono">${{row.answer_pp !== null && row.answer_pp !== undefined ? row.answer_pp.toFixed(4) : 'N/A'}}</div>
          </div>
        </div>
        ${{row.raw_system_prompt ? `
        <div class="field">
          <div class="field-label">System Prompt</div>
          <div class="field-value mono">${{escapeHtml(row.raw_system_prompt)}}</div>
        </div>` : ''}}
        ${{row.raw_user_prompt ? `
        <div class="field">
          <div class="field-label">User Prompt</div>
          <div class="field-value mono">${{escapeHtml(row.raw_user_prompt)}}</div>
        </div>` : ''}}
        ${{row.generation_prefix ? `
        <div class="field">
          <div class="field-label">Generation Prefix</div>
          <div class="field-value mono">${{escapeHtml(row.generation_prefix)}}</div>
        </div>` : ''}}
        <div class="field">
          <div class="field-label">Metadata</div>
          <div class="field-value mono">${{escapeHtml(JSON.stringify(meta, null, 2))}}</div>
        </div>
      </div>
    </div>`;
}}

function render() {{
  const container = document.getElementById('container');
  const pageSize = parseInt(document.getElementById('pageSize').value) || 50;
  const showing = filtered.slice(0, pageSize);

  if (showing.length === 0) {{
    container.innerHTML = '<div class="no-results">No results match your filters.</div>';
  }} else {{
    container.innerHTML = showing.map((r, i) => renderCard(r, i)).join('');
  }}

  document.getElementById('stats').textContent =
    `Showing ${{showing.length}} of ${{filtered.length}} (total: ${{DATA.length}})`;
}}

function applyFilters() {{
  const source = document.getElementById('filterSource').value;
  const search = document.getElementById('searchBox').value.toLowerCase();

  filtered = DATA.filter(row => {{
    if (source && row.data_source !== source) return false;
    if (search) {{
      const text = JSON.stringify(row).toLowerCase();
      if (!text.includes(search)) return false;
    }}
    return true;
  }});

  render();
}}

function randomSample() {{
  const pageSize = parseInt(document.getElementById('pageSize').value) || 50;
  const shuffled = [...filtered].sort(() => Math.random() - 0.5);
  filtered = shuffled;
  render();
}}

function expandAll() {{
  document.querySelectorAll('.card-body').forEach(el => el.classList.add('open'));
}}

function collapseAll() {{
  document.querySelectorAll('.card-body').forEach(el => el.classList.remove('open'));
}}

// Initial render
render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Interactive HTML viewer for parquet datasets")
    parser.add_argument("parquet_file", help="Path to parquet file")
    parser.add_argument("--port", type=int, default=8888, help="Port for HTTP server (default: 8888)")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows (for large datasets)")
    parser.add_argument("--output", type=str, default=None, help="Save HTML to file instead of serving")
    parser.add_argument("--no-serve", action="store_true", help="Don't start HTTP server, just save HTML")
    args = parser.parse_args()

    print(f"Loading {args.parquet_file}...")
    df = pd.read_parquet(args.parquet_file)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    if args.sample and args.sample < len(df):
        print(f"  Sampling {args.sample} rows...")
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    filename = os.path.basename(args.parquet_file)
    print("Building HTML...")
    html_content = build_html(df, filename)

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(tempfile.gettempdir(), f"dataset_viewer_{filename}.html")

    with open(out_path, 'w') as f:
        f.write(html_content)
    print(f"  Saved to {out_path}")

    if args.no_serve or args.output:
        print(f"Open in browser: file://{os.path.abspath(out_path)}")
        return

    # Serve
    os.chdir(os.path.dirname(out_path))
    html_filename = os.path.basename(out_path)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # suppress logs

    print(f"Serving at http://localhost:{args.port}/{html_filename}")
    print("Press Ctrl+C to stop")

    try:
        server = HTTPServer(('', args.port), Handler)
        webbrowser.open(f"http://localhost:{args.port}/{html_filename}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
