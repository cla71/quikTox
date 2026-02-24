#!/usr/bin/env python3
"""
generate_report.py  --  OFFTOXv3 Validation Report Generator
=============================================================
Interprets output files from the OFFTOXv3 pipeline:
  - outputs/standalone_predictions.csv  (novel compound predictions from workflow)
  - outputs/analysis_report.md          (training/validation summary from workflow)
  - outputs/workflow_summary.json       (machine-readable metrics)
  - model/safety_model.pkl              (optional; for SAS core analysis)

Produces a fully self-contained interactive HTML report at:
  outputs/validation_report.html

Run after executing the notebook via run_pipeline.py:
    python generate_report.py
"""

import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STANDALONE_CSV = BASE_DIR / "outputs" / "standalone_predictions.csv"
SUMMARY_JSON   = BASE_DIR / "outputs" / "workflow_summary.json"
ANALYSIS_MD    = BASE_DIR / "outputs" / "analysis_report.md"
MODEL_PKL      = BASE_DIR / "model" / "safety_model.pkl"
OUTPUT_HTML    = BASE_DIR / "outputs" / "validation_report.html"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "non_binding": "#2ecc71",
    "binding":     "#e74c3c",
}

LABEL_NICE = {
    "non_binding": "Non-Binding",
    "binding":     "Binding",
}


def _pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "N/A"


def _plotly_html(fig, height=None):
    if height:
        fig.update_layout(height=height)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _md_to_html(md_text: str) -> str:
    """Minimal Markdown-to-HTML converter for analysis_report.md content."""
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    in_code = False

    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line)
            continue

        # Tables
        if line.strip().startswith("|"):
            if not in_table:
                html_lines.append('<div class="table-wrap"><table>')
                in_table = True
            # Skip separator rows (|---|---|)
            if re.match(r"^\s*\|[\s\-|:]+\|\s*$", line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Detect header row: contains bold markers or is first row
            is_header = any(c.startswith("**") or c.startswith("#") for c in cells)
            tag = "th" if is_header else "td"
            row_html = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row_html}</tr>")
            continue
        else:
            if in_table:
                html_lines.append("</table></div>")
                in_table = False

        # Headings
        if line.startswith("#### "):
            html_lines.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h3>{line[3:]}</h3>")
        elif line.startswith("# "):
            html_lines.append(f"<h2>{line[2:]}</h2>")
        # Lists
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            content = line.strip()[2:]
            # Bold inline
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        # Blank lines
        elif line.strip() == "":
            if in_table:
                html_lines.append("</table></div>")
                in_table = False
            html_lines.append("<br>")
        # Normal paragraphs
        else:
            content = line
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html_lines.append(f"<p>{content}</p>")

    if in_table:
        html_lines.append("</table></div>")

    return "\n".join(html_lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load pipeline output files. Returns (preds_rows, summary, md_text, artifacts)."""
    # --- standalone predictions ---
    if not STANDALONE_CSV.exists():
        print(f"  ERROR: standalone_predictions.csv not found at {STANDALONE_CSV}")
        print("  Run the notebook (or run_pipeline.py) first to generate predictions.")
        sys.exit(1)

    import csv
    preds_rows = []
    with STANDALONE_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            preds_rows.append(row)
    print(f"  Standalone predictions: {len(preds_rows)} compound-target rows")

    # --- workflow summary JSON (may be malformed) ---
    summary = {}
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON) as fh:
                raw = fh.read()
            # Attempt repair: truncate at last valid top-level key if needed
            try:
                summary = json.loads(raw)
            except json.JSONDecodeError:
                # Try to load what we can by truncating at last complete value
                last_brace = raw.rfind("},")
                if last_brace > 0:
                    try:
                        summary = json.loads(raw[:last_brace] + "}")
                    except Exception:
                        pass
            if summary:
                print(f"  Summary model: {summary.get('best_model', 'N/A')}")
            else:
                print(f"  WARNING: workflow_summary.json could not be parsed — using defaults")
        except Exception as e:
            print(f"  WARNING: Could not read workflow_summary.json: {e}")
    else:
        print(f"  WARNING: workflow_summary.json not found — using defaults")

    # --- analysis report markdown ---
    md_text = ""
    if ANALYSIS_MD.exists():
        md_text = ANALYSIS_MD.read_text(encoding="utf-8")
        print(f"  Analysis report: {len(md_text)} chars")
    else:
        print(f"  WARNING: analysis_report.md not found at {ANALYSIS_MD}")

    # --- model artifacts (optional) ---
    artifacts = {}
    if MODEL_PKL.exists():
        try:
            with open(MODEL_PKL, "rb") as fh:
                artifacts = pickle.load(fh)
            print(f"  Model artifact loaded: {MODEL_PKL.name}")
        except Exception as e:
            print(f"  WARNING: Could not load model pkl: {e}")
    else:
        print(f"  WARNING: model pkl not found at {MODEL_PKL} — SAS section will be empty")

    return preds_rows, summary, md_text, artifacts


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_header(summary):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tm = summary.get("test_metrics", {})
    n_compounds = summary.get("n_compounds", "N/A")
    n_targets = len(summary.get("targets", []))
    best_model = summary.get("best_model", "N/A")
    top3 = ", ".join(summary.get("top3_models", []))
    method = summary.get("model_selection_method", "MCDA composite")

    return f"""
    <div class="header">
      <h1>OFFTOXv3 Validation Report &mdash; Safety Pharmacology &amp; NHR Predictions</h1>
      <p class="sub">Generated {ts} &nbsp;|&nbsp;
         Best Model: <strong>{best_model}</strong> &nbsp;|&nbsp;
         Top-3 Consensus: <strong>{top3 if top3 else 'N/A'}</strong></p>
      <p class="sub">Selection: {method}</p>
    </div>

    <div class="card">
      <h2>Training Summary</h2>
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-value">{n_compounds}</span>
          <span class="metric-label">Training compounds</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_targets}</span>
          <span class="metric-label">Targets</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{summary.get('train_size','?')} / {summary.get('val_size','?')} / {summary.get('test_size','?')}</span>
          <span class="metric-label">Train / Val / Test split</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{tm.get('roc_auc_macro', 0):.4f}</span>
          <span class="metric-label">Test ROC-AUC</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{tm.get('mcc', 0):.4f}</span>
          <span class="metric-label">Test MCC</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{tm.get('ece', 0):.4f}</span>
          <span class="metric-label">ECE (calibration)</span>
        </div>
      </div>
      <p><em>Class distribution — Non-Binding: {summary.get('class_distribution',{}).get('non_binding','?')}&nbsp;|&nbsp;
         Binding: {summary.get('class_distribution',{}).get('binding','?')}</em></p>
    </div>
    """


def build_analysis_md_section(md_text: str) -> str:
    """Section: embed analysis_report.md content as HTML."""
    if not md_text:
        return """
    <div class="card">
      <h2>Workflow Analysis Report</h2>
      <p>analysis_report.md not found. Run the notebook to generate it.</p>
    </div>
    """
    html_content = _md_to_html(md_text)
    return f"""
    <div class="card">
      <h2>Workflow Analysis Report</h2>
      <p><em>Content from <code>outputs/analysis_report.md</code> generated by the notebook.</em></p>
      <div class="md-content">
        {html_content}
      </div>
    </div>
    """


def build_prediction_summary_section(preds_rows: list, summary: dict) -> str:
    """Section: high-level summary of standalone predictions."""
    if not preds_rows:
        return "<div class='card'><h2>Prediction Summary</h2><p>No predictions found.</p></div>"

    # Aggregate stats
    n_total = len(preds_rows)
    compound_ids = set(r["compound_id"] for r in preds_rows)
    n_compounds = len(compound_ids)
    targets = sorted(set(r["target"] for r in preds_rows))
    n_targets = len(targets)

    n_active_best = sum(1 for r in preds_rows if r.get("predicted_class") == "1")
    n_active_con = sum(1 for r in preds_rows if r.get("consensus_class") == "1")

    def _safe_bool(v):
        return str(v).strip().lower() in ("true", "1", "yes")

    n_ambiguous = sum(1 for r in preds_rows if _safe_bool(r.get("conformal_ambiguous", "false")))
    n_ood = sum(1 for r in preds_rows if not _safe_bool(r.get("in_domain", "true")))
    n_sev_ood = sum(1 for r in preds_rows if _safe_bool(r.get("severe_ood", "false")))

    conformal_cov = summary.get("conformal_coverage", 0)
    avg_set_size = summary.get("avg_prediction_set_size", 0)
    ood_rate_train = summary.get("out_of_domain_rate", 0)

    return f"""
    <div class="card">
      <h2>1 &mdash; Prediction Overview</h2>
      <p>Predictions from <code>standalone_predictions.csv</code> on novel withdrawn/terminated drug compounds
         run against all 24 safety pharmacology panel targets.</p>
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-value">{n_compounds}</span>
          <span class="metric-label">Unique Compounds</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_targets}</span>
          <span class="metric-label">Targets Predicted</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_total}</span>
          <span class="metric-label">Total Compound-Target Rows</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_active_best} ({_pct(n_active_best, n_total)})</span>
          <span class="metric-label">Active &lt;10µM (Best Model)</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_active_con} ({_pct(n_active_con, n_total)})</span>
          <span class="metric-label">Active &lt;10µM (Consensus)</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_ambiguous} ({_pct(n_ambiguous, n_total)})</span>
          <span class="metric-label">Ambiguous Conformal</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_ood} ({_pct(n_ood, n_total)})</span>
          <span class="metric-label">Out-of-Domain</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{n_sev_ood} ({_pct(n_sev_ood, n_total)})</span>
          <span class="metric-label">Severe OOD (kNN&gt;100)</span>
        </div>
      </div>
      <p><strong>Conformal prediction coverage (training):</strong> {conformal_cov:.1%} (target: 95%)</p>
      <p><strong>Average prediction set size:</strong> {avg_set_size:.2f}</p>
      <p><strong>Out-of-domain rate (training set AD):</strong> {ood_rate_train:.1%}</p>
      {'<p class="note-warning"><strong>NOTE:</strong> &gt;30% ambiguous predictions — rely on <code>consensus_activity</code> and <code>ensemble_confidence</code> over raw probabilities.</p>' if n_ambiguous > 0.3 * n_total else ''}
    </div>
    """


def build_per_target_prediction_section(preds_rows: list) -> str:
    """Section: per-target prediction distribution chart."""
    if not preds_rows:
        return ""

    from collections import defaultdict
    target_counts = defaultdict(lambda: {"binding": 0, "non_binding": 0, "total": 0})
    for r in preds_rows:
        t = r.get("target", "unknown")
        label = r.get("predicted_label", "non_binding")
        target_counts[t][label] = target_counts[t].get(label, 0) + 1
        target_counts[t]["total"] += 1

    targets = sorted(target_counts.keys())
    binding_counts = [target_counts[t].get("binding", 0) for t in targets]
    nonbinding_counts = [target_counts[t].get("non_binding", 0) for t in targets]
    binding_pct = [
        round(100 * target_counts[t].get("binding", 0) / target_counts[t]["total"], 1)
        if target_counts[t]["total"] > 0 else 0
        for t in targets
    ]

    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(
        name="Binding (<10 µM)",
        x=targets,
        y=binding_counts,
        marker_color=CLASS_COLORS["binding"],
        text=[f"{p}%" for p in binding_pct],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Binding: %{y}<extra></extra>",
    ))
    fig_stacked.add_trace(go.Bar(
        name="Non-Binding",
        x=targets,
        y=nonbinding_counts,
        marker_color=CLASS_COLORS["non_binding"],
        hovertemplate="<b>%{x}</b><br>Non-Binding: %{y}<extra></extra>",
    ))
    fig_stacked.update_layout(
        barmode="stack",
        title="Per-Target Predicted Class Distribution (Best Model)",
        xaxis_title="Target",
        yaxis_title="Number of Compounds",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=100),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Consensus binding rate bar
    con_binding = [
        sum(1 for r in preds_rows if r["target"] == t and r.get("consensus_class") == "1")
        for t in targets
    ]
    con_total = [target_counts[t]["total"] for t in targets]
    con_pct = [round(100 * b / n, 1) if n > 0 else 0 for b, n in zip(con_binding, con_total)]

    fig_rate = go.Figure(go.Bar(
        x=targets,
        y=con_pct,
        marker_color=[
            "#c0392b" if p > 70 else "#e67e22" if p > 40 else "#27ae60"
            for p in con_pct
        ],
        text=[f"{p}%" for p in con_pct],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Consensus binding rate: %{y:.1f}%<extra></extra>",
    ))
    fig_rate.update_layout(
        title="Consensus Binding Rate per Target (% of compounds predicted as active)",
        xaxis_title="Target",
        yaxis_title="Binding Rate (%)",
        yaxis_range=[0, 110],
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=100),
        xaxis_tickangle=-45,
    )

    # Table
    table_rows_html = ""
    for t in targets:
        tc = target_counts[t]
        total = tc["total"]
        bind = tc.get("binding", 0)
        nonbind = tc.get("non_binding", 0)
        rate = round(100 * bind / total, 1) if total > 0 else 0
        risk_cls = "risk-high" if rate > 70 else "risk-mid" if rate > 40 else "risk-low"
        table_rows_html += f"""
        <tr>
          <td>{t}</td>
          <td class="text-right">{total}</td>
          <td class="text-right">{bind}</td>
          <td class="text-right">{nonbind}</td>
          <td class="text-right {risk_cls}">{rate}%</td>
        </tr>"""

    return f"""
    <div class="card">
      <h2>2 &mdash; Per-Target Prediction Distribution</h2>
      {_plotly_html(fig_stacked, 480)}
      {_plotly_html(fig_rate, 420)}
      <h3>Per-Target Summary Table</h3>
      <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Target</th><th>Total</th><th>Predicted Binding</th>
            <th>Predicted Non-Binding</th><th>Binding Rate</th>
          </tr>
        </thead>
        <tbody>{table_rows_html}</tbody>
      </table>
      </div>
      <p><em>Binding rate color: <span style="color:#c0392b">&#9632;</span> &gt;70% (high risk)
         <span style="color:#e67e22">&#9632;</span> 40-70% (moderate)
         <span style="color:#27ae60">&#9632;</span> &lt;40% (lower risk)</em></p>
    </div>
    """


def build_compound_table(preds_rows: list) -> str:
    """Section: interactive compound predictions table (top-risk compounds)."""
    if not preds_rows:
        return ""

    def _safe_bool(v):
        return str(v).strip().lower() in ("true", "1", "yes")

    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    # Aggregate per compound: count binding predictions, flag OOD, get avg confidence
    from collections import defaultdict
    compound_summary = defaultdict(lambda: {
        "smiles": "", "n_binding_best": 0, "n_binding_con": 0,
        "n_targets": 0, "n_ood": 0, "n_ambiguous": 0,
        "avg_confidence": 0.0, "conf_sum": 0.0,
        "binding_targets": [], "ambiguous_targets": [], "ood_targets": [],
        "chembl_id": "",
    })

    for r in preds_rows:
        cid = r.get("compound_id", "unknown")
        cs = compound_summary[cid]
        cs["smiles"] = cs["smiles"] or r.get("smiles", "")
        cs["chembl_id"] = cs["chembl_id"] or r.get("ChEMBL ID", "")
        cs["n_targets"] += 1
        conf = _safe_float(r.get("max_confidence", 0))
        cs["conf_sum"] += conf

        if r.get("predicted_class") == "1":
            cs["n_binding_best"] += 1
            cs["binding_targets"].append(r.get("target", ""))
        if r.get("consensus_class") == "1":
            cs["n_binding_con"] += 1
        if not _safe_bool(r.get("in_domain", "true")):
            cs["n_ood"] += 1
            cs["ood_targets"].append(r.get("target", ""))
        if _safe_bool(r.get("conformal_ambiguous", "false")):
            cs["n_ambiguous"] += 1
            cs["ambiguous_targets"].append(r.get("target", ""))

    for cid, cs in compound_summary.items():
        cs["avg_confidence"] = cs["conf_sum"] / cs["n_targets"] if cs["n_targets"] > 0 else 0

    # Sort by number of binding predictions (highest risk first)
    sorted_compounds = sorted(
        compound_summary.items(),
        key=lambda x: (x[1]["n_binding_best"], x[1]["avg_confidence"]),
        reverse=True,
    )

    rows_html = ""
    for cid, cs in sorted_compounds:
        smiles_short = cs["smiles"][:45] + "..." if len(cs["smiles"]) > 45 else cs["smiles"]
        binding_tgts = ", ".join(cs["binding_targets"][:5])
        if len(cs["binding_targets"]) > 5:
            binding_tgts += f" (+{len(cs['binding_targets'])-5} more)"
        ood_flag = f"<span class='flag-ood'>{cs['n_ood']} OOD</span>" if cs["n_ood"] > 0 else ""
        amb_flag = f"<span class='flag-amb'>{cs['n_ambiguous']} amb.</span>" if cs["n_ambiguous"] > 0 else ""
        risk_cls = "risk-high" if cs["n_binding_best"] > 12 else "risk-mid" if cs["n_binding_best"] > 6 else ""

        rows_html += f"""
        <tr class="{risk_cls}">
          <td><strong>{cid}</strong></td>
          <td>{cs['chembl_id']}</td>
          <td class="smiles" title="{cs['smiles']}">{smiles_short}</td>
          <td class="text-right">{cs['n_binding_best']}/{cs['n_targets']}</td>
          <td class="text-right">{cs['n_binding_con']}/{cs['n_targets']}</td>
          <td>{binding_tgts}</td>
          <td class="text-right">{cs['avg_confidence']:.3f}</td>
          <td>{ood_flag} {amb_flag}</td>
        </tr>"""

    return f"""
    <div class="card">
      <h2>3 &mdash; Compound Risk Profile (sorted by binding predictions)</h2>
      <p>Each row is one compound; columns show binding hits across all predicted targets.
         <span class="risk-high-label">Red rows</span>: predicted binding to &gt;12 targets.
         <span class="risk-mid-label">Orange rows</span>: binding to 7–12 targets.</p>
      <div class="legend">
        <span class="legend-item"><span class="flag-ood">N OOD</span> = N targets out-of-domain</span>
        <span class="legend-item"><span class="flag-amb">N amb.</span> = N ambiguous conformal predictions</span>
      </div>
      <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Compound ID</th><th>ChEMBL ID</th><th>SMILES</th>
            <th>Binding/Targets (Best)</th><th>Binding/Targets (Consensus)</th>
            <th>Binding Targets (Best)</th>
            <th>Avg Confidence</th><th>Flags</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
      </div>
    </div>
    """


def build_confidence_section(preds_rows: list, summary: dict) -> str:
    """Section: confidence distribution and OOD analysis."""
    if not preds_rows:
        return ""

    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _safe_bool(v):
        return str(v).strip().lower() in ("true", "1", "yes")

    confidences = [_safe_float(r.get("max_confidence", 0)) for r in preds_rows]
    knn_dists = [_safe_float(r.get("knn_distance", 0)) for r in preds_rows]
    labels = [r.get("predicted_label", "non_binding") for r in preds_rows]
    in_domain = [_safe_bool(r.get("in_domain", "true")) for r in preds_rows]

    # Confidence histogram by predicted class
    fig_hist = go.Figure()
    for cls in ["binding", "non_binding"]:
        cls_conf = [c for c, l in zip(confidences, labels) if l == cls]
        if cls_conf:
            fig_hist.add_trace(go.Histogram(
                x=cls_conf,
                name=LABEL_NICE[cls],
                marker_color=CLASS_COLORS[cls],
                opacity=0.7,
                nbinsx=20,
                hovertemplate=f"Confidence: %{{x:.2f}}<br>Count: %{{y}}<extra>{LABEL_NICE[cls]}</extra>",
            ))
    fig_hist.update_layout(
        barmode="overlay",
        title="Confidence Distribution by Predicted Class",
        xaxis_title="Max Confidence",
        yaxis_title="Count",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    # kNN distance scatter — in-domain vs OOD
    fig_knn = go.Figure()
    for dom, color, name in [(True, "#3498db", "In-Domain"), (False, "#e74c3c", "Out-of-Domain")]:
        x_pts = [d for d, ind in zip(knn_dists, in_domain) if ind == dom]
        y_pts = [c for c, ind in zip(confidences, in_domain) if ind == dom]
        if x_pts:
            fig_knn.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name=name,
                marker=dict(color=color, size=5, opacity=0.6),
                hovertemplate="kNN dist: %{x:.2f}<br>Confidence: %{y:.3f}<extra></extra>",
            ))
    ad_thresh = summary.get("out_of_domain_rate", 0)
    fig_knn.update_layout(
        title="Confidence vs. kNN Distance (applicability domain)",
        xaxis_title="kNN Distance (lower = closer to training data)",
        yaxis_title="Max Confidence",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    conformal_cov = summary.get("conformal_coverage", 0)
    avg_set = summary.get("avg_prediction_set_size", 0)

    return f"""
    <div class="card">
      <h2>4 &mdash; Confidence &amp; Applicability Domain</h2>
      {_plotly_html(fig_hist, 420)}
      {_plotly_html(fig_knn, 450)}
      <h3>Conformal Prediction Coverage</h3>
      <ul>
        <li><strong>Coverage:</strong> {conformal_cov:.1%} (target: 95%)</li>
        <li><strong>Average prediction set size:</strong> {avg_set:.2f}</li>
        <li><strong>Out-of-domain rate (training):</strong> {summary.get('out_of_domain_rate', 0):.1%}</li>
        <li><strong>Severe OOD rate (kNN &gt; 100):</strong> {summary.get('severe_ood_rate', 0):.2%}</li>
      </ul>
      <p>Compounds outside the applicability domain (in_domain = False) have higher
         uncertainty. Prefer <code>conformal_set</code> and <code>consensus_activity</code>
         over raw probabilities for decision-making on OOD compounds.</p>
    </div>
    """


def build_sas_section(artifacts: dict) -> str:
    """Section: SAS core analysis from model artifact."""
    sas_cores = artifacts.get("sas_cores", {})
    if not sas_cores:
        return """
    <div class="card">
      <h2>5 &mdash; SAS Core Analysis</h2>
      <p>No SAS core data available. Re-run the notebook to generate
         MCS cores (model pkl must contain the <code>sas_cores</code> key).</p>
    </div>
    """

    try:
        from rdkit import Chem
        rdkit_available = True
    except ImportError:
        rdkit_available = False

    def _core_atom_count(smarts):
        if not smarts or not rdkit_available:
            return None
        try:
            mol = Chem.MolFromSmarts(smarts)
            return mol.GetNumAtoms() if mol else None
        except Exception:
            return None

    table_rows = []
    core_sizes = {}

    for target in sorted(sas_cores.keys()):
        smarts = sas_cores[target]
        if smarts:
            n_atoms = _core_atom_count(smarts)
            core_sizes[target] = n_atoms if n_atoms is not None else 0
            smarts_short = smarts if len(smarts) <= 60 else smarts[:57] + "..."
            table_rows.append(f"""
            <tr>
              <td>{target}</td>
              <td class="smiles" title="{smarts}">{smarts_short}</td>
              <td class="text-right">{n_atoms if n_atoms is not None else "&mdash;"}</td>
            </tr>""")
        else:
            core_sizes[target] = 0
            table_rows.append(f"""
            <tr>
              <td>{target}</td>
              <td style="color:#999"><em>No core (too few actives or MCS degenerate)</em></td>
              <td class="text-right">&mdash;</td>
            </tr>""")

    table_html = f"""
    <h3>Per-Target Active Core (MCS SMARTS)</h3>
    <div class="table-wrap">
    <table>
      <thead><tr><th>Target</th><th>Core SMARTS</th><th>Core Atoms</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    </div>
    """

    targets_sorted = sorted(core_sizes, key=lambda t: core_sizes[t], reverse=True)
    sizes_sorted = [core_sizes[t] for t in targets_sorted]
    bar_colors = ["#3498db" if s > 0 else "#bdc3c7" for s in sizes_sorted]

    fig_bar = go.Figure(go.Bar(
        x=targets_sorted,
        y=sizes_sorted,
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>Core atoms: %{y}<extra></extra>",
    ))
    fig_bar.update_layout(
        title="Active Core Size per Target (MCS atom count)",
        xaxis_title="Target",
        yaxis_title="Core Atoms",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=80),
        xaxis_tickangle=-45,
    )

    non_zero_sizes = [s for s in core_sizes.values() if s > 0]
    pan_active_html = ""
    if non_zero_sizes:
        import statistics
        median_size = statistics.median(non_zero_sizes)
        large_core_targets = sorted([t for t, s in core_sizes.items() if s >= median_size and s > 0])
        small_core_targets = sorted([t for t, s in core_sizes.items() if 0 < s < median_size])
        no_core_targets = sorted([t for t, s in core_sizes.items() if s == 0])
        pan_active_html = f"""
        <h3>Cross-Target Pan-Active Core Detection</h3>
        <p>Targets with large MCS cores (&ge; median {median_size:.0f} atoms) have conserved
           active pharmacophores — compounds matching these cores have elevated pan-active risk:</p>
        <ul>
          <li><strong>Large-core targets ({len(large_core_targets)}):</strong>
              {', '.join(large_core_targets) if large_core_targets else 'None'}</li>
          <li><strong>Small-core targets ({len(small_core_targets)}):</strong>
              {', '.join(small_core_targets) if small_core_targets else 'None'}</li>
          <li><strong>No core available ({len(no_core_targets)}):</strong>
              {', '.join(no_core_targets) if no_core_targets else 'None'}</li>
        </ul>
        """

    return f"""
    <div class="card">
      <h2>5 &mdash; SAS Core Analysis</h2>
      <p>Maximum Common Substructure (MCS) cores derived from training active compounds
         per target. The <code>sas_score</code> in predictions is the Tanimoto similarity
         of a compound's Morgan fingerprint to the core fingerprint.</p>
      {table_html}
      {_plotly_html(fig_bar, 450)}
      {pan_active_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Assemble full HTML
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #f0f2f5;
  color: #333;
  line-height: 1.6;
}
.header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #fff;
  padding: 40px 32px 32px;
  text-align: center;
}
.header h1 { font-size: 1.8rem; margin-bottom: 8px; }
.header .sub { font-size: 0.95rem; color: #b0c4de; margin-top: 4px; }
.card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  padding: 28px 32px;
  margin: 24px auto;
  max-width: 1200px;
}
.card h2 {
  font-size: 1.35rem;
  margin-bottom: 16px;
  color: #1a1a2e;
  border-bottom: 2px solid #e8e8e8;
  padding-bottom: 8px;
}
.card h3 { font-size: 1.1rem; margin: 18px 0 8px; color: #16213e; }
.card h4 { font-size: 1.0rem; margin: 12px 0 6px; color: #16213e; }
.card p, .card li { margin-bottom: 8px; font-size: 0.97rem; }
.card ul { margin-left: 24px; }
.metrics-grid {
  display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0 18px;
}
.metric-box {
  background: #f7f9fc; border: 1px solid #e0e5ec; border-radius: 8px;
  padding: 16px 22px; text-align: center; min-width: 140px; flex: 1;
}
.metric-value { display: block; font-size: 1.4rem; font-weight: 700; color: #0f3460; }
.metric-label { display: block; font-size: 0.82rem; color: #666; margin-top: 4px; }
.table-wrap { overflow-x: auto; margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th {
  background: #16213e; color: #fff; padding: 10px 12px;
  text-align: left; white-space: nowrap;
}
td { padding: 7px 12px; border-bottom: 1px solid #eee; }
.text-right { text-align: right; }
.smiles { font-family: monospace; font-size: 0.82rem; max-width: 260px;
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
tr.risk-high td { background: #ffeaea; }
tr.risk-mid td { background: #fff3e0; }
.risk-high { color: #c0392b; font-weight: bold; }
.risk-mid  { color: #e67e22; font-weight: bold; }
.risk-low  { color: #27ae60; font-weight: bold; }
.risk-high-label { color: #c0392b; font-weight: bold; }
.risk-mid-label  { color: #e67e22; font-weight: bold; }
.flag-ood { background: #e74c3c; color: #fff; border-radius: 4px; padding: 2px 6px; font-size: 0.8rem; margin-right: 4px; }
.flag-amb { background: #f39c12; color: #fff; border-radius: 4px; padding: 2px 6px; font-size: 0.8rem; }
.legend { margin-bottom: 12px; }
.legend-item { display: inline-block; margin-right: 20px; font-size: 0.88rem; }
.note-warning { background: #fff3cd; border-left: 4px solid #f39c12; padding: 10px 14px; border-radius: 4px; }
.md-content { margin-top: 12px; }
.md-content h2 { font-size: 1.15rem; margin: 20px 0 8px; color: #1a1a2e; }
.md-content h3 { font-size: 1.05rem; margin: 16px 0 6px; color: #16213e; }
.md-content h4 { font-size: 0.95rem; margin: 12px 0 4px; }
.md-content table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 10px 0; }
.md-content th { background: #16213e; color: #fff; padding: 8px 12px; }
.md-content td { padding: 6px 12px; border-bottom: 1px solid #eee; }
.md-content pre { background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }
.md-content code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: monospace; }
footer {
  text-align: center; padding: 24px; font-size: 0.82rem; color: #888;
}
@media (max-width: 768px) {
  .card { padding: 16px; margin: 12px 8px; }
  .metrics-grid { flex-direction: column; }
  .header h1 { font-size: 1.3rem; }
}
"""


def assemble_html(sections):
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OFFTOXv3 Validation Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
{body}
<footer>
  OFFTOXv3 Validation Report &mdash; Generated by generate_report.py<br>
  Source files: {STANDALONE_CSV.name} &bull; {ANALYSIS_MD.name} &bull; {SUMMARY_JSON.name}
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OFFTOXv3 Report Generator")
    print(f"  Reading from : {BASE_DIR}")
    print(f"  Output       : {OUTPUT_HTML}")
    print("=" * 60)
    print("Loading data ...")
    preds_rows, summary, md_text, artifacts = load_data()

    sas_cores = artifacts.get("sas_cores", {})
    valid_cores = sum(v is not None for v in sas_cores.values()) if sas_cores else 0
    print(f"  SAS cores loaded: {valid_cores}/{len(sas_cores)} targets")

    print("Building report sections ...")
    sections = [
        build_header(summary),
        build_analysis_md_section(md_text),
        build_prediction_summary_section(preds_rows, summary),
        build_per_target_prediction_section(preds_rows),
        build_compound_table(preds_rows),
        build_confidence_section(preds_rows, summary),
        build_sas_section(artifacts),
    ]

    html = assemble_html(sections)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"Report written to {OUTPUT_HTML}  ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
