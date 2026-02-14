#!/usr/bin/env python3
"""
generate_report.py  --  OFFTOXv3 Validation Report Generator
=============================================================
Reads predictions.csv, workflow_summary.json, and validation_compounds.csv
from the OFFTOXv3 pipeline and produces a fully self-contained interactive
HTML report at outputs/validation_report.html.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_CSV = BASE_DIR / "outputs" / "predictions.csv"
SUMMARY_JSON = BASE_DIR / "outputs" / "workflow_summary.json"
VALIDATION_CSV = BASE_DIR / "data" / "validation_compounds.csv"
OUTPUT_HTML = BASE_DIR / "outputs" / "validation_report.html"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "inactive": "#2ecc71",
    "less_potent": "#f39c12",
    "potent": "#e74c3c",
}

LABEL_NICE = {
    "inactive": "Inactive",
    "less_potent": "Less Potent",
    "potent": "Potent",
}


def _pct(n, d):
    """Return percentage string."""
    return f"{100 * n / d:.1f}%" if d else "N/A"


def _plotly_html(fig, height=None):
    """Render a plotly figure to an embeddable HTML <div>."""
    if height:
        fig.update_layout(height=height)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    preds = pd.read_csv(PREDICTIONS_CSV)
    with open(SUMMARY_JSON) as fh:
        summary = json.load(fh)
    val = pd.read_csv(VALIDATION_CSV)
    return preds, summary, val


# ---------------------------------------------------------------------------
# Section builders  (each returns an HTML string)
# ---------------------------------------------------------------------------

def build_header(summary):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cd = summary.get("class_distribution", {})
    tm = summary.get("test_metrics", {})
    targets_list = ", ".join(summary.get("targets", []))

    return f"""
    <div class="header">
      <h1>OFFTOXv3 Validation Report &mdash; Safety Pharmacology &amp; NHR Predictions</h1>
      <p class="sub">Generated {ts} &nbsp;|&nbsp; Model: <strong>{summary.get('best_model','N/A')}</strong></p>
    </div>

    <div class="card">
      <h2>Model &amp; Training Summary</h2>
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-value">{summary.get('n_compounds','?')}</span>
          <span class="metric-label">Training compounds</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{len(summary.get('targets',[]))}</span>
          <span class="metric-label">Targets</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{summary.get('train_size','?')} / {summary.get('val_size','?')} / {summary.get('test_size','?')}</span>
          <span class="metric-label">Train / Val / Test split</span>
        </div>
      </div>
      <p><strong>Targets:</strong> {targets_list}</p>
      <p><strong>Class distribution:</strong>
         Inactive&nbsp;{cd.get('inactive','?')} &nbsp;|&nbsp;
         Less-potent&nbsp;{cd.get('less_potent','?')} &nbsp;|&nbsp;
         Potent&nbsp;{cd.get('potent','?')}</p>

      <h3>Test-Set Performance</h3>
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-value">{tm.get('roc_auc_macro',0):.4f}</span>
          <span class="metric-label">ROC-AUC (macro)</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{tm.get('pr_auc_macro',0):.4f}</span>
          <span class="metric-label">PR-AUC (macro)</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{tm.get('mcc',0):.4f}</span>
          <span class="metric-label">MCC</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{tm.get('ece',0):.4f}</span>
          <span class="metric-label">ECE</span>
        </div>
      </div>
    </div>
    """


def build_compound_table(preds):
    """Section 1 -- interactive compound profiles table."""
    rows_html = []
    for _, r in preds.iterrows():
        known = str(r.get("known_class", "")).strip()
        predicted = str(r.get("predicted_label", "")).strip()
        if not known or known == "nan":
            row_class = "row-unknown"
        elif known == predicted:
            row_class = "row-correct"
        else:
            row_class = "row-incorrect"

        smiles_short = str(r["smiles"])
        if len(smiles_short) > 45:
            smiles_short = smiles_short[:42] + "..."

        rows_html.append(f"""
        <tr class="{row_class}">
          <td>{r['compound_id']}</td>
          <td>{r['compound_name']}</td>
          <td>{r['target']}</td>
          <td class="smiles" title="{r['smiles']}">{smiles_short}</td>
          <td>{LABEL_NICE.get(known, known) if known and known != 'nan' else '&mdash;'}</td>
          <td>{LABEL_NICE.get(predicted, predicted)}</td>
          <td>{r['max_confidence']:.3f}</td>
          <td>{'Yes' if r['in_domain'] else 'No'}</td>
          <td>{r['conformal_set']}</td>
        </tr>""")

    return f"""
    <div class="card">
      <h2>1 &mdash; Compound Profiles</h2>
      <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:#c6efce"></span> Correct prediction</span>
        <span class="legend-item"><span class="swatch" style="background:#ffc7ce"></span> Incorrect prediction</span>
        <span class="legend-item"><span class="swatch" style="background:#fff3cd"></span> No known class</span>
      </div>
      <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Name</th><th>Target</th><th>SMILES</th>
            <th>Known Class</th><th>Predicted</th><th>Confidence</th>
            <th>In Domain</th><th>Conformal Set</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
      </div>
    </div>
    """


def build_accuracy_section(preds):
    """Section 2 -- prediction accuracy with confusion matrix."""
    # Only rows with a known_class
    df = preds.copy()
    df["known_class"] = df["known_class"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip()
    mask = df["known_class"].isin(["inactive", "less_potent", "potent"])
    df = df[mask].copy()

    n_total = len(df)
    n_correct = (df["known_class"] == df["predicted_label"]).sum()
    overall_acc = n_correct / n_total if n_total else 0

    # Per-class accuracy
    per_class_html = ""
    classes = ["inactive", "less_potent", "potent"]
    for c in classes:
        sub = df[df["known_class"] == c]
        n_c = len(sub)
        n_ok = (sub["predicted_label"] == c).sum()
        acc = n_ok / n_c if n_c else 0
        per_class_html += f"<li><strong>{LABEL_NICE[c]}</strong>: {n_ok}/{n_c} correct ({acc:.0%})</li>"

    # Confusion matrix
    labels = classes
    label_names = [LABEL_NICE[c] for c in classes]
    cm = pd.crosstab(df["known_class"], df["predicted_label"],
                     rownames=["Actual"], colnames=["Predicted"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm.values,
        x=label_names,
        y=label_names,
        colorscale="Blues",
        text=cm.values,
        texttemplate="%{text}",
        textfont={"size": 18},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig_cm.update_layout(
        title="Confusion Matrix (Validation Compounds)",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis_autorange="reversed",
        template="plotly_white",
        margin=dict(l=80, r=40, t=60, b=60),
    )

    # Misclassified compounds detail
    wrong = df[df["known_class"] != df["predicted_label"]]
    wrong_bullets = ""
    for _, r in wrong.iterrows():
        wrong_bullets += (
            f"<li><strong>{r['compound_name']}</strong> ({r['compound_id']}, {r['target']}): "
            f"known={LABEL_NICE.get(r['known_class'], r['known_class'])}, "
            f"predicted={LABEL_NICE.get(r['predicted_label'], r['predicted_label'])}, "
            f"confidence={r['max_confidence']:.3f}</li>"
        )

    return f"""
    <div class="card">
      <h2>2 &mdash; Prediction Accuracy</h2>
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-value">{n_correct}/{n_total}</span>
          <span class="metric-label">Correct</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">{overall_acc:.1%}</span>
          <span class="metric-label">Overall Accuracy</span>
        </div>
      </div>
      <h3>Per-Class Accuracy</h3>
      <ul>{per_class_html}</ul>
      {_plotly_html(fig_cm, 420)}
      {'<h3>Misclassified Compounds</h3><ul>' + wrong_bullets + '</ul>' if wrong_bullets else '<p>All compounds correctly classified.</p>'}
    </div>
    """


def build_probability_section(preds):
    """Section 3 -- probability distributions."""
    compound_names = preds["compound_name"].tolist()

    fig = go.Figure()
    for cls, col in [("inactive", "prob_inactive"),
                     ("less_potent", "prob_less_potent"),
                     ("potent", "prob_potent")]:
        fig.add_trace(go.Bar(
            name=LABEL_NICE[cls],
            x=compound_names,
            y=preds[col],
            marker_color=CLASS_COLORS[cls],
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{LABEL_NICE[cls]}: " + "%{y:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        title="Predicted Class Probabilities per Compound",
        xaxis_title="Compound",
        yaxis_title="Probability",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        legend_title="Class",
        margin=dict(l=60, r=30, t=60, b=120),
        xaxis_tickangle=-45,
    )
    return f"""
    <div class="card">
      <h2>3 &mdash; Probability Distributions</h2>
      <p>Grouped bar chart showing the three predicted class probabilities for each
         validation compound. Hover for exact values.</p>
      {_plotly_html(fig, 520)}
    </div>
    """


def build_confidence_section(preds, summary):
    """Section 4 -- confidence & uncertainty QC."""
    df = preds.copy()
    df["known_class"] = df["known_class"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip()
    df["correct"] = df.apply(
        lambda r: "Correct" if r["known_class"] == r["predicted_label"]
                  else ("Incorrect" if r["known_class"] in ("inactive", "less_potent", "potent")
                        else "Unknown"),
        axis=1,
    )
    color_map = {"Correct": "#2ecc71", "Incorrect": "#e74c3c", "Unknown": "#95a5a6"}

    # Scatter: confidence vs knn_distance
    fig_scatter = go.Figure()
    for status, color in color_map.items():
        sub = df[df["correct"] == status]
        if sub.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sub["knn_distance"],
            y=sub["max_confidence"],
            mode="markers+text",
            text=sub["compound_name"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=color, size=12, line=dict(width=1, color="white")),
            name=status,
            hovertemplate=(
                "<b>%{text}</b><br>kNN dist: %{x:.2f}<br>"
                "Confidence: %{y:.3f}<extra></extra>"
            ),
        ))

    # AD threshold -- use OOD rate to derive a rough threshold from the data
    ood_rate = summary.get("out_of_domain_rate", None)
    if ood_rate is not None:
        q = 1 - ood_rate  # e.g. 0.815
        ad_thresh = float(df["knn_distance"].quantile(q)) if len(df) > 2 else 20.0
    else:
        ad_thresh = 20.0

    fig_scatter.add_shape(
        type="line", x0=ad_thresh, x1=ad_thresh,
        y0=0, y1=1, line=dict(color="orange", dash="dash", width=2),
    )
    fig_scatter.add_annotation(
        x=ad_thresh, y=1.02, text=f"AD threshold ~ {ad_thresh:.1f}",
        showarrow=False, font=dict(color="orange", size=11),
    )
    fig_scatter.update_layout(
        title="Confidence vs. kNN Distance",
        xaxis_title="kNN Distance (lower = more similar to training data)",
        yaxis_title="Max Confidence",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    # Box plot of confidence by predicted label
    fig_box = go.Figure()
    for cls in ["inactive", "less_potent", "potent"]:
        sub = df[df["predicted_label"] == cls]
        if sub.empty:
            continue
        fig_box.add_trace(go.Box(
            y=sub["max_confidence"],
            name=LABEL_NICE[cls],
            marker_color=CLASS_COLORS[cls],
            boxpoints="all",
            text=sub["compound_name"],
            hovertemplate="<b>%{text}</b><br>Confidence: %{y:.3f}<extra></extra>",
        ))
    fig_box.update_layout(
        title="Confidence Distribution by Predicted Class",
        yaxis_title="Max Confidence",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    conformal_cov = summary.get("conformal_coverage", 0)
    avg_set = summary.get("avg_prediction_set_size", 0)

    return f"""
    <div class="card">
      <h2>4 &mdash; Confidence &amp; Uncertainty QC</h2>
      <p>The scatter plot below maps each compound by its kNN distance to training data
         (x-axis) and prediction confidence (y-axis). Compounds beyond the AD threshold
         are considered out-of-domain.</p>
      {_plotly_html(fig_scatter, 500)}
      {_plotly_html(fig_box, 400)}
      <h3>Conformal Prediction Coverage</h3>
      <ul>
        <li><strong>Coverage:</strong> {conformal_cov:.1%} (target: 95%)</li>
        <li><strong>Average prediction set size:</strong> {avg_set:.2f}</li>
        <li><strong>Out-of-domain rate (training):</strong> {summary.get('out_of_domain_rate',0):.1%}</li>
      </ul>
      <p>A conformal coverage near 95% indicates the prediction sets are well-calibrated.
         Smaller prediction sets mean more decisive predictions.</p>
    </div>
    """


def build_target_section(preds):
    """Section 5 -- per-target analysis."""
    df = preds.copy()
    df["known_class"] = df["known_class"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip()

    targets = sorted(df["target"].unique())
    classes = ["inactive", "less_potent", "potent"]

    fig = go.Figure()
    for cls in classes:
        counts = []
        for t in targets:
            counts.append(int(((df["target"] == t) & (df["predicted_label"] == cls)).sum()))
        fig.add_trace(go.Bar(
            name=f"Pred {LABEL_NICE[cls]}",
            x=targets,
            y=counts,
            marker_color=CLASS_COLORS[cls],
        ))
    fig.update_layout(
        barmode="group",
        title="Predicted Class Distribution by Target",
        xaxis_title="Target",
        yaxis_title="Number of Compounds",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    # Known vs predicted comparison table for targets with known_class
    comp_rows = ""
    for t in targets:
        sub = df[df["target"] == t]
        for _, r in sub.iterrows():
            known_c = r["known_class"]
            pred_c = r["predicted_label"]
            if known_c not in ("inactive", "less_potent", "potent"):
                continue
            match_icon = "correct-icon" if known_c == pred_c else "wrong-icon"
            comp_rows += f"""
            <tr>
              <td>{t}</td>
              <td>{r['compound_name']}</td>
              <td>{LABEL_NICE.get(known_c, known_c)}</td>
              <td>{LABEL_NICE.get(pred_c, pred_c)}</td>
              <td class="{match_icon}">{'&#10003;' if known_c == pred_c else '&#10007;'}</td>
            </tr>"""

    return f"""
    <div class="card">
      <h2>5 &mdash; Per-Target Analysis</h2>
      {_plotly_html(fig, 440)}
      <h3>Known vs. Predicted (where ground truth available)</h3>
      <div class="table-wrap">
      <table>
        <thead><tr><th>Target</th><th>Compound</th><th>Known</th><th>Predicted</th><th>Match</th></tr></thead>
        <tbody>{comp_rows}</tbody>
      </table>
      </div>
    </div>
    """


def build_interpretive_summary(preds, summary):
    """Section 6 -- auto-generated interpretation."""
    df = preds.copy()
    df["known_class"] = df["known_class"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip()

    has_known = df["known_class"].isin(["inactive", "less_potent", "potent"])
    df_known = df[has_known]
    n_total = len(df_known)
    n_correct = (df_known["known_class"] == df_known["predicted_label"]).sum()
    acc = n_correct / n_total if n_total else 0

    correct_names = df_known.loc[
        df_known["known_class"] == df_known["predicted_label"], "compound_name"
    ].tolist()
    wrong_names = df_known.loc[
        df_known["known_class"] != df_known["predicted_label"], "compound_name"
    ].tolist()

    in_domain_count = int(df["in_domain"].sum())
    n_all = len(df)
    ad_pct = in_domain_count / n_all if n_all else 0

    avg_conf = df["max_confidence"].mean()
    tm = summary.get("test_metrics", {})

    targets = summary.get("targets", [])
    nhr_targets = {"AR", "ERa", "GR", "PPARg", "PR", "PXR", "RXRa"}
    ion_targets = {"hERG", "Nav1.5", "Cav1.2"}
    cyp_targets = {"CYP1A2", "CYP2D6", "CYP3A4"}

    covered_nhr = sorted(nhr_targets & set(targets))
    covered_ion = sorted(ion_targets & set(targets))
    covered_cyp = sorted(cyp_targets & set(targets))

    wrong_detail = ""
    for _, r in df_known[df_known["known_class"] != df_known["predicted_label"]].iterrows():
        wrong_detail += (
            f"<li><strong>{r['compound_name']}</strong>: known {LABEL_NICE.get(r['known_class'], r['known_class'])} "
            f"but predicted {LABEL_NICE.get(r['predicted_label'], r['predicted_label'])} "
            f"(confidence {r['max_confidence']:.3f}, in-domain={'Yes' if r['in_domain'] else 'No'})</li>"
        )

    return f"""
    <div class="card">
      <h2>6 &mdash; Interpretive Summary</h2>

      <h3>Overall Model Performance</h3>
      <p>The {summary.get('best_model','model')} achieved a macro ROC-AUC of
         <strong>{tm.get('roc_auc_macro',0):.4f}</strong> and a macro PR-AUC of
         <strong>{tm.get('pr_auc_macro',0):.4f}</strong> on the held-out test set
         ({summary.get('test_size','?')} compounds). The Matthews Correlation Coefficient
         (MCC) is {tm.get('mcc',0):.4f}, indicating {'moderate' if tm.get('mcc',0) < 0.5 else 'good'}
         multi-class discrimination. The Expected Calibration Error (ECE) of
         {tm.get('ece',0):.4f} suggests {'the predicted probabilities may be poorly calibrated and should be interpreted with caution' if tm.get('ece',0) > 0.15 else 'reasonable probability calibration'}.</p>

      <h3>Validation Compound Results</h3>
      <p>Of the <strong>{n_total}</strong> validation compounds with known class labels,
         <strong>{n_correct}</strong> ({acc:.0%}) were predicted correctly.</p>
      <p><strong>Correctly predicted:</strong> {', '.join(correct_names) if correct_names else 'None'}</p>
      <p><strong>Incorrectly predicted:</strong> {', '.join(wrong_names) if wrong_names else 'None'}</p>
      {('<ul>' + wrong_detail + '</ul>') if wrong_detail else ''}

      <h3>Applicability Domain Coverage</h3>
      <p><strong>{in_domain_count}/{n_all}</strong> ({ad_pct:.0%}) of the validation compounds
         fall within the model's applicability domain. Compounds outside the AD should be
         interpreted with additional caution, as the model has limited experience with
         their chemical space.</p>

      <h3>Confidence Levels</h3>
      <p>The average maximum confidence across all validation predictions is
         <strong>{avg_conf:.3f}</strong>.
         {'Confidence values are moderate, reflecting inherent uncertainty in the three-class problem.' if avg_conf < 0.8 else 'Confidence values are generally high, suggesting the model is decisive for these compounds.'}</p>

      <h3>Target Coverage Assessment</h3>
      <ul>
        <li><strong>Nuclear Hormone Receptors ({len(covered_nhr)}):</strong> {', '.join(covered_nhr)}</li>
        <li><strong>Ion Channels ({len(covered_ion)}):</strong> {', '.join(covered_ion)}</li>
        <li><strong>CYP Enzymes ({len(covered_cyp)}):</strong> {', '.join(covered_cyp)}</li>
      </ul>
      <p>The model covers {len(targets)} safety-pharmacology targets spanning nuclear
         hormone receptors, cardiac ion channels, and drug-metabolizing CYP enzymes,
         providing a broad off-target liability profile.</p>

      <h3>Caveats &amp; Limitations</h3>
      <ul>
        <li>Predictions are based on molecular fingerprint features only; 3D binding
            mode information is not captured.</li>
        <li>The three-class potency binning (inactive / less-potent / potent) is a
            simplification; borderline compounds near class boundaries may be
            misclassified.</li>
        <li>Conformal prediction sets should be consulted alongside point predictions
            to gauge uncertainty.</li>
        <li>Compounds outside the applicability domain (in_domain = False) have higher
            risk of unreliable predictions.</li>
        <li>The ECE of {tm.get('ece',0):.4f} indicates that raw probability values may
            not be perfectly calibrated; use prediction sets and confidence rankings
            rather than absolute probability thresholds for decision-making.</li>
        <li>This validation set is small ({n_all} compounds) and may not capture all
            failure modes. Production use requires larger-scale prospective validation.</li>
      </ul>
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
.header .sub { font-size: 0.95rem; color: #b0c4de; }
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
.card h3 {
  font-size: 1.1rem;
  margin: 18px 0 8px;
  color: #16213e;
}
.card p, .card li {
  margin-bottom: 8px;
  font-size: 0.97rem;
}
.card ul { margin-left: 24px; }
.metrics-grid {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin: 12px 0 18px;
}
.metric-box {
  background: #f7f9fc;
  border: 1px solid #e0e5ec;
  border-radius: 8px;
  padding: 16px 22px;
  text-align: center;
  min-width: 140px;
  flex: 1;
}
.metric-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: #0f3460;
}
.metric-label {
  display: block;
  font-size: 0.82rem;
  color: #666;
  margin-top: 4px;
}
/* Table styles */
.table-wrap { overflow-x: auto; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  margin-top: 8px;
}
th {
  background: #16213e;
  color: #fff;
  padding: 10px 12px;
  text-align: left;
  white-space: nowrap;
}
td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.smiles { font-family: monospace; font-size: 0.82rem; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row-correct { background: #c6efce; }
.row-incorrect { background: #ffc7ce; }
.row-unknown { background: #fff3cd; }
.correct-icon { color: #27ae60; font-weight: bold; text-align: center; }
.wrong-icon { color: #e74c3c; font-weight: bold; text-align: center; }
.legend { margin-bottom: 12px; }
.legend-item { display: inline-block; margin-right: 20px; font-size: 0.88rem; }
.swatch { display: inline-block; width: 16px; height: 16px; border-radius: 3px; vertical-align: middle; margin-right: 4px; border: 1px solid #ccc; }
footer {
  text-align: center;
  padding: 24px;
  font-size: 0.82rem;
  color: #888;
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
  OFFTOXv3 Validation Report &mdash; Generated by generate_report.py
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    preds, summary, val = load_data()
    print(f"  Predictions: {len(preds)} rows")
    print(f"  Summary model: {summary.get('best_model')}")
    print(f"  Validation compounds: {len(val)} rows")

    print("Building report sections ...")
    sections = [
        build_header(summary),
        build_compound_table(preds),
        build_accuracy_section(preds),
        build_probability_section(preds),
        build_confidence_section(preds, summary),
        build_target_section(preds),
        build_interpretive_summary(preds, summary),
    ]

    html = assemble_html(sections)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"Report written to {OUTPUT_HTML}  ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
