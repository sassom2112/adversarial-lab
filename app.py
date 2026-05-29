"""
adversarial-lab — Constraint Inflation Interactive Demo

Shows the constraint inflation gap in real time:
  - Select a pre-loaded attack flow from the UNSW-NB15 test set
  - Choose a perturbation budget ε
  - Watch constrained PGD vs unconstrained PGD attack the classifier
  - See exactly which features go physically impossible

Setup: run scripts/prepare_artifacts.py (or the Colab notebook) first to
generate model/weights.pt, model/scaler.pkl, model/bounds.pkl, and data/fixtures.json.
"""

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request

from model.mlp import MLP, load_model

app = Flask(__name__)

# ── Load artifacts ────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)

def _load_artifacts():
    weights_path = os.path.join(BASE, "model", "weights.pt")
    scaler_path  = os.path.join(BASE, "model", "scaler.pkl")
    bounds_path  = os.path.join(BASE, "model", "bounds.pkl")
    fixture_path = os.path.join(BASE, "data",  "fixtures.json")

    missing = [p for p in [weights_path, scaler_path, bounds_path, fixture_path]
               if not os.path.exists(p)]
    if missing:
        return None, None, None, None, (
            f"Missing artifacts: {missing}. "
            "Run scripts/prepare_artifacts.py (or the Colab notebook) first."
        )

    with open(scaler_path,  "rb") as f: scaler = pickle.load(f)
    with open(bounds_path,  "rb") as f: bounds = pickle.load(f)
    with open(fixture_path, "r")  as f: fixtures = json.load(f)

    input_dim = bounds["lb"].shape[0]
    model = load_model(weights_path, input_dim)
    return model, scaler, bounds, fixtures, None


MODEL, SCALER, BOUNDS, FIXTURES, LOAD_ERROR = _load_artifacts()


# ── Attack helpers ────────────────────────────────────────────────────────────

def _project(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, lb, ub)


def _pgd(model, x0_np, epsilon, n_steps=40, constrained=True):
    device = torch.device("cpu")
    lb = torch.tensor(BOUNDS["lb"], dtype=torch.float32)
    ub = torch.tensor(BOUNDS["ub"], dtype=torch.float32)
    alpha = epsilon / n_steps * 2.5

    x0  = torch.tensor(x0_np, dtype=torch.float32).unsqueeze(0)
    crit = nn.BCEWithLogitsLoss()

    delta = torch.zeros_like(x0).uniform_(-epsilon, epsilon)
    x_adv = x0 + delta
    if constrained:
        x_adv = _project(x_adv, lb, ub)

    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = crit(model(x_adv), torch.ones(1))
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x0, -epsilon, epsilon)
            x_adv = x0 + delta
            if constrained:
                x_adv = _project(x_adv, lb, ub)

    return x_adv.squeeze(0).detach().numpy()


def _predict(model, x_np):
    with torch.no_grad():
        logit = model(torch.tensor(x_np, dtype=torch.float32).unsqueeze(0))
    return "ATTACK" if logit.item() > 0 else "BENIGN"


def _constraint_violations(x_scaled_np, feature_names, scaler_mean, scaler_std):
    """Return list of {feature, original_value, valid_lo, valid_hi} for violated bounds."""
    lb = BOUNDS["lb"]
    ub = BOUNDS["ub"]
    violations = []
    n = len(feature_names)
    for i, name in enumerate(feature_names):
        val = float(x_scaled_np[i])
        lo  = float(lb[i]) if not np.isinf(lb[i]) else None
        hi  = float(ub[i]) if not np.isinf(ub[i]) else None
        # Convert back to original domain
        orig_val = val * scaler_std[i] + scaler_mean[i]
        lo_orig  = lo  * scaler_std[i] + scaler_mean[i] if lo is not None else None
        hi_orig  = hi  * scaler_std[i] + scaler_mean[i] if hi is not None else None
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            violations.append({
                "feature":  name,
                "value":    round(orig_val, 3),
                "valid_lo": round(lo_orig, 3) if lo_orig is not None else None,
                "valid_hi": round(hi_orig, 3) if hi_orig is not None else None,
            })
    return violations


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if LOAD_ERROR:
        return f"<pre>Setup required:\n{LOAD_ERROR}</pre>", 503
    samples = [{"id": i, "label": s["label"], "description": s["description"]}
               for i, s in enumerate(FIXTURES)]
    return render_template("index.html", samples=samples)


@app.route("/api/attack", methods=["POST"])
def attack():
    if LOAD_ERROR:
        return jsonify({"error": LOAD_ERROR}), 503

    data    = request.get_json()
    sample_id = int(data.get("sample_id", 0))
    epsilon   = float(data.get("epsilon", 0.20))
    epsilon   = max(0.01, min(0.50, epsilon))

    if sample_id >= len(FIXTURES):
        return jsonify({"error": "invalid sample_id"}), 400

    sample   = FIXTURES[sample_id]
    x0_np    = np.array(sample["features_scaled"], dtype=np.float32)
    feat_names = sample["feature_names"]
    scaler_mean = np.array(sample["scaler_mean"])
    scaler_std  = np.array(sample["scaler_std"])

    # Original prediction
    orig_pred = _predict(MODEL, x0_np)

    # Constrained PGD
    x_con  = _pgd(MODEL, x0_np, epsilon, constrained=True)
    con_pred = _predict(MODEL, x_con)

    # Unconstrained PGD
    x_uncon = _pgd(MODEL, x0_np, epsilon, constrained=False)
    uncon_pred = _predict(MODEL, x_uncon)

    # Build feature table (show only DISPLAY_FEATURES if defined)
    display = sample.get("display_features", feat_names[:12])
    rows = []
    for name in display:
        if name not in feat_names:
            continue
        i = feat_names.index(name)
        orig_orig  = float(x0_np[i])    * scaler_std[i] + scaler_mean[i]
        con_orig   = float(x_con[i])    * scaler_std[i] + scaler_mean[i]
        uncon_orig = float(x_uncon[i])  * scaler_std[i] + scaler_mean[i]

        lb_i = float(BOUNDS["lb"][i])
        ub_i = float(BOUNDS["ub"][i])
        con_violated   = (not np.isinf(lb_i) and float(x_con[i])   < lb_i) or \
                         (not np.isinf(ub_i) and float(x_con[i])   > ub_i)
        uncon_violated = (not np.isinf(lb_i) and float(x_uncon[i]) < lb_i) or \
                         (not np.isinf(ub_i) and float(x_uncon[i]) > ub_i)

        lo_orig = (lb_i * scaler_std[i] + scaler_mean[i]) if not np.isinf(lb_i) else None
        hi_orig = (ub_i * scaler_std[i] + scaler_mean[i]) if not np.isinf(ub_i) else None

        rows.append({
            "feature":        name,
            "original":       round(orig_orig,  3),
            "constrained":    round(con_orig,   3),
            "unconstrained":  round(uncon_orig, 3),
            "con_violated":   con_violated,
            "uncon_violated": uncon_violated,
            "valid_lo":       round(lo_orig, 3) if lo_orig is not None else None,
            "valid_hi":       round(hi_orig, 3) if hi_orig is not None else None,
        })

    violations = _constraint_violations(x_uncon, feat_names, scaler_mean, scaler_std)

    return jsonify({
        "epsilon":          epsilon,
        "original_pred":    orig_pred,
        "con_pred":         con_pred,
        "uncon_pred":       uncon_pred,
        "con_evaded":       (orig_pred == "ATTACK" and con_pred   == "BENIGN"),
        "uncon_evaded":     (orig_pred == "ATTACK" and uncon_pred == "BENIGN"),
        "feature_rows":     rows,
        "violations":       violations,
        "n_violations":     len(violations),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
