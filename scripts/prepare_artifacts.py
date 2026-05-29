"""
prepare_artifacts.py — generate model weights + fixture samples for the adversarial lab.

Run this in Colab (with the UNSW-NB15 parquet available) or locally:

    python scripts/prepare_artifacts.py \
        --parquet /path/to/traffic_cleaned.parquet \
        --out-dir .

Outputs:
    model/weights.pt   — trained MLP weights
    model/scaler.pkl   — fitted StandardScaler
    model/bounds.pkl   — ConstraintBounds (lb/ub arrays + n_num)
    data/fixtures.json — 8 representative attack samples with metadata

The lab app loads these at startup. No dataset file is needed at runtime.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ── Find netadv library — check several locations ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
_candidates = [
    REPO_ROOT,                                      # adversarial-lab root (if netadv copied here)
    REPO_ROOT.parent / "netadv-ccs",                # local: sibling repo named netadv-ccs
    REPO_ROOT.parent / "catt-ccs",                  # local: sibling repo named catt-ccs
    Path("/content/netadv-ccs"),                    # Colab: cloned as netadv-ccs
    Path("/content/catt-ccs"),                      # Colab: cloned as catt-ccs
]
_found = False
for _p in _candidates:
    if (_p / "netadv").exists():
        sys.path.insert(0, str(_p))
        print(f"Found netadv at: {_p}")
        _found = True
        break
if not _found:
    print("ERROR: Could not find the netadv library. Clone catt-ccs first:")
    print("  !git clone https://github.com/sassom2112/catt-ccs /content/netadv-ccs")
    sys.exit(1)

from netadv.constraints.bounds import ConstraintBounds, validity_report
from netadv.constraints.datasets.unsw_nb15 import NUM_FEATURES, UNSW_NB15_SPEC
from netadv.attacks.fgsm import fgsm
from netadv.attacks.pgd import pgd

CAT_FEATURES = ["proto", "state", "service"]
TARGET       = "label"

# Features shown in the lab UI — most interpretable in original units
DISPLAY_FEATURES = [
    "dur", "sbytes", "dbytes", "spkts", "dpkts",
    "sttl", "dttl", "sintpkt", "dintpkt",
]

# Human-readable descriptions for each attack category
ATTACK_LABELS = {
    "DoS":       "DoS Attack — high-rate connection flood",
    "Exploits":  "Exploit Attempt — protocol-level attack",
    "Fuzzers":   "Fuzzer — random/malformed traffic",
    "Backdoor":  "Backdoor — persistent C2 connection",
    "Shellcode": "Shellcode — binary payload delivery",
    "Generic":   "Generic Attack — signature-matched flow",
    "Recon":     "Reconnaissance — network scanning activity",
    "Worms":     "Worm — self-propagating traffic pattern",
    "Analysis":  "Traffic Analysis — port/service probing",
}


# ── MLP ──────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(X_tr, y_tr, X_val, y_val, device, epochs=25, patience=5):
    from torch.utils.data import DataLoader, TensorDataset
    n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=2048, shuffle=True)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    model = _MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_f1, best_state, no_imp = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            preds = (model(Xv).cpu().numpy() > 0).astype(int)
        vf1 = f1_score(y_val, preds, zero_division=0)
        sched.step(total / len(X_tr))
        print(f"  epoch {ep:3d}  loss={total/len(X_tr):.4f}  val_f1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  early stop (best val_f1={best_f1:.4f})"); break
    model.load_state_dict(best_state); model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",    required=True, help="Path to traffic_cleaned.parquet")
    parser.add_argument("--out-dir",    default=".",   help="Root of adversarial-lab repo")
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n-fixtures", type=int, default=8,
                        help="Number of fixture samples to save")
    args = parser.parse_args()

    out = Path(args.out_dir)
    model_dir = out / "model"; model_dir.mkdir(exist_ok=True)
    data_dir  = out / "data";  data_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data…")
    df = pd.read_parquet(args.parquet)
    if args.max_samples < len(df):
        df = df.sample(args.max_samples, random_state=args.seed)
    print(f"Using {len(df):,} rows")

    X_raw = df[NUM_FEATURES + CAT_FEATURES]
    y     = df[TARGET].to_numpy(dtype=int)
    cats  = df["attack_cat"].to_numpy(dtype=str) if "attack_cat" in df.columns else None

    X_tr_raw, X_te_raw, y_tr, y_te, *cat_split = train_test_split(
        X_raw, y, *([cats] if cats is not None else []),
        test_size=0.2, random_state=args.seed, stratify=y)
    cats_te = cat_split[1] if cat_split else None

    # ── Build pipeline ────────────────────────────────────────────────────────
    print("Fitting pipeline…")
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                          ("scaler",  StandardScaler())]), NUM_FEATURES),
        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), CAT_FEATURES),
    ])
    pipeline = Pipeline([("prep", pre)])
    pipeline.fit(X_tr_raw)

    X_tr = pipeline.transform(X_tr_raw).astype(np.float32)
    X_te = pipeline.transform(X_te_raw).astype(np.float32)

    scaler = pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
    bounds = ConstraintBounds.from_spec(UNSW_NB15_SPEC, pipeline)

    # ── Validity gate ─────────────────────────────────────────────────────────
    report = validity_report(X_te, bounds)
    if report.empty:
        print("✓ validity_report: zero violations")
    else:
        print(f"WARNING: {len(report)} violation(s) — check spec")

    # ── Train MLP ─────────────────────────────────────────────────────────────
    print("Training MLP…")
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=args.seed, stratify=y_tr)
    model = _train_mlp(X_tr2, y_tr2, X_val, y_val, device)
    with torch.no_grad():
        test_preds = (model(torch.tensor(X_te).to(device)).cpu().numpy() > 0).astype(int)
    print(f"Test F1: {f1_score(y_te, test_preds, zero_division=0):.4f}")

    # ── Save model artifacts ──────────────────────────────────────────────────
    torch.save(model.state_dict(), model_dir / "weights.pt")
    with open(model_dir / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(model_dir / "bounds.pkl", "wb") as f:
        pickle.dump({"lb": bounds.lb, "ub": bounds.ub,
                     "n_num": bounds.n_num, "n_total": bounds.n_total}, f)
    print(f"Saved: {model_dir}/weights.pt, scaler.pkl, bounds.pkl")

    # ── Find good fixture samples ─────────────────────────────────────────────
    # Want: true attacks that CONSTRAINED PGD fails to evade but UNCONSTRAINED does.
    # These most dramatically illustrate the constraint inflation gap.
    print("\nSelecting fixture samples…")

    no_bounds = ConstraintBounds(
        lb=np.full(bounds.n_total, -np.inf, dtype=np.float32),
        ub=np.full(bounds.n_total,  np.inf, dtype=np.float32),
        n_num=bounds.n_num, n_total=bounds.n_total)

    attack_idx = np.where(y_te == 1)[0]
    eps_demo   = 0.20
    alpha      = eps_demo / 40 * 2.5

    # Score each attack sample
    candidates = []
    # Check first 500 attack samples for efficiency
    check_idx = attack_idx[:500]
    for i, idx in enumerate(check_idx):
        x = X_te[idx:idx+1]
        y_s = y_te[idx:idx+1]

        x_con   = pgd(model, x, y_s, epsilon=eps_demo, alpha=alpha,
                      n_steps=40, bounds=bounds,    device=device, random_init=False)
        x_uncon = pgd(model, x, y_s, epsilon=eps_demo, alpha=alpha,
                      n_steps=40, bounds=no_bounds, device=device, random_init=False)

        with torch.no_grad():
            pred_con   = model(torch.tensor(x_con,   dtype=torch.float32).to(device)).item()
            pred_uncon = model(torch.tensor(x_uncon, dtype=torch.float32).to(device)).item()

        con_evades   = pred_con   < 0   # <0 = benign logit
        uncon_evades = pred_uncon < 0

        # Count violations in unconstrained
        n_viol = 0
        for fi in range(bounds.n_num):
            val = float(x_uncon[0, fi])
            if not np.isinf(bounds.lb[fi]) and val < bounds.lb[fi]:
                n_viol += 1
            elif not np.isinf(bounds.ub[fi]) and val > bounds.ub[fi]:
                n_viol += 1

        cat = cats_te[idx] if cats_te is not None else "Unknown"
        candidates.append({
            "dataset_idx":   int(idx),
            "con_evades":    con_evades,
            "uncon_evades":  uncon_evades,
            "n_violations":  n_viol,
            "attack_cat":    cat,
            # Score: prefer samples where unconstrained evades but constrained doesn't
            "score": (2 if (uncon_evades and not con_evades) else
                      1 if uncon_evades else 0) + n_viol * 0.1,
        })
        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(check_idx)} samples…")

    # Pick diverse set: one per attack category where possible
    candidates.sort(key=lambda c: -c["score"])
    seen_cats, fixtures_raw = set(), []
    for c in candidates:
        cat = c["attack_cat"]
        if cat not in seen_cats or len(fixtures_raw) < args.n_fixtures:
            seen_cats.add(cat)
            fixtures_raw.append(c)
        if len(fixtures_raw) >= args.n_fixtures:
            break

    # ── Build fixture JSON ────────────────────────────────────────────────────
    ohe = pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["encoder"]
    fixtures = []
    for fix in fixtures_raw:
        idx = fix["dataset_idx"]
        x_raw_row = X_te_raw.iloc[idx] if hasattr(X_te_raw, "iloc") else None
        x_scaled  = X_te[idx]

        # Original-domain values for display features
        orig_vals = {}
        for feat in DISPLAY_FEATURES:
            if feat in NUM_FEATURES:
                fi = NUM_FEATURES.index(feat)
                orig_vals[feat] = float(x_scaled[fi] * scaler.scale_[fi] + scaler.mean_[fi])

        cat_name = fix["attack_cat"].strip()
        label    = ATTACK_LABELS.get(cat_name, f"{cat_name} Attack")

        fixtures.append({
            "label":            label,
            "description":      f"Real {cat_name} flow from UNSW-NB15 test set. "
                                f"At ε=0.20: unconstrained PGD {'evades ✓' if fix['uncon_evades'] else 'does not evade'}; "
                                f"constrained PGD {'evades ✓' if fix['con_evades'] else 'does not evade'}. "
                                f"{fix['n_violations']} constraint violation(s) in unconstrained output.",
            "features_scaled":  x_scaled.tolist(),
            "feature_names":    NUM_FEATURES,
            "display_features": DISPLAY_FEATURES,
            "scaler_mean":      scaler.mean_.tolist(),
            "scaler_std":       scaler.scale_.tolist(),
            "original_values":  orig_vals,
            "attack_cat":       cat_name,
        })

    with open(data_dir / "fixtures.json", "w") as f:
        json.dump(fixtures, f, indent=2)
    print(f"\nSaved {len(fixtures)} fixture samples to {data_dir}/fixtures.json")

    print("\n── All artifacts ready ──────────────────────────────────────────────────")
    print(f"  {model_dir}/weights.pt")
    print(f"  {model_dir}/scaler.pkl")
    print(f"  {model_dir}/bounds.pkl")
    print(f"  {data_dir}/fixtures.json")
    print("\nNow run:  flask --app app run  (or  python app.py)")


if __name__ == "__main__":
    main()
