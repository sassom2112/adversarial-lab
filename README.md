# Adversarial NIDS Lab

Interactive demo for the constraint inflation paper — select a real network flow,
choose a perturbation budget, and watch constrained vs. unconstrained PGD attack
a trained NIDS classifier in real time.

Live at: `https://adversarial-lab.di-sasso.com` *(deploy instructions below)*

---

## What it shows

Unconstrained PGD (standard evaluation) is free to push features into physically
impossible territory — negative TTLs, sub-zero packet counts, rates outside [0,1].
Constrained PGD clips features to documented domain bounds after every gradient
step and stays within valid traffic space.

The gap between their evasion rates at ε=0.20 on UNSW-NB15 is **+72 percentage points**.
This demo lets you see that gap on individual flows and inspect exactly which
features go impossible.

---

## Setup

**Step 1 — Generate artifacts** (run once, in Colab or locally with UNSW-NB15 data):

```bash
# In Colab with the parquet uploaded:
python scripts/prepare_artifacts.py \
    --parquet /content/drive/MyDrive/Colab\ Notebooks/netadv-data/traffic_cleaned.parquet \
    --out-dir /content/adversarial-lab
```

This produces:
- `model/weights.pt` — trained MLP (3-layer, 256-128-64)
- `model/scaler.pkl` — fitted StandardScaler
- `model/bounds.pkl` — constraint bounds (lb/ub in preprocessed space)
- `data/fixtures.json` — 8 pre-selected attack samples with metadata

**Step 2 — Run locally:**

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5050
```

**Step 3 — Deploy to Render:**

1. Push this repo to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Upload `model/` and `data/` via Render's file system or store in the repo

---

## Related

- [catt-ccs](https://github.com/sassom2112/catt-ccs) — the full paper and benchmark suite
- [network-intrusion-detection](https://github.com/sassom2112/network-intrusion-detection) — the original IDS project
