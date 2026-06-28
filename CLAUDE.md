# adversarial-lab — Claude Context

## What this is
Interactive demo for the constraint-inflation paper (Python/Flask web): pick a real network flow,
choose a perturbation budget, watch constrained vs unconstrained PGD attack a trained NIDS classifier
in real time. Live at `https://adversarial-lab.di-sasso.com`. Part of the **adversarial-ML research**
workstream (Cowork Space: `adversarial-ml-research`); the public-facing face of `catt`/`catt-ccs`.

## Layout
- `app.py` — Flask app; `model/` — trained NIDS + artifacts; `static/`, `templates/` — web UI
- `scripts/` — deploy; `requirements.txt`

## Run
`pip install -r requirements.txt` ; `python app.py`  (then open the local URL)

## Session continuity (per-repo)
Commit trailers per the global protocol.

## Note
The WSL working copy is on `master` with ~14 uncommitted files (reconcile separately). This is the
deployed demo — keep source ↔ live (di-sasso.com) in sync.
