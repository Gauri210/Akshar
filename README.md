# MODI script recognition

Real-time recognition of MODI characters. The main system is a YOLO detector; it is benchmarked
against two baselines: Fisher LDA on raw pixels, and a CNN whose output layer is a linear SVM
(squared hinge loss). Served through FastAPI and a webcam frontend. Built for the Foundations of
Machine Learning mini project; designed to grow into a longer research project.

## Layout

```
backend/    FastAPI app (app/) and inference wrappers (ml/), Dockerfile, tests
training/   subsample, baselines, scene generator, YOLO training
frontend/   static webcam page (HTML/JS)
data/       raw/ and processed/ (git-ignored)
models/     trained weights (git-ignored)
reports/    metrics, confusion matrices, curves
docs/       architecture and AWS deployment notes
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. subsample (15-20 classes first)
python training/subsample_dataset.py --src data/raw/modi_hchar_full \
    --dst data/processed/offline_subset --per-class 300 --max-classes 20

# 2. baselines (LDA, then CNN with SVM head; add --head softmax for the ablation)
python training/train_baseline.py --data data/processed/offline_subset
python training/train_cnn.py --data data/processed/offline_subset

# 3. YOLO on synthetic scenes (same --per-class and --seed as step 2 = same split)
python training/generate_scenes.py --src data/processed/offline_subset
python training/train_yolo.py

# 4. serve
cp .env.example .env
docker compose up --build        # backend :8000, frontend :8080
# or without Docker:
cd backend && pip install -r requirements-dev.txt && uvicorn app.main:app --reload
```

Run tests: `cd backend && pytest`.

## Status

- [x] Backend, inference wrappers, frontend, Docker
- [x] Subsampling, baseline, CNN, scene generator, YOLO training scripts
- [ ] Train all models and fill in `reports/`
- [ ] Real-photo test set (50-100 images) to measure the synthetic-to-real gap
- [ ] Deploy on AWS (see `docs/aws_deploy.md`)
- [ ] Report: LDA vs CNN (SVM head) vs YOLO

## Notes and known gaps

- LDA, the CNN and the scene generator share `training/common.py`, so a source image is in the same
  train/val/test partition for every method. This only holds if `--data`/`--src`, `--per-class` and
  `--seed` are identical across the scripts.
- The CNN reports an SVM decision score, not a probability. The API shows `(score + 1) / 2`, clipped to
  [0, 1], as a display-only confidence; the predicted label comes from the raw scores.
- The classifier endpoints (baseline, CNN) expect one character in the frame. Multi-character scenes go through YOLO.
- The dataset folder layout is assumed to be one folder per class. Adjust if MODI-HChar differs.
- Split is random per source image. If writer IDs exist, split by writer.
