# MODI character recognition

Real-time recognition of one MODI character at a time. Two models are trained on the same split of
MODI-HChar and compared:

- **CNN + SVC:** a small CNN extracts a 128-d feature vector; an sklearn SVC classifies it.
- **YOLO:** an Ultralytics YOLO classification model, fine-tuned from pretrained weights.

Both are served through FastAPI and a drawing-canvas page. Built for the Foundations of Machine Learning
mini project; designed to grow into a longer research project (multi-character detection).

## Layout

```
backend/    FastAPI app (app/), inference wrappers (ml/), Dockerfile, tests
training/   subsample, CNN + SVC, YOLO classification (+ future detection scripts)
frontend/   static drawing-canvas page (HTML/JS)
data/       raw/ and processed/ (git-ignored)
models/     trained weights (git-ignored)
reports/    metrics, confusion matrices, curves
docs/       architecture and AWS deployment notes
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. subsample (use --max-classes 20 for a first trial, drop it for all 57 classes)
python training/subsample_dataset.py --src data/raw/modi_hchar_full \
    --dst data/processed/offline_subset --per-class 300 --max-classes 20

# 2. train both models (same --data, --per-class and --seed = same split)
python training/train_cnn.py --data data/processed/offline_subset
python training/train_yolo_cls.py --data data/processed/offline_subset

# 3. serve
cp .env.example .env
docker compose up --build        # backend :8000, frontend :8080
# or without Docker:
cd backend && pip install -r requirements-dev.txt && uvicorn app.main:app --reload
```

Run tests: `cd backend && pytest`.

Outputs: `models/cnn.pth`, `models/svc.joblib`, `models/yolo_cls.pt`, and metrics, confusion matrices
and curves under `reports/`. The comparison table is `reports/cnn_svc/metrics.json` against
`reports/yolo_cls/metrics.json` (test accuracy and ms per image).

## Status

- [x] Backend, inference wrappers, drawing-canvas frontend, Docker
- [x] Subsampling, CNN + SVC and YOLO classification training scripts
- [ ] Train both models on real data and fill in `reports/`
- [ ] Deploy on AWS (see `docs/aws_deploy.md`)
- [ ] Report: CNN + SVC vs YOLO

## Notes

- Both models see the same preprocessing: tight crop around the ink, square pad, resize
  (`backend/ml/preprocessing.py`). The canvas page sends the drawing as a PNG (black pen on white).
  Stroke thickness relative to the character matters to the models, so use the pen-size slider to
  match the look of the training images.
- The split is shared through `training/common.py`; it only holds if `--data`, `--per-class` and
  `--seed` are identical across the two training scripts.
- YOLO flips are disabled during training: a mirrored MODI character is a different character.
- The SVC gets probabilities from sigmoid calibration. Train and serve with the same scikit-learn
  version, since the SVC is stored with joblib.
- The dataset folder layout may be flat or grouped (digits / consonants / vowels); the subsample
  script flattens it into one folder per character.
- Split is random per source image. If writer IDs exist, split by writer.
- Not in the current scope, kept for later: `generate_scenes.py` and `train_yolo.py` (synthetic
  multi-character scenes and a YOLO detector that reads a line of text).
