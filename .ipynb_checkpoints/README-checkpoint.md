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
notebooks/  all training code: 01 subsample, 02 CNN + SVC, 03 YOLO classification, 04-05 future detector
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

# 1. subsample, 2. train both models: run the notebooks in order
jupyter lab notebooks/      # 01_subsample_dataset, 02_cnn_svc, 03_yolo_classification

# 3. serve
cp .env.example .env
docker compose up --build        # backend :8000, frontend :8080
# or without Docker:
cd backend && pip install -r requirements-dev.txt && uvicorn app.main:app --reload
```

Notebook settings are in the second cell. Keep `DATA`, `PER_CLASS` and `SEED` identical in 02 and 03 so both
models use the same split. Start Jupyter from the project root (or `notebooks/`).

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
- The split comes from `select_and_split` (shared cell in notebooks 02 to 05); it only holds if `DATA`,
  `PER_CLASS` and `SEED` are identical in 02 and 03.
- YOLO flips are disabled during training: a mirrored MODI character is a different character.
- The SVC gets probabilities from sigmoid calibration. Train and serve with the same scikit-learn
  version, since the SVC is stored with joblib.
- The dataset folder layout may be flat or grouped (digits / consonants / vowels); the subsample
  notebook flattens it into one folder per character.
- Split is random per source image. If writer IDs exist, split by writer.
- Not in the current scope, kept for later: notebooks 04 and 05 (synthetic
  multi-character scenes and a YOLO detector that reads a line of text).
