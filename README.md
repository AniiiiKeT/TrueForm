# TrueForm

> An AI-driven archery posture analysis and feedback system powered by pose estimation, biomechanics, and 3D visualization.

---

## Table of contents

1. Project overview
2. Features
3. Quick start
4. Installation
5. Repository layout
6. How to run (recommended workflows)
7. Notebooks and experiments
8. Models, logs and data
9. Development notes & architecture
10. Contributing
11. License & contact
12. TODO / next steps for the repo owner

---

## 1. Project overview

**TrueForm** is an experimental system that analyses an archer's posture and movement using pose-estimation, extracts biomechanical metrics, and provides visual feedback (2D/3D). The project collects pose keypoints, computes metrics (joint angles, velocities, symmetry measures, etc.), and supports visualization and model-based analysis.

Use this document as a living README — it explains how to set up, explore the notebooks, run experiments, and contribute.

---

## 2. Key features

* Pose estimation pipeline (keypoint extraction)
* Biomechanical metric computation (joint angles, range-of-motion, temporal segmentation)
* 2D overlay and 3D visualization utilities
* Notebooks for experiments, visualisation and model training
* Saved model artifacts and logs (in `models/` and `logs/`)

---

## 3. Quick start

1. Clone the repository:

```bash
git clone https://github.com/AniiiiKeT/TrueForm.git
cd TrueForm
```

2. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Start JupyterLab / Jupyter Notebook and open the notebooks in the `notebooks/` folder for runnable demos:

```bash
jupyter lab
```

4. Inspect `src/` for core scripts and utilities (pose extraction, metric computation, visualization). Run scripts or notebooks to reproduce analyses.

---

## 4. Installation & requirements

* Python 3.8+ recommended.
* Installed packages are listed in `requirements.txt`. Typical dependencies for this type of project include: `opencv-python`, `mediapipe` (or other pose-estimation libraries), `numpy`, `pandas`, `matplotlib`, and ML/visualization libraries.

If you have GPU-accelerated model training components, install appropriate TensorFlow / PyTorch builds that match your CUDA version.

---

## 5. Repository layout

```
TrueForm/
├─ archive/         # historical/backup files
├─ logs/            # training/processing logs
├─ models/          # trained model artifacts and checkpoints
├─ notebooks/       # Jupyter notebooks: experiments & demos
├─ src/             # core source code (pose processing, metrics, visualization)
├─ .gitignore
├─ LICENSE
├─ README.md        # short project summary (this repo)
├─ requirements.txt
```

### Notes on important folders

* `src/` should contain modularized code for: data ingestion, pose estimation wrappers, metric computation, visualization utilities, and any model training/evaluation scripts.
* `notebooks/` is the best place to see example workflows and to reproduce experiments.
* `models/` should include versioned artifacts or a README describing how models were produced and how to load them.

---

## 6. How to run (recommended workflows)

### Option A — Explore with notebooks (recommended)

1. Open `notebooks/` in JupyterLab.
2. Run cells sequentially. Typical notebooks will show: data loading → pose extraction → metric computation → visualization.

### Option B — Run scripts (for automation)

* Identify the script in `src/` that performs pose extraction (e.g., `extract_pose.py` or similar). Typical usage:

```bash
python src/extract_pose.py --input data/video.mp4 --output data/keypoints.npy
```

* Then run metric computation script:

```bash
python src/compute_metrics.py --keypoints data/keypoints.npy --out results.csv
```

> *Note*: Replace script names and flags with the actual names present in `src/`. If the repo owner wants, these example commands can be made exact — I left them generic to avoid accidental mismatches.

---

## 7. Notebooks and experiments

The notebooks folder contains interactive analysis and visualizations. Typical recommended steps when working with notebooks:

1. Create a copy of any example notebook before editing.
2. Run the notebook top-to-bottom to ensure reproducibility.
3. Save outputs (figures, CSVs) under a dedicated `outputs/` folder to keep notebooks reproducible.

---

## 8. Models, logs and data

* `models/`: store model weights and provide a minimal `models/README.md` describing format, input signature, and how to load the model.
* `logs/`: training and processing logs. Use consistent naming (e.g., `exp-YYYYMMDD-runX/`).

**Recommendation:** Add a small script `models/load_model.py` with an example showing how to load weights and run inference on a single frame or keypoint sequence.

---

## 9. Development notes & architecture

Suggested modular architecture:

* `src/data/` — I/O utilities (video → frames → keypoints)
* `src/pose/` — wrappers for pose-estimation backends (MediaPipe, OpenPose, etc.)
* `src/metrics/` — biomechanical metrics (joint angles, ROM, temporal segmentation)
* `src/visualization/` — 2D overlays + 3D viewer utilities
* `src/models/` — model training, evaluation, inference code
* `src/utils.py` — shared utilities and configuration loader

**Testing:** Add unit tests for metric functions (angle calculations, smoothing) to `tests/` to avoid regressions.

---

## 10. License & contact

This repository is licensed under the MIT license (see `LICENSE`).

---

## 11. TODO


* [ ] Provide exact script names and example command-lines in section **5**.
* [ ] Add short descriptions for each `src/` file (function-level docstrings help here).
* [ ] Add a `models/README.md` describing model artifact format and how to reproduce them.
* [ ] Add a `CONTRIBUTING.md` and tests (optional but recommended).

---
