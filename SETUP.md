# SENTINEL-X — Setup & Run Guide

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Download Age & Gender Caffe Models (optional but recommended)
Place these 4 files in the same folder as `app.py`:

| File | Link |
|------|------|
| `age_net.caffemodel`    | https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel |
| `age_deploy.prototxt`   | https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_deploy.prototxt |
| `gender_net.caffemodel` | https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel |
| `gender_deploy.prototxt`| https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_deploy.prototxt |

## 3. Run
```bash
streamlit run app.py
```

## 4. Default Login
- **Username:** `admin`
- **Password:** `admin123`

---

## Feature Overview

| Page | What it does |
|------|-------------|
| 📊 Dashboard | System health, recent alerts, quick-start guide |
| 🎥 Live Detection | Webcam stream with **LBPH + DeepFace hybrid** classification |
| 🧠 Train Model | Upload / capture intruder & safe faces, train LBPH |
| 🖼️ Gallery | Browse, download, delete all snapshots |
| 👤 Age & Gender | Demographic analysis on intruder snapshots |

## Detection Logic

```
For each detected face:
  ├─ Layer 1 (LBPH) — fast, runs first
  │    LBPH confidence < threshold → "Intruder"
  │    LBPH confidence ≥ threshold → "Safe"
  │    No model trained            → Unknown → proceed to Layer 2
  │
  └─ Layer 2 (DeepFace Facenet512) — deeper, runs if needed
       cosine similarity ≥ df_threshold → match label
       No match                         → "Unknown"
```

## Notes
- DeepFace is **optional** — app works with LBPH only if not installed
- Age/Gender models are **optional** — detection still works without them
- All snapshots auto-save at configurable intervals during live detection
- Users stored in `users.json` with SHA-256 hashed passwords
