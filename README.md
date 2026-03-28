# 🔍 Surface Defect Detector

A YOLOv8-based surface defect detection system for metal components and PCBs, with a Streamlit web app for real-time inference.

---

## 📁 Project Structure

```
surface_defect_detector/
├── notebooks/
│   └── defect_detection_pipeline.ipynb   # Full pipeline: data → train → inference
├── models/
│   └── yolov8n_defects.pt                # Trained model weights (generated after training)
├── data/
│   ├── metal/                            # NEU-DET metal defect images + labels
│   ├── pcb/                              # PCB defect images + labels
│   └── merged/                           # Merged dataset in YOLO format
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── defects.yaml                  # YOLOv8 dataset config
├── streamlit_app/
│   └── app.py                            # Streamlit web app
├── reports/
│   └── defect_report.csv                 # Generated detection reports
└── README.md
```

---

## 🚀 Quick Setup Guide

### A. Set Up the Environment

**Step 1: Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 2: Install required packages**
```bash
pip install ultralytics opencv-python streamlit pillow pandas
```

**What each package does:**
- `ultralytics` — The YOLOv8 framework for training and running object detection models.
- `opencv-python` — Used to read, process, and display images and videos.
- `streamlit` — Turns a Python script into an interactive web app (no HTML needed).
- `pillow` — Image processing library used by Streamlit for image display.
- `pandas` — Used to create and export the CSV detection report.

---

### B. Download and Prepare Datasets

#### Metal Dataset — NEU-DET
- **What it contains:** Steel surface defects — crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.
- **Download from Kaggle:**
  1. Go to: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
  2. Download and unzip into: `data/metal/`

  Or use the Kaggle API:
  ```bash
  pip install kaggle
  kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p data/metal/ --unzip
  ```

#### PCB Dataset — DeepPCB
- **What it contains:** PCB defects — open, short, mousebite, spur, copper, pin-hole.
- **Download from GitHub:**
  ```bash
  git clone https://github.com/tangsanli5201/DeepPCB.git data/pcb/DeepPCB
  ```
  Or from Kaggle: https://www.kaggle.com/datasets/akhatova/pcb-defects

---

### C. Data Processing & Label Mapping

Run **Cell 4** of the Jupyter notebook to:
1. Read original label files from both datasets.
2. Remap class names to our 4 unified classes.
3. Save everything in YOLO format under `data/merged/`.

**Class Mapping:**
| Original Label | Unified Class | Class ID |
|---|---|---|
| scratch, scuff, seam, crazing | scratch | 0 |
| crack, fissure, fracture | crack | 1 |
| pit, pitted_surface, inclusion, dent | dent | 2 |
| missing_part, open, broken_edge, pin-hole | missing_part | 3 |

---

### D. Train the YOLOv8 Model

Run **Cell 5** of the notebook. Training takes ~10–30 minutes on a laptop CPU for 10 epochs.

```bash
# Or run directly from terminal:
yolo detect train data=data/merged/defects.yaml model=yolov8n.pt epochs=10 imgsz=640
```

After training, the best model is saved automatically. Copy it:
```bash
cp runs/detect/train/weights/best.pt models/yolov8n_defects.pt
```

---

### E. Run Inference

**On an image:**
```python
from ultralytics import YOLO
model = YOLO("models/yolov8n_defects.pt")
results = model("path/to/image.jpg")
results[0].show()
```

**On a video:** Run **Cell 7** of the notebook.

---

### F. Run the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Then open your browser to: **http://localhost:8501**

**App features:**
- 📁 Upload an image → see bounding boxes + class labels
- 🎥 Upload a video → frame-by-frame detection
- 📷 Use webcam → real-time live detection
- 📊 View detection report → download as CSV

---

## 🧠 Classes Detected

| ID | Class | Examples |
|----|-------|---------|
| 0 | scratch | Surface scratches, scuffs, seams |
| 1 | crack | Cracks, fissures, fractures |
| 2 | dent | Pits, dents, inclusions |
| 3 | missing_part | Missing parts, open circuits, broken edges |

---

## 📋 Requirements

- Python 3.8+
- No GPU required (GPU makes training much faster)
- ~2GB disk space for datasets
