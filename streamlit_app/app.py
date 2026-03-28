"""
Surface Defect Detector - Streamlit App
========================================
This app lets you:
  1. Upload an image and detect defects.
  2. Upload a video and detect defects frame by frame.
  3. Use your webcam for real-time detection.
  4. Download a CSV report of all detections.

Run with:  streamlit run streamlit_app/app.py
"""

import streamlit as st          # The web framework that creates the UI
import cv2                      # OpenCV: reads/writes images and video
import numpy as np              # Numerical operations on image arrays
import pandas as pd             # For building the CSV report table
import os                       # For file path operations
import tempfile                 # For saving uploaded video to a temp file
from PIL import Image           # Pillow: converts between image formats
from datetime import datetime   # For timestamping detections
from ultralytics import YOLO    # The YOLOv8 detection framework

# ─────────────────────────────────────────────────────────
# CONFIGURATION (FIXED)
# ─────────────────────────────────────────────────────────

import os
import streamlit as st
from ultralytics import YOLO

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Possible model locations (priority order)
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "best.pt"),  # ✅ your trained model (recommended)
    os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),  # training output
    os.path.join(BASE_DIR, "notebooks", "yolov8n.pt"),  # fallback model
]

# Class names must match the order used during training
CLASS_NAMES = ["scratch", "dent", "crack", "missing_part"]

# Bounding box / highlight color per class (BGR format for OpenCV)
CLASS_COLORS = {
    "scratch":      (255, 100,   0),
    "dent":         ( 50, 200,  50),
    "crack":        (  0,  50, 255),
    "missing_part": (  0, 220, 220),
}

# Default confidence threshold
CONFIDENCE_THRESHOLD = 0.10


# ─────────────────────────────────────────────────────────
# MODEL LOADING (FIXED)
# ─────────────────────────────────────────────────────────

def get_valid_model_path():
    """Find first valid model file."""
    for path in MODEL_CANDIDATES:
        if os.path.isfile(path):   # ✅ ensures it's a FILE (not folder)
            return path
    return None


@st.cache_resource
def load_model() -> YOLO:
    """
    Load YOLO model safely (no crash even if path is wrong).
    """
    model_path = get_valid_model_path()

    if model_path:
        st.success(f"✅ Model loaded from: {model_path}")
        return YOLO(model_path)

    # Fallback (never crash)
    st.warning("⚠️ No trained model found. Using default YOLOv8n.")
    return YOLO("yolov8n.pt")


# ─────────────────────────────────────────────────────────
# IMAGE CONVERSION HELPERS
# ─────────────────────────────────────────────────────────

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for display in Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV-compatible numpy array (BGR)."""
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────

def run_inference_on_frame(model: YOLO, frame: np.ndarray, conf: float = CONFIDENCE_THRESHOLD):
    """
    Run defect detection on a single image frame.

    Args:
        model:  The loaded YOLO model.
        frame:  A numpy array representing an image (BGR format from OpenCV).
        conf:   Confidence threshold — detections below this are ignored.

    Returns:
        annotated_frame: The image with plain bounding boxes drawn on it.
        detections:      A list of dicts, one per detected defect.
    """
    results         = model(frame, conf=conf, verbose=False)
    annotated_frame = frame.copy()
    detections      = []
    result          = results[0]

    for box in result.boxes:
        class_id   = int(box.cls[0])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(class_name, (200, 200, 200))

        # Draw bounding box outline
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=2)

        # Draw label with background
        label = f"{class_name} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated_frame, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), thickness=1)

        detections.append({
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    return annotated_frame, detections


def draw_highlighted_detections(frame: np.ndarray, detections: list, alpha: float = 0.35) -> np.ndarray:
    """
    Draw semi-transparent filled highlights over each detected defect area.
    Much cleaner than plain boxes when there are many detections on one image.

    How it works:
      - We draw solid filled rectangles on a copy (overlay).
      - We draw crisp borders + labels on another copy (output).
      - We blend them: output = alpha*overlay + (1-alpha)*output
      - Result: colored glow inside each box, solid border outside.

    Args:
        frame:      Original BGR image (not modified).
        detections: List of detection dicts from run_inference_on_frame().
        alpha:      Transparency of the highlight fill (0.0=invisible, 1.0=solid).
                    0.35 gives a nice visible glow without hiding the defect.

    Returns:
        Annotated image with colored semi-transparent highlights and labels.
    """
    overlay = frame.copy()   # Will hold filled color rectangles
    output  = frame.copy()   # Will hold borders + labels

    for det in detections:
        x1, y1 = det['x1'], det['y1']
        x2, y2 = det['x2'], det['y2']
        cls    = det['class_name']
        conf   = det['confidence']
        color  = CLASS_COLORS.get(cls, (200, 200, 200))

        # Filled rectangle on overlay → becomes the transparent highlight
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

        # Crisp border on output
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)

        # Label background + text on output
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Blend: alpha = strength of the fill highlight
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def get_display_frame(frame: np.ndarray, detections: list, highlight_mode: bool) -> np.ndarray:
    """
    Return the right annotated frame based on the display mode toggle.

    Args:
        frame:          Original BGR image.
        detections:     List of detection dicts.
        highlight_mode: True = highlight fills, False = plain boxes.

    Returns:
        Annotated frame ready for Streamlit display.
    """
    if highlight_mode:
        return draw_highlighted_detections(frame, detections)

    # Plain box mode — redraw boxes on a clean copy
    plain = frame.copy()
    for det in detections:
        color = CLASS_COLORS.get(det['class_name'], (200, 200, 200))
        cv2.rectangle(plain, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(plain, (det['x1'], det['y1'] - th - 6),
                      (det['x1'] + tw + 4, det['y1']), color, -1)
        cv2.putText(plain, label, (det['x1'] + 2, det['y1'] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return plain


# ─────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────

def generate_csv_report(all_detections: list) -> bytes:
    """
    Convert detection records into a downloadable CSV file.

    Args:
        all_detections: List of detection dicts.

    Returns:
        CSV content as bytes for st.download_button.
    """
    if not all_detections:
        df = pd.DataFrame(columns=["frame_id", "source_name", "class_name",
                                   "confidence", "timestamp", "x1", "y1", "x2", "y2"])
    else:
        df = pd.DataFrame(all_detections)
    return df.to_csv(index=False).encode("utf-8")


def build_summary_table(all_detections: list) -> pd.DataFrame:
    """
    Build a count of how many times each class was detected.

    Args:
        all_detections: List of detection dicts.

    Returns:
        DataFrame with columns: class_name, count.
    """
    if not all_detections:
        return pd.DataFrame({"class_name": CLASS_NAMES, "count": [0] * len(CLASS_NAMES)})
    df      = pd.DataFrame(all_detections)
    summary = df["class_name"].value_counts().reset_index()
    summary.columns = ["class_name", "count"]
    return summary


# ─────────────────────────────────────────────────────────
# MODE HANDLERS
# ─────────────────────────────────────────────────────────

def handle_image_upload(model: YOLO, conf_thresh: float, highlight_mode: bool):
    """
    Handle the 'Upload Image' mode.

    Args:
        model:          Loaded YOLO model.
        conf_thresh:    Confidence threshold from sidebar.
        highlight_mode: Whether to use highlight fills or plain boxes.

    Returns:
        List of detection dicts for the report.
    """
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a photo of a metal part or PCB to detect defects."
    )

    if uploaded_file is None:
        st.info("👆 Upload an image above to get started.")
        return []

    pil_image = Image.open(uploaded_file).convert("RGB")
    bgr_frame = pil_to_bgr(pil_image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(pil_image, use_container_width=True)

    with st.spinner("🔍 Detecting defects..."):
        _, detections = run_inference_on_frame(model, bgr_frame, conf_thresh)

    display_frame = get_display_frame(bgr_frame, detections, highlight_mode)

    with col2:
        mode_label = "✨ Highlighted" if highlight_mode else "🔎 Detected"
        st.subheader(f"{mode_label}: {len(detections)} defect(s)")
        st.image(bgr_to_rgb(display_frame), use_container_width=True)

    timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_records = []
    for det in detections:
        report_records.append({
            "frame_id":    0,
            "source_name": uploaded_file.name,
            "class_name":  det["class_name"],
            "confidence":  det["confidence"],
            "timestamp":   timestamp,
            "x1": det["x1"], "y1": det["y1"],
            "x2": det["x2"], "y2": det["y2"],
        })

    return report_records


def handle_video_upload(model: YOLO, conf_thresh: float, highlight_mode: bool):
    """
    Handle the 'Upload Video' mode.
    Processes every 3rd frame for speed on CPU.

    Args:
        model:          Loaded YOLO model.
        conf_thresh:    Confidence threshold.
        highlight_mode: Whether to use highlight fills or plain boxes.

    Returns:
        List of detection dicts for the report.
    """
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a short video (< 60s recommended for speed on CPU)."
    )

    if uploaded_video is None:
        st.info("👆 Upload a video above to get started.")
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    cap          = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    st.write(f"📹 Video info: **{total_frames} frames** at **{fps:.1f} FPS**")
    st.write("⏱️ Processing every 3rd frame to save time on CPU...")

    frame_display  = st.empty()
    progress_bar   = st.progress(0)
    status_text    = st.empty()
    report_records = []
    frame_idx      = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % 3 != 0:
            continue

        _, detections     = run_inference_on_frame(model, frame, conf_thresh)
        display_frame_img = get_display_frame(frame, detections, highlight_mode)

        frame_display.image(
            bgr_to_rgb(display_frame_img),
            caption=f"Frame {frame_idx} — {len(detections)} detection(s)",
            use_container_width=True
        )

        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
        status_text.text(f"Processing frame {frame_idx}/{total_frames}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for det in detections:
            report_records.append({
                "frame_id":    frame_idx,
                "source_name": uploaded_video.name,
                "class_name":  det["class_name"],
                "confidence":  det["confidence"],
                "timestamp":   timestamp,
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
            })

    cap.release()
    os.unlink(tmp_path)
    progress_bar.progress(1.0)
    status_text.success("✅ Video processing complete!")

    return report_records


def handle_webcam(model: YOLO, conf_thresh: float, highlight_mode: bool):
    """
    Handle the 'Webcam' mode.
    Captures a single photo from the webcam and runs detection.

    Args:
        model:          Loaded YOLO model.
        conf_thresh:    Confidence threshold.
        highlight_mode: Whether to use highlight fills or plain boxes.

    Returns:
        List of detection dicts for the report.
    """
    st.info("📷 Your browser will ask for camera permission. Click 'Allow'.")

    camera_photo = st.camera_input("Take a photo to detect defects")

    if camera_photo is None:
        return []

    pil_image = Image.open(camera_photo).convert("RGB")
    bgr_frame = pil_to_bgr(pil_image)

    with st.spinner("🔍 Detecting defects..."):
        _, detections = run_inference_on_frame(model, bgr_frame, conf_thresh)

    display_frame = get_display_frame(bgr_frame, detections, highlight_mode)

    mode_label = "✨ Highlighted" if highlight_mode else "🔎 Detected"
    st.subheader(f"{mode_label}: {len(detections)} defect(s)")
    st.image(bgr_to_rgb(display_frame), use_container_width=True)

    timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_records = []
    for det in detections:
        report_records.append({
            "frame_id":    0,
            "source_name": "webcam_capture",
            "class_name":  det["class_name"],
            "confidence":  det["confidence"],
            "timestamp":   timestamp,
            "x1": det["x1"], "y1": det["y1"],
            "x2": det["x2"], "y2": det["y2"],
        })

    return report_records


# ─────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────

def main():
    """Entry point — sets up the page, sidebar, and calls the right handler."""

    st.set_page_config(
        page_title="Surface Defect Detector",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Surface Defect Detector")
    st.markdown(
        "Detect **scratches, dents, cracks, and missing parts** on metal and PCB surfaces "
        "using a YOLOv8 deep learning model."
    )
    st.divider()

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Input Source",
            options=["📁 Upload Image", "🎥 Upload Video", "📷 Webcam"],
            index=0,
        )

        st.divider()

        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=0.90,
            value=CONFIDENCE_THRESHOLD,
            step=0.01,
            help=(
                "Minimum confidence to show a detection.\n\n"
                "• Recommended for 20-epoch model: 0.10\n"
                "• After 50+ epoch retraining: 0.25\n"
                "• Too low → false detections\n"
                "• Too high → misses real defects"
            ),
        )

        st.divider()

        # ── Highlight mode toggle ─────────────────────────
        highlight_mode = st.toggle(
            "✨ Highlight Mode",
            value=True,
            help=(
                "ON  → Semi-transparent colored fills inside each bounding box.\n"
                "      Best when there are many overlapping detections.\n\n"
                "OFF → Classic outline boxes only."
            )
        )

        if highlight_mode:
            st.caption("🎨 Mode: colored highlight fills + labels")
        else:
            st.caption("📦 Mode: plain bounding boxes + labels")

        st.divider()

        # Class color legend
        st.markdown("**Classes Detected:**")
        for cls in CLASS_NAMES:
            color_bgr = CLASS_COLORS[cls]
            hex_color = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])
            st.markdown(
                f"<span style='color:{hex_color}; font-size:18px;'>■</span> `{cls}`",
                unsafe_allow_html=True
            )

        st.divider()
        st.caption("Model: YOLOv8n  |  4 defect classes")

    # ── Load model ───────────────────────────────────────
    with st.spinner("Loading model..."):
        model = load_model()

    # ── Dispatch to correct mode ─────────────────────────
    all_detections = []

    if mode == "📁 Upload Image":
        all_detections = handle_image_upload(model, conf_thresh, highlight_mode)
    elif mode == "🎥 Upload Video":
        all_detections = handle_video_upload(model, conf_thresh, highlight_mode)
    elif mode == "📷 Webcam":
        all_detections = handle_webcam(model, conf_thresh, highlight_mode)

    # ── Report Section ───────────────────────────────────
    st.divider()
    st.header("📊 Detection Report")

    if not all_detections:
        st.info("No defects detected yet. Upload an image or video to generate a report.")
    else:
        # Class count metrics
        summary_df  = build_summary_table(all_detections)
        st.subheader("Class Count Summary")
        metric_cols = st.columns(len(CLASS_NAMES))
        for i, cls in enumerate(CLASS_NAMES):
            row   = summary_df[summary_df["class_name"] == cls]
            count = int(row["count"].values[0]) if not row.empty else 0
            metric_cols[i].metric(label=cls, value=count)

        # Full detections table
        st.subheader("All Detections")
        st.caption(
            "x1,y1 = top-left corner of defect box  |  "
            "x2,y2 = bottom-right corner  |  values in pixels"
        )
        st.dataframe(pd.DataFrame(all_detections), use_container_width=True)

        # Download button
        csv_bytes = generate_csv_report(all_detections)
        st.download_button(
            label="⬇️ Download Report as CSV",
            data=csv_bytes,
            file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption("Surface Defect Detector  |  Built with YOLOv8 + Streamlit  |  College Project")


if __name__ == "__main__":
    main()
