"""Real-Time Sign Language Recognition System Using AI and Computer Vision.

Interactive Streamlit Application featuring real-time webcam hand detection,
MediaPipe landmark extraction, Scikit-learn gesture classification, temporal
sentence construction, and text-to-speech synthesis.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    DATASET_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    GESTURE_CLASSES,
    CONFIDENCE_THRESHOLD,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    USE_3D_LANDMARKS
)
from src.hand_detection import HandDetector
from src.gesture_classifier import GestureClassifier
from src.text_generator import TextGenerator
from src.speech_generator import SpeechGenerator


# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="SignSense AI - Real-Time Sign Language Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM DARK FUTURISTIC CSS STYLING
# ==========================================
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.75);
        --accent-cyan: #00d4ff;
        --accent-purple: #7928ca;
        --accent-indigo: #6366f1;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --border-glass: rgba(255, 255, 255, 0.08);
    }

    body, .stApp {
        background-color: var(--bg-primary);
        font-family: 'Outfit', sans-serif;
        color: var(--text-primary);
    }

    /* Top Header Bar */
    .hero-header {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(30, 27, 75, 0.8) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 212, 255, 0.4);
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--accent-cyan);
        font-family: 'Outfit', sans-serif;
    }

    /* Gesture Prediction Display */
    .prediction-box {
        background: radial-gradient(circle at top, rgba(99, 102, 241, 0.15) 0%, rgba(17, 24, 39, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }

    .gesture-large {
        font-size: 5rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.7);
        line-height: 1;
        margin: 10px 0;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .status-idle {
        background-color: rgba(156, 163, 175, 0.15);
        color: #9ca3af;
        border: 1px solid rgba(156, 163, 175, 0.3);
    }
    .status-alert {
        background-color: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* Text Builder Card */
    .sentence-card {
        background: #131b2e;
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 14px;
        padding: 18px 24px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .sentence-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        color: #38bdf8;
        font-weight: 600;
        min-height: 48px;
        word-wrap: break-word;
    }

    /* Custom Button styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "text_generator" not in st.session_state:
    st.session_state.text_generator = TextGenerator()

if "speech_generator" not in st.session_state:
    st.session_state.speech_generator = SpeechGenerator()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "last_predicted_letter" not in st.session_state:
    st.session_state.last_predicted_letter = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "tts_status_msg" not in st.session_state:
    st.session_state.tts_status_msg = ""


# ==========================================
# RESOURCE CACHING (DETECTOR & CLASSIFIER)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_hand_detector():
    """Cache hand detector instance."""
    try:
        return HandDetector()
    except Exception as e:
        st.error(f"Failed to initialize MediaPipe Hand Detector: {e}")
        return None

def load_gesture_classifier():
    """Load fresh machine learning model instance with automatic reload."""
    import importlib
    import src.gesture_classifier
    importlib.reload(src.gesture_classifier)
    clf = src.gesture_classifier.GestureClassifier()
    if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
        loaded = clf.load(MODEL_PATH, LABEL_ENCODER_PATH)
        if loaded:
            return clf
    return None


detector = load_hand_detector()
classifier = load_gesture_classifier()


def safe_predict(clf, feat):
    """Safely call classifier predict whether cached instance is old or updated."""
    try:
        res = clf.predict(feat, return_raw=True)
        return res[0], res[1], res[2], res[3]
    except (TypeError, ValueError):
        res = clf.predict(feat)
        p_label, c, p_dict = res[0], res[1], res[2]
        r_label = p_label
        if r_label is None and p_dict:
            r_label = max(p_dict, key=p_dict.get)
        return p_label, c, p_dict, r_label


# ==========================================
# SIDEBAR NAVIGATION & SETTINGS
# ==========================================
with st.sidebar:
    st.image("assets/logo.png" if Path("assets/logo.png").exists() else "🖐️", width=90)
    st.title("SignSense AI")
    st.caption("PBL Academic Project | CSE (AI & ML)")

    st.markdown("---")
    navigation = st.radio(
        "Navigation Menu",
        [
            "Live Recognition Studio",
            "Model Analytics & Evaluation",
            "Dataset & Training Hub",
            "System Architecture & Pipeline",
            "Faculty Presentation Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("System Controls")
    confidence_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Predictions below this confidence level are treated as unconfirmed."
    )
    if classifier:
        classifier.confidence_threshold = confidence_thresh

    auto_confirm = st.checkbox(
        "Auto-Confirm Stabilized Letters",
        value=True,
        help="Automatically appends letter when stably held for 1 second."
    )

    mirror_mode = st.checkbox(
        "Mirror Camera Preview",
        value=True,
        help="Flips video feed horizontally for intuitive natural gesture interaction."
    )

    st.markdown("---")
    st.caption("Developed with MediaPipe Tasks, OpenCV, Scikit-Learn & Streamlit.")


# ==========================================
# TOP HERO HEADER
# ==========================================
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">Real-Time Sign Language Recognition System</h1>
        <p class="hero-subtitle">
            Artificial Intelligence & Computer Vision Powered Gesture-to-Text & Speech Translation System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# TAB 1: LIVE RECOGNITION STUDIO
# ==============================================================================
if navigation == "Live Recognition Studio":
    col_cam, col_pred = st.columns([1.4, 1.0], gap="large")

    # Define right column telemetry placeholders FIRST so they can be updated dynamically
    with col_pred:
        st.subheader("Recognition Telemetry")
        pred_box_slot = st.empty()
        conf_progress_slot = st.empty()
        conf_caption_slot = st.empty()
        st.markdown("---")
        st.markdown("**Supported ASL Static Alphabet:**")
        alphabet_slot = st.empty()

    with col_cam:
        st.subheader("Webcam Vision Feed")
        feed_mode = st.radio(
            "Camera Input Mode",
            ["Continuous Webcam Stream", "Browser Snapshot (Photo)", "Upload Test Image"],
            horizontal=True
        )

        video_placeholder = st.empty()
        status_placeholder = st.empty()

    # Define Sentence Section placeholders FIRST
    st.markdown("---")
    st.subheader("Sentence Constructor & Voice Synthesis")
    sentence_card_slot = st.empty()

    # Helper function to dynamically update telemetry cards & sentence card (throttled)
    def render_telemetry_ui(p_letter, r_letter, c_val, curr_sent):
        if p_letter:
            b_cls = "status-active"
            b_txt = "GESTURE CONFIRMED"
            disp_char = p_letter
        elif r_letter and c_val >= 0.30:
            b_cls = "status-idle"
            b_txt = f"DETECTING: {r_letter} (HOLD STEADY)"
            disp_char = r_letter
        else:
            b_cls = "status-alert"
            b_txt = "WAITING FOR GESTURE"
            disp_char = "—"

        pred_box_slot.markdown(
            f"""
            <div class="prediction-box">
                <div class="metric-label">Predicted Gesture Class</div>
                <div class="gesture-large">{disp_char}</div>
                <div>
                    <span class="status-badge {b_cls}">
                        {b_txt}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        conf_progress_slot.progress(min(max(float(c_val), 0.0), 1.0))
        conf_caption_slot.caption(
            f"Estimated Confidence: **{c_val * 100:.1f}%** (Threshold: {confidence_thresh * 100:.0f}%)"
        )

        # Alphabet highlights (compact grid for all 26 letters A-Z)
        active_c = p_letter if p_letter else (r_letter if c_val >= 0.30 else None)
        pills = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(36px, 1fr)); gap: 4px;'>"
        for g in GESTURE_CLASSES:
            act = (g == active_c)
            col = "#00d4ff" if act else "#64748b"
            bg = "rgba(0, 212, 255, 0.25)" if act else "rgba(100, 116, 139, 0.08)"
            bord = f"2px solid {col}" if act else "1px solid rgba(100, 116, 139, 0.25)"
            pills += f"<div style='text-align: center; padding: 4px 2px; border: {bord}; border-radius: 6px; font-weight: bold; font-size: 0.85rem; color: {col}; background: {bg};'>{g}</div>"
        pills += "</div>"
        alphabet_slot.markdown(pills, unsafe_allow_html=True)

        sent_disp = curr_sent if curr_sent else "<span style='color: #475569;'>No text generated yet. Hold a gesture to auto-add...</span>"
        sentence_card_slot.markdown(
            f"""
            <div class="sentence-card">
                <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Constructed Sentence:</div>
                <div class="sentence-text">{sent_disp}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Initial render of telemetry
    render_telemetry_ui(
        st.session_state.last_predicted_letter,
        None,
        st.session_state.last_confidence,
        st.session_state.text_generator.get_text()
    )

    with col_cam:
        # Complete 26-Letter ASL Gesture Guide
        with st.expander("📖 Complete 26-Letter ASL Gesture Guide (A to Z)", expanded=False):
            st.markdown(
                """
                - **`A`**: Fist with thumb straight UP alongside index finger.
                - **`B`**: 4 fingers straight UP together, thumb tucked over palm.
                - **`C`**: Curved fingers and thumb forming a 'C' shape.
                - **`D`**: Index pointing straight UP, thumb touching middle/ring/pinky tips.
                - **`E`**: All fingers curled down tight, thumb tucked horizontally below finger pads.
                - **`F`**: Thumb & index tips touch ('OK' sign), middle/ring/pinky extended UP.
                - **`G`**: Index pointing horizontally left, thumb extended alongside.
                - **`H`**: Index & middle pointing horizontally left together.
                - **`I`**: Pinky extended straight UP, others curled into fist.
                - **`J`**: Pinky extended with curved wrist/tip.
                - **`K`**: Index UP, middle tilted forward, thumb between them.
                - **`L`**: Thumb and index extended forming an **"L"** (other 3 curled). ⭐ *Fastest to test!*
                - **`M`**: 3 fingers (index, middle, ring) draped over thumb.
                - **`N`**: 2 fingers (index, middle) draped over thumb.
                - **`O`**: All fingertips touching thumb to form an **"O"** circle.
                - **`P`**: Like 'K' tilted downwards.
                - **`Q`**: Index and thumb pointing down like tweezers.
                - **`R`**: Index & middle crossed over each other pointing UP.
                - **`S`**: Closed fist with thumb crossed in FRONT of curled fingers.
                - **`T`**: Thumb tucked between index and middle fingers.
                - **`U`**: Index and middle fingers extended straight UP together (touching).
                - **`V`**: Peace sign / **'V'** shape (Index & Middle fingers spread UP). ⭐ *Fastest to test!*
                - **`W`**: **3 fingers UP** (Index, Middle, Ring spread apart). ⭐ *Fastest to test!*
                - **`X`**: Index finger bent into a hook shape.
                - **`Y`**: Thumb and pinky extended wide (**"Hang loose" / Shaka**). ⭐ *Fastest to test!*
                - **`Z`**: Index finger pointing up/forward, others curled.
                """
            )

        # ----------------------------------------------------
        # Mode A: Continuous Webcam Stream (OpenCV cv2)
        # ----------------------------------------------------
        if feed_mode == "Continuous Webcam Stream":
            cam_toggle = st.toggle("Activate Live Webcam Stream", value=st.session_state.camera_running)
            st.session_state.camera_running = cam_toggle

            if st.session_state.camera_running:
                cap = cv2.VideoCapture(CAMERA_INDEX)
                # Optimize capture buffer size to prevent frame queuing lag
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

                if not cap.isOpened():
                    status_placeholder.error(
                        "Could not connect to webcam. Please verify camera permissions or switch to 'Browser Snapshot' mode above."
                    )
                    st.session_state.camera_running = False
                else:
                    status_placeholder.markdown(
                        "<span class='status-badge status-active'>Live Feed Active (High-Speed Mode)</span>",
                        unsafe_allow_html=True
                    )

                    last_ui_update_time = 0.0
                    last_ui_pred = None
                    last_ui_sent = None

                    # Ultra-smooth high-performance stream loop
                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.warning("Failed to grab video frame.")
                            break

                        if mirror_mode:
                            frame = cv2.flip(frame, 1)

                        pred_label = None
                        raw_label = None
                        conf = 0.0

                        if detector:
                            landmarks_list, handedness = detector.process_frame(frame)
                            annotated_frame = detector.draw_landmarks(frame, landmarks_list)

                            if landmarks_list and classifier:
                                features = detector.extract_features(landmarks_list[0], include_z=USE_3D_LANDMARKS)
                                pred_label, conf, prob_dict, raw_label = safe_predict(classifier, features)

                                st.session_state.last_predicted_letter = pred_label
                                st.session_state.last_confidence = conf

                                if auto_confirm and pred_label:
                                    confirmed = st.session_state.text_generator.update_prediction(pred_label)
                                    if confirmed:
                                        st.session_state.text_generator.add_letter(confirmed)
                            else:
                                st.session_state.last_predicted_letter = None
                                st.session_state.last_confidence = 0.0
                                st.session_state.text_generator.update_prediction(None)
                        else:
                            annotated_frame = frame

                        current_sent = st.session_state.text_generator.get_text()

                        # In-Frame HUD: Zero-latency real-time overlay
                        h_f, w_f, _ = annotated_frame.shape

                        # Semi-transparent top HUD banner
                        overlay_top = annotated_frame.copy()
                        cv2.rectangle(overlay_top, (0, 0), (w_f, 55), (10, 14, 23), -1)
                        cv2.rectangle(overlay_top, (0, h_f - 40), (w_f, h_f), (10, 14, 23), -1)
                        cv2.addWeighted(overlay_top, 0.70, annotated_frame, 0.30, 0, annotated_frame)

                        if pred_label:
                            cv2.putText(annotated_frame, f"PREDICTION: {pred_label}", (15, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(annotated_frame, f"CONF: {conf*100:.0f}%", (w_f - 170, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 128), 2, cv2.LINE_AA)
                        elif raw_label and conf >= 0.30:
                            cv2.putText(annotated_frame, f"DETECTING: {raw_label} ({conf*100:.0f}%)", (15, 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
                            cv2.putText(annotated_frame, "HOLD STEADY", (w_f - 170, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(annotated_frame, "SHOW ASL SIGN (L, V, W, Y, B...)", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 2, cv2.LINE_AA)

                        # Bottom Sentence Banner
                        disp_bottom = f">> {current_sent}" if current_sent else ">> Sentence: [Empty - hold a sign to add]"
                        cv2.putText(annotated_frame, disp_bottom, (15, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 212, 255), 2, cv2.LINE_AA)

                        # Fast JPEG compression to eliminate Streamlit WebSocket frame lag
                        _, jpeg_buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        video_placeholder.image(jpeg_buffer.tobytes(), channels="BGR", use_container_width=True)

                        # Intelligent UI Throttling: Only push heavy markdown updates when state changes or at 2.5Hz
                        now = time.time()
                        state_changed = (pred_label != last_ui_pred) or (current_sent != last_ui_sent)
                        time_elapsed = (now - last_ui_update_time) >= 0.40

                        if state_changed or time_elapsed:
                            render_telemetry_ui(pred_label, raw_label, conf, current_sent)
                            last_ui_update_time = now
                            last_ui_pred = pred_label
                            last_ui_sent = current_sent

                        time.sleep(0.005)  # Micro-yield for ultra-fast frame rates

                    cap.release()
                    status_placeholder.markdown(
                        "<span class='status-badge status-idle'>Stream Stopped</span>",
                        unsafe_allow_html=True
                    )
            else:
                status_placeholder.info("Toggle 'Activate Live Webcam Stream' above to start real-time recognition.")
                placeholder_img = np.zeros((360, 540, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder_img,
                    "Camera Stream Offline",
                    (130, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (100, 116, 139),
                    2
                )
                cv2.putText(
                    placeholder_img,
                    "Toggle 'Activate Live Webcam Stream' to begin",
                    (70, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (71, 85, 105),
                    1
                )
                video_placeholder.image(placeholder_img, channels="RGB", use_container_width=True)

        # ----------------------------------------------------
        # Mode B: Browser Snapshot (st.camera_input)
        # ----------------------------------------------------
        elif feed_mode == "Browser Snapshot (Photo)":
            st.info("Uses browser-native camera access directly.")
            camera_img = st.camera_input("Capture Hand Gesture")
            if camera_img is not None:
                pil_img = Image.open(camera_img)
                frame_rgb = np.array(pil_img)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if detector:
                    landmarks_list, _ = detector.process_frame(frame_bgr)
                    annotated_bgr = detector.draw_landmarks(frame_bgr, landmarks_list)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    if landmarks_list and classifier:
                        features = detector.extract_features(landmarks_list[0], include_z=USE_3D_LANDMARKS)
                        pred_label, conf, _, raw_label = safe_predict(classifier, features)
                        st.session_state.last_predicted_letter = pred_label
                        st.session_state.last_confidence = conf
                        render_telemetry_ui(pred_label, raw_label, conf, st.session_state.text_generator.get_text())
                        if pred_label:
                            status_placeholder.success(f"Recognized: {pred_label} ({conf*100:.1f}%)")
                    else:
                        st.session_state.last_predicted_letter = None
                        st.session_state.last_confidence = 0.0
                        render_telemetry_ui(None, None, 0.0, st.session_state.text_generator.get_text())
                        status_placeholder.warning("No hand detected in captured photo.")

        # ----------------------------------------------------
        # Mode C: Upload Test Image
        # ----------------------------------------------------
        elif feed_mode == "Upload Test Image":
            uploaded_file = st.file_uploader("Upload Hand Gesture Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                frame_rgb = np.array(pil_img)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if detector:
                    landmarks_list, _ = detector.process_frame(frame_bgr)
                    annotated_bgr = detector.draw_landmarks(frame_bgr, landmarks_list)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    if landmarks_list and classifier:
                        features = detector.extract_features(landmarks_list[0], include_z=USE_3D_LANDMARKS)
                        pred_label, conf, _, raw_label = safe_predict(classifier, features)
                        st.session_state.last_predicted_letter = pred_label
                        st.session_state.last_confidence = conf
                        render_telemetry_ui(pred_label, raw_label, conf, st.session_state.text_generator.get_text())
                        if pred_label:
                            status_placeholder.success(f"Recognized Gesture: '{pred_label}' ({conf*100:.1f}%)")
                    else:
                        st.session_state.last_predicted_letter = None
                        st.session_state.last_confidence = 0.0
                        render_telemetry_ui(None, None, 0.0, st.session_state.text_generator.get_text())
                        status_placeholder.warning("No hand landmarks identified in image.")

    # Text & Speech Action Controls
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        if st.button("➕ Add Letter", use_container_width=True):
            if pred_letter:
                st.session_state.text_generator.add_letter(pred_letter)
                st.rerun()
            else:
                st.toast("No gesture detected to add!", icon="⚠️")

    with c2:
        if st.button("␣ Space", use_container_width=True):
            st.session_state.text_generator.add_space()
            st.rerun()

    with c3:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state.text_generator.delete_last_character()
            st.rerun()

    with c4:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.text_generator.clear()
            st.rerun()

    with c5:
        if st.button("🔊 Speak Text", type="primary", use_container_width=True):
            if current_sentence.strip():
                started = st.session_state.speech_generator.speak(current_sentence)
                if started:
                    st.toast(f"Synthesizing: '{current_sentence}'", icon="🎙️")
                # Also inject browser-native speech synthesis fallback
                st.markdown(
                    f"""
                    <script>
                        if ('speechSynthesis' in window) {{
                            const utterance = new SpeechSynthesisUtterance({json.dumps(current_sentence)});
                            utterance.rate = 1.0;
                            window.speechSynthesis.speak(utterance);
                        }}
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.toast("Sentence buffer is empty. Construct text before speaking.", icon="⚠️")

    with c6:
        if st.button("⏹️ Stop Audio", use_container_width=True):
            st.session_state.speech_generator.stop()
            st.markdown(
                """
                <script>
                    if ('speechSynthesis' in window) {
                        window.speechSynthesis.cancel();
                    }
                </script>
                """,
                unsafe_allow_html=True
            )
            st.toast("Speech stopped.", icon="⏹️")


# ==============================================================================
# TAB 2: MODEL ANALYTICS & EVALUATION
# ==============================================================================
elif navigation == "Model Analytics & Evaluation":
    st.subheader("Model Performance & Evaluation Metrics")
    st.write("Rigorous quantitative assessment of the machine learning classifier evaluated on held-out test data.")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">{metrics['accuracy']*100:.2f}%</div></div>""",
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Precision (Weighted)</div><div class="metric-value">{metrics['precision']*100:.2f}%</div></div>""",
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Recall (Weighted)</div><div class="metric-value">{metrics['recall']*100:.2f}%</div></div>""",
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">F1-Score (Weighted)</div><div class="metric-value">{metrics['f1_score']*100:.2f}%</div></div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        col_cm, col_rep = st.columns([1.2, 1.0], gap="large")

        with col_cm:
            st.subheader("Confusion Matrix")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
            else:
                st.info("Run scripts/evaluate_model.py to render confusion matrix.")

        with col_rep:
            st.subheader("Per-Class Classification Report")
            report_data = metrics.get("classification_report", {})
            class_rows = []
            for cls_key in metrics.get("classes", []):
                if cls_key in report_data:
                    c_info = report_data[cls_key]
                    class_rows.append({
                        "Gesture Class": cls_key,
                        "Precision": f"{c_info['precision']*100:.1f}%",
                        "Recall": f"{c_info['recall']*100:.1f}%",
                        "F1-Score": f"{c_info['f1-score']*100:.1f}%",
                        "Support": c_info["support"]
                    })
            if class_rows:
                st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

            st.caption(
                f"Evaluation protocol: 80/20 Stratified Split ({metrics['train_samples']} Train, {metrics['test_samples']} Test samples, {metrics['n_features']} features)."
            )
    else:
        st.warning("No evaluation metrics found. Please train the model using `scripts/train_model.py`.")


# ==============================================================================
# TAB 3: DATASET & TRAINING HUB
# ==============================================================================
elif navigation == "Dataset & Training Hub":
    st.subheader("Sign Language Dataset & Retraining Management")

    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        st.write(f"Dataset path: `{DATASET_PATH}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Labeled Samples", f"{len(df):,}")
        with c2:
            st.metric("Number of Gesture Classes", f"{df['label'].nunique()}")
        with c3:
            st.metric("Feature Dimensions", f"{df.shape[1] - 1} coordinates")

        st.markdown("---")
        st.subheader("Class Distribution")
        class_dist = df["label"].value_counts().reset_index()
        class_dist.columns = ["Gesture Class", "Sample Count"]
        st.bar_chart(class_dist.set_index("Gesture Class"))

        st.markdown("---")
        st.subheader("Explore Raw Landmark Dataset")
        st.dataframe(df.head(25), use_container_width=True)

        st.markdown("---")
        st.subheader("Retrain Model from UI")
        st.write("Retrain the Random Forest classifier using the updated dataset above:")
        if st.button("Trigger Retraining Pipeline", type="primary"):
            with st.spinner("Retraining Random Forest model on dataset..."):
                from scripts.train_model import train_and_evaluate
                from scripts.evaluate_model import evaluate_and_plot
                success = train_and_evaluate(classifier_type="rf")
                if success:
                    evaluate_and_plot()
                    st.cache_resource.clear()
                    st.success("Model successfully retrained and cached metrics updated!")
                    st.rerun()
                else:
                    st.error("Retraining failed. Check logs.")
    else:
        st.error(f"Dataset not found at {DATASET_PATH}.")


# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE & PIPELINE
# ==============================================================================
elif navigation == "System Architecture & Pipeline":
    st.subheader("System Architecture & Engineering Methodology")

    st.markdown(
        """
        ### 1. High-Level End-to-End Architecture
        The system follows an asynchronous computer vision and machine learning pipeline:
        """
    )

    st.code(
        """
Webcam Feed (640x480 @ 30 FPS)
   │
   ▼
MediaPipe Hands (Tasks API / 21 3D Anatomical Landmarks)
   │
   ▼
Coordinate Invariance Transformation
   ├─ Translation Invariance: P'_i = P_i - P_wrist
   └─ Scale Invariance:       P''_i = P'_i / max(||P'_i||)
   │
   ▼
Scikit-Learn Classifier (Random Forest / SVM)
   │
   ▼
Confidence Threshold Filter (Threshold >= 75%)
   │
   ▼
Temporal Prediction Stabilization Buffer (Moving-window Debounce)
   │
   ▼
Sentence Constructor (Word aggregation, Space, Backspace)
   │
   ▼
Voice Synthesizer (pyttsx3 SAPI5 & Web Speech API)
        """,
        language="text"
    )

    st.markdown(
        """
        ### 2. Landmark Invariance Mathematical Formulation
        To ensure the model is robust against variations in hand position, hand distance, and user anatomy:
        1. **Translation Invariance**: The wrist coordinate (landmark 0) acts as the local origin:
           $$\\mathbf{P}'_i = \\mathbf{P}_i - \\mathbf{P}_0 \\quad \\forall i \\in [0, 20]$$
           This normalizes the wrist position to $(0, 0, 0)$ regardless of where the hand is located in the video frame.

        2. **Scale Invariance**: We divide by the maximum Euclidean distance between any joint and the wrist:
           $$d_{\\max} = \\max_{i} \\|\\mathbf{P}'_i\\|$$
           $$\\mathbf{P}''_i = \\frac{\\mathbf{P}'_i}{d_{\\max}}$$
           This guarantees that moving closer or farther from the camera does not alter the feature vector representation.

        ### 3. Biomechanical 21-Keypoint Topology
        Each hand is modeled as a kinematic tree of 21 keypoints spanning the wrist, CMC, MCP, PIP, DIP, and fingertip nodes.
        """
    )


# ==============================================================================
# TAB 5: FACULTY PRESENTATION GUIDE
# ==============================================================================
elif navigation == "Faculty Presentation Guide":
    st.subheader("Academic Project-Based Learning (PBL) Presentation Guide")

    st.markdown(
        """
        ### Executive Summary
        - **Project Title:** Real-Time Sign Language Recognition System Using Artificial Intelligence and Computer Vision
        - **Specialization:** Computer Science and Engineering (AI & Machine Learning)
        - **Core Value Proposition:** Real-time bridge enabling non-signers to comprehend American Sign Language (ASL) finger-spelling via ubiquitous laptop webcams without specialized data gloves or hardware sensors.

        ---

        ### Faculty Q&A Preparation Cheat-Sheet

        **Q1: Why use MediaPipe landmarks instead of raw CNN pixels?**
        - *Answer:* Raw pixel CNNs are highly sensitive to background clutter, lighting variations, skin tone differences, and camera angles. Extracting 21 spatial landmarks decouples visual appearance from geometric hand pose, drastically reducing model size, improving inference speed to real-time 30+ FPS on CPU, and guaranteeing translation and scale invariance.

        **Q2: What is your machine learning model and why?**
        - *Answer:* A Random Forest classifier (120 estimators). On normalized geometric vectors of size 63, Random Forest provides non-linear boundary separation, avoids overfitting through bootstrap aggregation, outputs calibrated class probabilities for confidence thresholding, and executes inference in under 2 milliseconds on CPU.

        **Q3: How do you handle continuous video fluctuations?**
        - *Answer:* We implemented a temporal stabilization ring-buffer. Instead of emitting continuous characters every frame (which would spam identical letters 30 times a second), the debouncer requires $N$ consecutive identical predictions and enforces a cooldown before confirming character addition.

        **Q4: What are the current limitations and future scope?**
        - *Current Limitations:* Prototype focuses on 10 static ASL alphabet gestures. Dynamic letters (like J and Z) and continuous two-handed sentences with facial grammar are beyond static classifiers.
        - *Future Scope:* Integrating temporal LSTM/GRU/Transformer networks for continuous sign language sentences, two-handed conversational vocabularies, and mobile deployment via ONNX Runtime.
        """
    )
