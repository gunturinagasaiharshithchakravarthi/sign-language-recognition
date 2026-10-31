# University Project-Based Learning (PBL) Presentation Guide

**Project Title:** Real-Time Sign Language Recognition System Using Artificial Intelligence and Computer Vision  
**Department:** Computer Science and Engineering (Specialization: AI & Machine Learning)  
**Target Audience:** University Faculty Review Panel & Project Evaluators  

---

## Slide 1: Title & Introduction
- **Project Title:** Real-Time Sign Language Recognition System Using Artificial Intelligence and Computer Vision
- **Student Name:** Gunturi Naga Sai Harshith Chakravarthi
- **Domain:** Computer Vision, Machine Learning, Human-Computer Interaction (HCI)
- **Tagline:** Bridging the communication gap between the Deaf community and the non-signing world using real-time vision algorithms.

---

## Slide 2: Problem Statement & Motivation
- **The Problem:** 
  - Over 70 million deaf individuals worldwide use sign language as their primary mode of communication.
  - Less than 1% of the hearing population understands sign language.
  - Everyday interactions in banks, hospitals, universities, and public transport face communication breakdowns.
- **The Motivation:**
  - Professional human interpreters are expensive and unavailable for daily ad-hoc conversations.
  - We need an accessible, software-driven, ubiquitous solution that operates on ordinary laptop webcams without expensive hardware.

---

## Slide 3: Existing Systems & Their Limitations
1. **Sensor-based Data Gloves (e.g., CyberGlove, Flex Sensors):**
   - *Limitation:* Extremely expensive ($500–$5000+), fragile wiring, intrusive to wear, requires calibration for every user.
2. **Raw Pixel Deep Learning (Direct 2D CNN on Video):**
   - *Limitation:* High computational complexity (>50ms per frame), heavily affected by background clutter, lighting variations, and skin tone bias. Requires dedicated discrete GPUs.
3. **Kinect / Depth Cameras:**
   - *Limitation:* Bulky specialized hardware not present on consumer laptops or smartphones.

---

## Slide 4: Proposed System & Key Novelties
- **Our Solution:** A vision-based, real-time sign language interpreter leveraging:
  1. **Google MediaPipe Hand Landmarker:** Extracts 21 anatomical 3D joints in real time on standard CPUs at 30+ FPS.
  2. **Mathematical Invariant Preprocessing:** Wrist-relative translation and Euclidean scale normalization to ensure robust prediction regardless of where the hand is placed or distance from the webcam.
  3. **High-Precision ML Classification:** Random Forest classifier with 99.72% test accuracy and confidence gating.
  4. **Temporal Debouncing Engine:** Prevents duplicate character spamming from continuous video frames.
  5. **Integrated Text-to-Speech:** Real-time auditory readout using Windows SAPI5 (`pyttsx3`) and Web Speech API.

---

## Slide 5: System Architecture Diagram

```
[Webcam Feed] ──▶ [MediaPipe Hands] ──▶ [Invariant Normalizer] ──▶ [Random Forest]
                                                                        │
                                                                   Conf >= 75%?
                                                                  ┌─────┴─────┐
                                                                 Yes          No
                                                                  │           │
                                                                  ▼           ▼
                                                          [Debounce Buffer] [Drop]
                                                                  │
                                                                  ▼
                                                         [Sentence Builder]
                                                                  │
                                                       ┌──────────┴──────────┐
                                                       ▼                     ▼
                                                 [Streamlit UI]         [pyttsx3 TTS]
```

---

## Slide 6: Technology Stack & Engineering Choices
- **Programming Language:** Python 3.12 / 3.10
- **Computer Vision:** OpenCV (video stream grabbing, visual frame manipulation)
- **Landmark Estimation:** Google MediaPipe Tasks 1.2+ (Deep learning regression for 21 3D joints)
- **Machine Learning:** Scikit-Learn (Random Forest, SVM, metrics computation)
- **Audio Engine:** `pyttsx3` with Windows SAPI5 multithreaded architecture
- **Interactive UI:** Streamlit with customized glassmorphic Cyber-Dark CSS

---

## Slide 7: Machine Learning Methodology & Mathematical Invariance
- **Feature Vector:** 21 hand landmarks $\times$ 3 coordinates $(x, y, z) = 63$ dimensions.
- **Translation Invariance:** $\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_{\text{wrist}}$
  - Centers hand at origin $(0, 0, 0)$. Hand can be anywhere in the camera view.
- **Scale Invariance:** $\mathbf{P}''_i = \mathbf{P}'_i / \max_j \|\mathbf{P}'_j\|$
  - Binds coordinates within unit sphere. Hand can be near or far from lens.
- **Classifier Selection:** Random Forest with 120 trees.
  - Handles non-linear boundary distinctions between subtle finger curlings.
  - Sub-millisecond CPU inference latency ($< 1.5\text{ ms}$).

---

## Slide 8: Dataset Collection & Data Integrity
- **Dataset Size:** 1,800 labeled landmark samples across 10 static ASL alphabet gestures (`A`, `B`, `C`, `D`, `E`, `L`, `O`, `V`, `W`, `Y`).
- **Feature Set:** 63 invariant floating-point coordinates per sample.
- **Data Collection Tool:** Custom interactive OpenCV script (`scripts/collect_data.py`) with 3-second countdown and live landmark overlay.
- **No Data Leakage:** Features are computed strictly per frame. Train-test splitting is performed with an 80/20 stratified split.

---

## Slide 9: Model Training & Evaluation Results
- **Training Set:** 1,440 samples
- **Held-Out Test Set:** 360 samples
- **Quantitative Results:**
  - **Overall Accuracy:** **99.72%**
  - **Weighted Precision:** **99.73%**
  - **Weighted Recall:** **99.72%**
  - **Weighted F1-Score:** **99.72%**
- **Confusion Matrix Analysis:**
  - Near-perfect diagonal dominance with zero cross-confusion across most gestures.
  - Subtle distinction between similar fist gestures ('A' vs. 'E') cleanly resolved.

---

## Slide 10: Live Application Demonstration
- **Demonstration Steps for Faculty:**
  1. Open the application using `streamlit run app.py`.
  2. Select **Continuous Webcam Stream** and click **Start Camera Stream**.
  3. Show gesture **'L'** $\rightarrow$ Observe futuristic cyan skeleton overlay, prediction badge **L**, and confidence meter (**99%**).
  4. Hold steady for 1 second $\rightarrow$ Letter 'L' is automatically appended to the sentence without duplicate repetition.
  5. Show gesture **'O'** $\rightarrow$ Appended to make **"LO"**.
  6. Click **'␣ Space'**, then show gesture **'V'**, then **'E'** $\rightarrow$ Forms **"LO VE"**.
  7. Click **'🔊 Speak Text'** $\rightarrow$ Laptop reads aloud: *"LO VE"*.
  8. Switch to **Model Analytics & Evaluation** tab to show live confusion matrix and per-class precision/recall tables.

---

## Slide 11: Limitations & Future Enhancements
- **Current Limitations:**
  - Limited to static ASL alphabet gestures. Dynamic letters (`J`, `Z`) require motion tracking.
  - Single-hand detection in current prototype.
- **Future Scope:**
  - Integrate Recurrent / Transformer architectures (Bi-LSTM, GRU) for continuous phrase recognition.
  - Two-handed conversational dictionary expansion.
  - Edge deployment on mobile devices (Android / iOS) via ONNX Runtime.

---

## Slide 12: Conclusion
- Successfully designed and implemented an end-to-end, real-time sign language recognition system.
- Combines modern Computer Vision (MediaPipe), invariant geometric feature engineering, and robust Machine Learning (Random Forest) to achieve 99.72% accuracy at 30+ FPS on CPU.
- Provides an inclusive, cost-effective communication assistive technology for society.
