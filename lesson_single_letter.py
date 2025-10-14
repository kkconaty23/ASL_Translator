# lesson_single_letter.py
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
from time import time
import os

# ========== CONFIG ==========
MODEL_PATH = 'model_rf_336.p'   # your trained model path
CAMERA_INDEX = 0
WIN_NAME = 'ASL Lesson — Target: A'

TARGET_LETTER = 'C'            # the lesson asks only for "A"
CORRECT_THRESHOLD = 0.80       # min avg probability for A
CORRECT_STABLE_FRAMES = 6      # frames in a row above threshold
COOLDOWN_AFTER_CORRECT = 1.0   # seconds to display "Correct!"

# These let the script work with either 84-D (static) or 336-D (motion) models
SMOOTH_K = 7                   # smoothing window for 84-D
SEQ_WINDOW = 30                # frames used for 336-D features
MIN_SEQ_FOR_PRED = 8
MOTION_ONLY_CLASSES = {'J', 'Z'}
MOTION_THRESHOLD = 0.002

# Manual fallback label mapping if the model only has numeric classes
CLASS_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}
# ===========================


# --------- Helpers (self-contained) ---------
def get_letter_mapping():
    print(f"[INFO] Using manual letter mapping: {CLASS_TO_LETTER}")
    return CLASS_TO_LETTER

def load_model_safely(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)

    # Support dicts like {'model': <clf>, 'classes': [...]} or raw model pickles
    model = obj.get('model', obj)
    model_classes = list(getattr(model, 'classes_', []))
    meta_classes = obj.get('classes', None)
    n_features = getattr(model, 'n_features_in_', None)

    letter_mapping = get_letter_mapping()

    if model_classes:
        if all(isinstance(c, (int, np.integer)) for c in model_classes):
            label_names = [letter_mapping.get(int(c), str(c)) for c in model_classes]
        else:
            label_names = [str(c) for c in model_classes]
    elif meta_classes:
        if all(isinstance(c, (int, np.integer)) for c in meta_classes):
            label_names = [letter_mapping.get(int(c), str(c)) for c in meta_classes]
        else:
            label_names = [str(c) for c in meta_classes]
    else:
        max_classes = len(letter_mapping)
        label_names = [letter_mapping.get(i, str(i)) for i in range(max_classes)]
        print(f"[WARNING] No model classes found, assuming {max_classes} classes")

    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] n_features: {n_features}")
    print(f"[INFO] classes (proba order): {model_classes if model_classes else meta_classes}")
    print(f"[INFO] label_names: {label_names}")
    return model, label_names, n_features

def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_draw, mp_styles, hands

def order_hands(results):
    if not getattr(results, 'multi_hand_landmarks', None):
        return []
    hands_list = []
    if getattr(results, "multi_handedness", None):
        for handed, hl in zip(results.multi_handedness, results.multi_hand_landmarks):
            label = handed.classification[0].label
            confidence = handed.classification[0].score
            hands_list.append((label, hl, confidence))
    else:
        for hl in results.multi_hand_landmarks:
            mean_x = sum(lm.x for lm in hl.landmark) / 21.0
            label = "Left" if mean_x < 0.5 else "Right"
            hands_list.append((label, hl, 1.0))
    hands_list.sort(key=lambda item: sum(lm.x for lm in item[1].landmark) / 21.0)
    return hands_list[:2]

def feat84_from_results(results):
    hlists = order_hands(results)
    if not hlists:
        return None, 0.0
    hand_confidence = float(np.mean([conf for _, _, conf in hlists]))
    xs = [lm.x for _, hl, _ in hlists for lm in hl.landmark]
    ys = [lm.y for _, hl, _ in hlists for lm in hl.landmark]
    min_x, min_y = min(xs), min(ys)
    feat = []
    for slot in range(2):
        if slot < len(hlists):
            _, hl, _ = hlists[slot]
            for lm in hl.landmark:
                feat.append(lm.x - min_x)
                feat.append(lm.y - min_y)
        else:
            feat.extend([0.0] * 42)
    return np.asarray(feat, dtype=np.float32), hand_confidence

def to_336_from_seq(seq_Tx84):
    T = seq_Tx84.shape[0]
    mean = seq_Tx84.mean(axis=0)
    std = seq_Tx84.std(axis=0)
    last_first = seq_Tx84[-1] - seq_Tx84[0] if T > 1 else np.zeros_like(mean)
    if T >= 2:
        diffs = np.diff(seq_Tx84, axis=0)
        mad = np.mean(np.abs(diffs), axis=0)
    else:
        mad = np.zeros_like(mean)
    return np.concatenate([mean, std, last_first, mad], axis=0).astype(np.float32)

def window_motion_level(seq_Tx84):
    if seq_Tx84.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(seq_Tx84, axis=0))
    return float(diffs.mean())

def get_target_index(label_names, target):
    try:
        return label_names.index(target)
    except ValueError:
        lower = [s.lower() for s in label_names]
        t = target.lower()
        return lower.index(t) if t in lower else None

def draw_banner(frame, text, ok=False):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.2
    thick = 6
    color = (0, 200, 0) if ok else (0, 0, 255)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = int(h * 0.15) + th
    pad = 20
    x1, y1, x2, y2 = x - pad, y - th - pad, x + tw + pad, y + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def draw_small_text(frame, text, x=20, y=30):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


# --------- Main (single-letter lesson) ---------
def main():
    model, label_names, n_features = load_model_safely(MODEL_PATH)
    target_idx = get_target_index(label_names, TARGET_LETTER)
    if target_idx is None:
        raise RuntimeError(f"Target letter '{TARGET_LETTER}' not found in model labels: {label_names}")

    mp_hands, mp_draw, mp_styles, hands = initialize_mediapipe()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    feat84_buffer = deque(maxlen=max(SEQ_WINDOW, SMOOTH_K))
    proba_buffer = deque(maxlen=8)

    lesson_correct = False
    lesson_correct_until = 0.0
    lesson_stable_count = 0

    prev_time = time()

    print("[INFO] Lesson started. Press 'q' to quit, 'r' to reset.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw landmarks for feedback
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            feat84, _ = feat84_from_results(results)
            pred_proba = None
            motion_level = None

            if feat84 is not None:
                feat84_buffer.append(feat84)

                if n_features == 84:
                    # 84-D static model
                    if len(feat84_buffer) >= SMOOTH_K:
                        X = np.mean(np.stack(list(feat84_buffer)[-SMOOTH_K:]), axis=0).reshape(1, -1)
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X)[0]

                elif n_features == 336:
                    # 336-D motion model
                    if len(feat84_buffer) >= MIN_SEQ_FOR_PRED:
                        seq = np.stack(list(feat84_buffer)[-SEQ_WINDOW:], axis=0)
                        motion_level = window_motion_level(seq)

                        if motion_level < MOTION_THRESHOLD:
                            mean = seq.mean(axis=0).astype(np.float32)
                            zeros = np.zeros_like(mean)
                            X336 = np.concatenate([mean, zeros, zeros, zeros], axis=0)
                        else:
                            X336 = to_336_from_seq(seq)

                        X336 = X336.reshape(1, -1)
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X336)[0]

                        # If hardly moving, zero-out motion-only letters (J, Z) to avoid false positives
                        if pred_proba is not None and motion_level < MOTION_THRESHOLD:
                            for i, letter in enumerate(label_names):
                                if i < len(pred_proba) and letter in MOTION_ONLY_CLASSES:
                                    pred_proba[i] = 0.0
                            s = pred_proba.sum()
                            if s > 0:
                                pred_proba = pred_proba / s

            else:
                feat84_buffer.clear()

            # Smooth probabilities for display/decision
            if pred_proba is not None:
                proba_buffer.append(pred_proba)

            proba_display = None
            if proba_buffer:
                proba_display = np.mean(np.stack(proba_buffer, axis=0), axis=0)

            # Lesson logic
            now = time()
            if proba_display is not None and len(proba_display) > target_idx:
                p = float(proba_display[target_idx])

                if lesson_correct and now < lesson_correct_until:
                    draw_banner(frame, f"{TARGET_LETTER} — Correct!", ok=True)
                else:
                    if lesson_correct and now >= lesson_correct_until:
                        lesson_correct = False
                        lesson_stable_count = 0

                    if p >= CORRECT_THRESHOLD:
                        lesson_stable_count += 1
                    else:
                        lesson_stable_count = 0

                    if lesson_stable_count >= CORRECT_STABLE_FRAMES:
                        lesson_correct = True
                        lesson_correct_until = now + COOLDOWN_AFTER_CORRECT
                        draw_banner(frame, f"{TARGET_LETTER} — Correct!", ok=True)
                    else:
                        draw_banner(frame, f"Show: {TARGET_LETTER}  ({int(round(p*100))}%)", ok=False)
            else:
                draw_banner(frame, f"Show: {TARGET_LETTER}", ok=False)

            # Status line (FPS etc.)
            fps = 1.0 / max(time() - prev_time, 1e-6)
            prev_time = time()
            small = f"FPS: {fps:.1f} | Features: {n_features} | Buffer: {len(feat84_buffer)}"
            if motion_level is not None:
                small += f" | Motion: {motion_level:.4f}"
            draw_small_text(frame, small)

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                feat84_buffer.clear()
                proba_buffer.clear()
                lesson_correct = False
                lesson_stable_count = 0
                lesson_correct_until = 0.0
                print("[INFO] Reset lesson state")

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    main()
