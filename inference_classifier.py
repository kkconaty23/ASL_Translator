import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
from time import time
import os

# ========== CONFIG ==========
MODEL_PATH = 'model_rf_336.p'   # or 'model.p'
CAMERA_INDEX = 0
WIN_NAME = 'ASL Inference'
CONFIDENCE_BAR = True

# MANUAL LETTER MAPPING - Update this to match your folder organization
# Map class indices to letters (adjust based on your training data order)
CLASS_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Alternative: Auto-detect from folder structure (set DATA_PATH to your training data folder)
DATA_PATH = None  # e.g., '/path/to/your/asl_dataset' or None to use manual mapping

# Letters that require motion
MOTION_ONLY_CLASSES = {'J', 'Z'}   # must match your class names
MOTION_THRESHOLD = 0.002           # tweak after quick calibration

# For 84-D models (static): smooth predictions across last N frames
SMOOTH_K = 7

# For 336-D models (motion): frames per window (e.g., ~1 sec @ 30 fps)
SEQ_WINDOW = 30
MIN_SEQ_FOR_PRED = 8

# Enhanced display settings
MIN_CONFIDENCE = 0.60      # require this probability or higher
STABLE_N = 6               # frames in a row the same top class must appear
DISPLAY_MS_AFTER_LOCK = 700
CONFIDENCE_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]  # Red to Green
# ===========================

def get_letter_mapping(data_path=None):
    """Get letter mapping from folder structure or use manual mapping."""
    if data_path and os.path.exists(data_path):
        # Auto-detect from folder structure
        folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        folders.sort()  # Ensure consistent ordering
        letter_mapping = {i: folder for i, folder in enumerate(folders)}
        print(f"[INFO] Auto-detected letters from {data_path}: {letter_mapping}")
        return letter_mapping
    else:
        # Use manual mapping
        print(f"[INFO] Using manual letter mapping: {CLASS_TO_LETTER}")
        return CLASS_TO_LETTER

def inspect_model_classes(model_path):
    """Helper function to inspect what classes your model actually has."""
    try:
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        
        model = obj.get('model', obj)
        model_classes = getattr(model, 'classes_', None)
        meta_classes = obj.get('classes', None)
        
        print("=== MODEL CLASS INSPECTION ===")
        print(f"model.classes_: {model_classes}")
        print(f"meta classes: {meta_classes}")
        print(f"classes_ type: {type(model_classes[0]) if model_classes is not None and len(model_classes) > 0 else 'None'}")
        
        if model_classes is not None:
            print(f"Number of classes: {len(model_classes)}")
            print(f"First few classes: {model_classes[:10] if len(model_classes) > 10 else model_classes}")
        
        return model_classes, meta_classes
    
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None, None

def load_model_safely(model_path):
    """Load model with error handling and validation."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        
        model = obj.get('model', obj)
        model_classes = list(getattr(model, 'classes_', []))
        meta_classes = obj.get('classes', None)
        n_features = getattr(model, 'n_features_in_', None)
        
        # Get proper letter mapping
        letter_mapping = get_letter_mapping(DATA_PATH)
        
        # Create label names using the letter mapping
        if model_classes:
            # If model has classes, try to map them
            if all(isinstance(c, (int, np.integer)) for c in model_classes):
                # Classes are numeric indices - map to letters
                label_names = [letter_mapping.get(int(c), str(c)) for c in model_classes]
            else:
                # Classes are already strings (letters)
                label_names = [str(c) for c in model_classes]
        elif meta_classes:
            # Use meta classes if available
            if all(isinstance(c, (int, np.integer)) for c in meta_classes):
                label_names = [letter_mapping.get(int(c), str(c)) for c in meta_classes]
            else:
                label_names = [str(c) for c in meta_classes]
        else:
            # Fallback: assume sequential indices 0, 1, 2, ...
            max_classes = len(letter_mapping)
            label_names = [letter_mapping.get(i, str(i)) for i in range(max_classes)]
            print(f"[WARNING] No model classes found, assuming {max_classes} classes")
        
        print(f"[INFO] Model loaded successfully from {model_path}")
        print(f"[INFO] n_features: {n_features}")
        print(f"[INFO] model.classes_ (proba order): {model_classes}")
        print(f"[INFO] label_names (letters for UI): {label_names}")
        
        return model, model_classes, label_names, n_features, letter_mapping
    
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def initialize_mediapipe():
    """Initialize MediaPipe hands detection."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,  # Slightly higher for better accuracy
        min_tracking_confidence=0.5
    )
    
    return mp_hands, mp_draw, mp_styles, hands

def draw_enhanced_label(frame, text, confidence=None, bottom_center=True):
    """Draw an enhanced label with confidence-based styling."""
    if text is None:
        return
    
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.8
    thickness = 6
    
    # Color based on confidence
    if confidence is not None:
        if confidence >= 0.9:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.75:
            color = (0, 255, 255)  # Yellow
        elif confidence >= 0.6:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
    else:
        color = (255, 255, 255)  # White
    
    # Add confidence percentage to text if available
    display_text = f"{text}" + (f" ({confidence*100:.0f}%)" if confidence else "")
    
    (tw, th), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    pad = 20
    
    if bottom_center:
        cx, cy = w // 2, int(h * 0.85)
        x1 = cx - tw // 2 - pad
        y1 = cy - th - pad
        x2 = cx + tw // 2 + pad
        y2 = cy + pad
        tx = cx - tw // 2
        ty = cy
    else:
        x1, y1 = 20, 20
        x2, y2 = x1 + tw + 2 * pad, y1 + th + 2 * pad
        tx, ty = x1 + pad, y1 + th + pad
    
    # Draw background with slight transparency effect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw text
    cv2.putText(frame, display_text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

def order_hands(results):
    """Order hands consistently (left-to-right in image)."""
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
    
    def sort_key(item):
        label, hl, conf = item
        mean_x = sum(lm.x for lm in hl.landmark) / 21.0
        return mean_x  # Sort by x-position
    
    hands_list.sort(key=sort_key)
    return hands_list[:2]

def feat84_from_results(results):
    """Extract 84-dimensional feature vector from MediaPipe results."""
    hlists = order_hands(results)
    if not hlists:
        return None, 0
    
    # Get hand confidence (average if multiple hands)
    hand_confidence = np.mean([conf for _, _, conf in hlists])
    
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
    """Convert sequence of 84-D features to 336-D motion features."""
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

def window_motion_level(seq_Tx84: np.ndarray) -> float:
    """Calculate motion level in the sequence."""
    if seq_Tx84.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(seq_Tx84, axis=0))
    return float(diffs.mean())

def draw_enhanced_info(frame, proba, names, motion=None, hand_conf=None, k=3, x=20, y=70, dy=30):
    """Draw enhanced information panel."""
    if proba is None:
        cv2.putText(frame, "No hands detected", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return
    
    # Top-k predictions
    idxs = np.argsort(proba)[::-1][:k]
    for i, idx in enumerate(idxs):
        label = names[idx] if idx < len(names) else str(idx)
        p = proba[idx]
        
        # Color coding based on probability
        if p >= 0.8:
            color = (0, 255, 0)  # Green
        elif p >= 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 255, 255)  # White
        
        text = f"{i+1}. {label}: {p*100:.1f}%"
        cv2.putText(frame, text, (x, y + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Additional info
    info_y = y + k*dy + 10
    if motion is not None:
        motion_text = f"Motion: {motion:.4f}"
        cv2.putText(frame, motion_text, (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        info_y += 25
    
    if hand_conf is not None:
        conf_text = f"Hand Conf: {hand_conf:.2f}"
        cv2.putText(frame, conf_text, (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def main():
    """Main inference loop."""
    try:
        # Load model
        model, model_classes, label_names, n_features, letter_mapping = load_model_safely(MODEL_PATH)
        
        # Initialize MediaPipe
        mp_hands, mp_draw, mp_styles, hands = initialize_mediapipe()
        
        # Initialize camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
        
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        
        # Buffers for smoothing
        feat84_buffer = deque(maxlen=max(SEQ_WINDOW, SMOOTH_K))
        proba_buffer = deque(maxlen=8)
        
        # Stability tracking
        stable_idx = None
        stable_run = 0
        lock_text = None
        lock_until = 0.0
        lock_confidence = 0.0
        
        prev_time = time()
        frame_count = 0
        
        print("[INFO] Starting inference loop. Press 'q' to quit, 'r' to reset buffers.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
            
            # Extract features
            feat84, hand_confidence = feat84_from_results(results)
            pred_proba = None
            motion_level = None
            
            if feat84 is not None:
                feat84_buffer.append(feat84)
                
                if n_features == 84:
                    # Static model
                    if len(feat84_buffer) >= SMOOTH_K:
                        X = np.mean(np.stack(list(feat84_buffer)[-SMOOTH_K:]), axis=0)
                        X = X.reshape(1, -1)
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X)[0]
                
                elif n_features == 336:
                    # Motion model
                    if len(feat84_buffer) >= MIN_SEQ_FOR_PRED:
                        seq = np.stack(list(feat84_buffer)[-SEQ_WINDOW:], axis=0)
                        motion_level = window_motion_level(seq)
                        
                        if motion_level < MOTION_THRESHOLD:
                            # Static fallback
                            mean = seq.mean(axis=0).astype(np.float32)
                            zeros = np.zeros_like(mean)
                            X336 = np.concatenate([mean, zeros, zeros, zeros], axis=0)
                        else:
                            X336 = to_336_from_seq(seq)
                        
                        X336 = X336.reshape(1, -1)
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X336)[0]
                        
                        # Apply motion gating for letter-based classes
                        if pred_proba is not None and motion_level < MOTION_THRESHOLD:
                            for i, letter in enumerate(label_names):
                                if letter in MOTION_ONLY_CLASSES and i < len(pred_proba):
                                    pred_proba[i] = 0.0
                            
                            # Renormalize
                            total = pred_proba.sum()
                            if total > 0:
                                pred_proba = pred_proba / total
            
            else:
                feat84_buffer.clear()
            
            # Smooth probabilities
            if pred_proba is not None:
                proba_buffer.append(pred_proba)
            
            proba_display = None
            if proba_buffer:
                proba_display = np.mean(np.stack(proba_buffer, axis=0), axis=0)
            
            # Stability logic for main display
            current_time = time()
            if proba_display is not None and len(proba_display) > 0:
                top_idx = int(np.argmax(proba_display))
                top_prob = float(np.max(proba_display))
                
                if stable_idx == top_idx:
                    stable_run += 1
                else:
                    stable_idx = top_idx
                    stable_run = 1
                
                if stable_run >= STABLE_N and top_prob >= MIN_CONFIDENCE:
                    lock_text = label_names[top_idx] if top_idx < len(label_names) else str(top_idx)
                    lock_confidence = top_prob
                    lock_until = current_time + (DISPLAY_MS_AFTER_LOCK / 1000.0)
            
            # Draw main prediction
            if current_time < lock_until and lock_text is not None:
                draw_enhanced_label(frame, lock_text, lock_confidence, bottom_center=True)
            
            # Draw info panel
            if CONFIDENCE_BAR:
                draw_enhanced_info(frame, proba_display, label_names, motion_level, hand_confidence)
            
            # FPS and system info
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            
            status_text = f"FPS: {fps:.1f} | Features: {n_features} | Buffer: {len(feat84_buffer)}"
            cv2.putText(frame, status_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            cv2.imshow(WIN_NAME, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                feat84_buffer.clear()
                proba_buffer.clear()
                print("[INFO] Buffers reset")
            elif key == ord(' '):  # Space to pause/unpause
                cv2.waitKey(0)
    
    except Exception as e:
        print(f"[ERROR] {e}")
    
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        if 'hands' in locals():
            hands.close()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    main()