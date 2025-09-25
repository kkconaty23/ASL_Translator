import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt  # optional visualization

DATA_DIR = Path("./data")

# Classes you consider "dynamic" (have sequences folder structure)
DYNAMIC_LABELS = {"J", "Z"}  # edit if you use lowercase or different names

# Acceptable image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# static_image_mode=True is correct for single-frame processing
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# -------------------------- Utilities --------------------------
def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if is_image_file(p)])

def list_subdirs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_dir()])

def order_hands(results) -> List[Tuple[str, Any]]:
    """
    Returns up to 2 hands [(label, hand_landmarks), ...] ordered Left then Right
    (fallback by x-position if handedness missing).
    """
    hands_list = []
    if getattr(results, "multi_hand_landmarks", None) is None:
        return hands_list

    if getattr(results, "multi_handedness", None):
        for handed, hl in zip(results.multi_handedness, results.multi_hand_landmarks):
            label = handed.classification[0].label  # "Left"/"Right"
            hands_list.append((label, hl))
    else:
        for hl in results.multi_hand_landmarks:
            mean_x = sum(lm.x for lm in hl.landmark) / 21.0
            label = "Left" if mean_x < 0.5 else "Right"
            hands_list.append((label, hl))

    def order_key(item):
        label, hl = item
        if label in ("Left", "Right"):
            return 0 if label == "Left" else 1
        mean_x = sum(lm.x for lm in hl.landmark) / 21.0
        return 0 if mean_x < 0.5 else 1

    hands_list.sort(key=order_key)
    return hands_list[:2]

def extract_84(img_bgr) -> Tuple[List[float], Dict]:
    """
    Returns (84-d feature, meta).
    Feature is two slots * (21 landmarks * (x,y)), padded with zeros if one hand missing.
    Coordinates are normalized by subtracting (min_x, min_y) across both hands.

    NOTE: If you want scale invariance too, also divide by (max_x-min_x, max_y-min_y).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None, {"detected": 0}

    hands_list = order_hands(results)
    xs = [lm.x for _, hl in hands_list for lm in hl.landmark]
    ys = [lm.y for _, hl in hands_list for lm in hl.landmark]
    if len(xs) == 0:
        return None, {"detected": 0}

    min_x, min_y = min(xs), min(ys)

    data84 = []
    for slot in range(2):
        if slot < len(hands_list):
            _, hl = hands_list[slot]
            for lm in hl.landmark:
                x = lm.x - min_x
                y = lm.y - min_y
                # Optional scale norm:
                # width = max(xs) - min_x
                # height = max(ys) - min_y
                # x = x / max(width, 1e-6)
                # y = y / max(height, 1e-6)
                data84.extend([x, y])
        else:
            data84.extend([0.0] * 42)

    return data84, {
        "detected": len(hands_list),
        "mp_results": results,  # keep if you want to visualize
    }

# -------------------------- Scans --------------------------
# Counters for logging
total_imgs_seen = 0
total_imgs_readable = 0
total_imgs_with_hands = 0

# For quick visualization (one per class if available)
examples_vis = {}  # class_name -> (img_rgb_with_landmarks, path, n_hands)

# Collect class names (folders directly under DATA_DIR)
class_names = [p.name for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
# Build stable label map
label_to_idx = {c: i for i, c in enumerate(class_names)}

# Static outputs
X_static: List[List[float]] = []
y_static: List[int] = []

# Dynamic outputs (sequences)
X_dyn: List[np.ndarray] = []     # each element: array shape (T, 84)
y_dyn: List[int] = []
seq_lengths: List[int] = []

for class_name in class_names:
    class_dir = DATA_DIR / class_name
    class_idx = label_to_idx[class_name]
    is_dynamic = class_name in DYNAMIC_LABELS

    if not is_dynamic:
        # --------- STATIC CLASS: images directly inside class_dir ----------
        imgs = list_images(class_dir)
        for img_path in imgs:
            total_imgs_seen += 1
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            total_imgs_readable += 1

            feat84, meta = extract_84(img_bgr)
            if feat84 is None:
                continue
            total_imgs_with_hands += 1

            X_static.append(feat84)
            y_static.append(class_idx)

            # keep first vis per class
            if class_name not in examples_vis:
                vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
                # draw all landmarks found
                res = meta["mp_results"]
                for hl in (res.multi_hand_landmarks or []):
                    mp_drawing.draw_landmarks(
                        vis, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                n_hands = len(res.multi_hand_landmarks or [])
                examples_vis[class_name] = (vis, str(img_path), n_hands)
    else:
        # --------- DYNAMIC CLASS: treat each subfolder as a sequence ----------
        seq_dirs = list_subdirs(class_dir)
        for seq_dir in seq_dirs:
            frames = list_images(seq_dir)
            seq_feats = []
            for fpath in frames:
                total_imgs_seen += 1
                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    continue
                total_imgs_readable += 1

                feat84, meta = extract_84(img_bgr)
                if feat84 is None:
                    # no hands in this frame; you can choose to drop or pad
                    # here we skip frame
                    continue
                total_imgs_with_hands += 1
                seq_feats.append(feat84)

                # keep one visualization per dynamic class if not set
                if class_name not in examples_vis:
                    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
                    res = meta["mp_results"]
                    for hl in (res.multi_hand_landmarks or []):
                        mp_drawing.draw_landmarks(
                            vis, hl, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                    n_hands = len(res.multi_hand_landmarks or [])
                    examples_vis[class_name] = (vis, str(fpath), n_hands)

            if len(seq_feats) > 0:
                arr = np.asarray(seq_feats, dtype=np.float32)  # shape (T, 84)
                X_dyn.append(arr)
                y_dyn.append(class_idx)
                seq_lengths.append(arr.shape[0])
            # else: sequence had no usable frames → drop it

# Clean up mediapipe
hands.close()

# -------------------------- Save --------------------------
if len(X_static) > 0:
    with open('data_static.pickle', 'wb') as f:
        pickle.dump(
            {'X': X_static, 'y': y_static, 'classes': class_names, 'label_to_idx': label_to_idx},
            f
        )

if len(X_dyn) > 0:
    with open('data_dynamic_seq.pickle', 'wb') as f:
        pickle.dump(
            {'X': X_dyn, 'lengths': seq_lengths, 'y': y_dyn, 'classes': class_names, 'label_to_idx': label_to_idx},
            f
        )

print("=== SUMMARY ===")
print(f"Classes discovered: {len(class_names)} -> {class_names}")
print(f"Images seen: {total_imgs_seen}")
print(f"Images readable: {total_imgs_readable}")
print(f"Images with >=1 hand: {total_imgs_with_hands}")
print(f"Static samples: {len(X_static)} saved to data_static.pickle" if len(X_static) else "Static: none")
print(f"Dynamic sequences: {len(X_dyn)} saved to data_dynamic_seq.pickle" if len(X_dyn) else "Dynamic: none")

# -------------------------- Visualize examples --------------------------
# Show up to 3 examples in class index order, if available
show_classes = class_names[:3]
to_show = [(c, examples_vis[c]) for c in show_classes if c in examples_vis]

if to_show:
    n = len(to_show)
    plt.figure(figsize=(5*n, 5))
    for i, (c, (img_vis, img_path, n_hands)) in enumerate(to_show, 1):
        plt.subplot(1, n, i)
        plt.imshow(img_vis)
        plt.title(f"{c} | {Path(img_path).name} | hands: {n_hands}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] No example visualizations available.")
 