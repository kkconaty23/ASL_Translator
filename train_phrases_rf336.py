import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import mediapipe as mp

# ====== CONFIG ======
DATA_DIR = Path("./phrases")           # your capture root
FRAMES_PER_SEQUENCE = 30               # must match your collector
IMG_GLOB = "frame_*.jpg"
MODEL_PATH = "model_rf_336_phrases.p"
LABELS_PATH = "label_names.json"
FEATURE_SPEC_PATH = "feature_spec.json"
MIN_DET_CONF = 0.5

# MediaPipe setup
mp_hands = mp.solutions.hands

# --------- IO helpers ---------
def list_sequences(root: Path) -> List[Tuple[str, Path]]:
    """
    Returns list of (phrase_text, seq_dir) for every sequence folder.
    Assumes: DATA_DIR/<slug>/seq_*/frame_*.jpg and optional _phrase.json with 'phrase'.
    """
    seqs = []
    for slug_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        phrase_text = slug_dir.name.replace("-", " ")
        pj = slug_dir / "_phrase.json"
        if pj.exists():
            try:
                phrase_text = json.loads(pj.read_text(encoding="utf-8")).get("phrase", phrase_text)
            except Exception:
                pass
        for seq_dir in sorted([p for p in slug_dir.iterdir() if p.is_dir() and p.name.startswith("seq_")]):
            if any(seq_dir.glob(IMG_GLOB)):
                seqs.append((phrase_text, seq_dir))
    return seqs

def read_sequence_frames(seq_dir: Path, target_len: int) -> List[np.ndarray]:
    frames = sorted(seq_dir.glob(IMG_GLOB))
    imgs = []
    for fp in frames[:target_len]:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        imgs.append(img)
    if len(imgs) == 0:
        return []
    while len(imgs) < target_len:
        imgs.append(imgs[-1])
    return imgs[:target_len]

# --------- Feature helpers ---------
def hand_landmarks_xy(image_bgr, hands_ctx) -> Optional[np.ndarray]:
    """
    Returns (84,) = left(42) + right(42) per frame.
    For each visible hand: normalize by translating to wrist (id=0) and scaling by palm size (wrist->middle_mcp id=9).
    Missing hand -> zeros(42).
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands_ctx.process(img_rgb)

    left = None
    right = None
    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = hd.classification[0].label.lower()  # 'left'/'right'
            xy = np.array([(pt.x, pt.y) for pt in lm.landmark], dtype=np.float32)  # (21,2)
            wrist = xy[0]
            xy -= wrist
            palm = np.linalg.norm(xy[9]) + 1e-6
            xy /= palm
            if label == "left":
                left = xy
            else:
                right = xy

    def pack(xy): return xy.reshape(-1)  # (21,2)->(42,)
    left_vec  = np.zeros(42, dtype=np.float32) if left  is None else pack(left)
    right_vec = np.zeros(42, dtype=np.float32) if right is None else pack(right)
    return np.concatenate([left_vec, right_vec], axis=0)  # (84,)

def features_336_from_seq(M: np.ndarray) -> np.ndarray:
    """Aggregate (T,84) -> (336,) using mean/std and diff mean/std."""
    mu  = M.mean(axis=0)
    sd  = M.std(axis=0) + 1e-6
    dM  = np.diff(M, axis=0)
    dmu = dM.mean(axis=0)
    dsd = dM.std(axis=0) + 1e-6
    return np.concatenate([mu, sd, dmu, dsd], axis=0).astype(np.float32)

# --------- Hand-invariance (mirror + swap) ---------
def _negate_x_inplace_42(vec42: np.ndarray):
    """vec42 = [x0,y0,x1,y1,...,x20,y20] -> negate X only (inplace)."""
    vec42[0::2] *= -1.0
    return vec42

def mirror_swap_seq_Tx84(M: np.ndarray) -> np.ndarray:
    """
    (T,84) = [left(42), right(42)] -> mirror (x->-x) both halves, then swap halves.
    Returns a new (T,84).
    """
    left  = M[:, :42].copy()
    right = M[:, 42:].copy()
    _negate_x_inplace_42(left)
    _negate_x_inplace_42(right)
    return np.concatenate([right, left], axis=1)

# --------- Dataset build ---------
def build_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    seqs = list_sequences(root)
    if not seqs:
        raise RuntimeError(f"No sequences found under {root.resolve()}")

    phrases = sorted(sorted(set([p for p, _ in seqs])), key=str.lower)
    label2idx = {p: i for i, p in enumerate(phrases)}

    X, y = [], []

    with mp_hands.Hands(
        static_image_mode=True,   # process frames independently
        max_num_hands=2,
        min_detection_confidence=MIN_DET_CONF
    ) as hands_ctx:
        for phrase_text, seq_dir in tqdm(seqs, desc="Extracting features"):
            frames = read_sequence_frames(seq_dir, FRAMES_PER_SEQUENCE)
            if not frames:
                continue

            # Build per-frame 84D once
            feats = []
            for img in frames:
                v84 = hand_landmarks_xy(img, hands_ctx)
                if v84 is None:
                    feats = []
                    break
                feats.append(v84)
            if not feats:
                continue
            M = np.stack(feats, axis=0)  # (T,84)

            # Original 336-D
            X.append(features_336_from_seq(M))
            y.append(label2idx[phrase_text])

            # Augmented: mirror + swap hands
            M_m = mirror_swap_seq_Tx84(M)
            X.append(features_336_from_seq(M_m))
            y.append(label2idx[phrase_text])

    if not X:
        raise RuntimeError("No usable sequences (all failed landmark detection?).")

    X = np.stack(X, axis=0)  # (N,336)
    y = np.array(y, dtype=np.int64)
    return X, y, phrases

# --------- Train ---------
def train_and_save(X, y, phrases):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    print("\n=== Evaluation ===")
    print(classification_report(yte, ypred, target_names=phrases, digits=3))
    print("Confusion matrix:\n", confusion_matrix(yte, ypred))

    # Save model & metadata
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model_name": "RandomForest_RF336_Phrases",
                "model": clf,
                "feature_dim": X.shape[1],
                "label_names": phrases,
                "normalization": "per-frame wrist-centered, palm-scale; xy only; left+right order",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    Path(LABELS_PATH).write_text(json.dumps(phrases, indent=2), encoding="utf-8")
    Path(FEATURE_SPEC_PATH).write_text(json.dumps({
        "per_frame": "xy only, 21 landmarks per hand, left then right (zeros if missing)",
        "time_agg": ["mean", "std", "diff_mean", "diff_std"],
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "feature_dim": 336,
        "augmentation": "mirror x for both hands, then swap left/right halves"
    }, indent=2), encoding="utf-8")

    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved labels → {LABELS_PATH}")
    print(f"Saved spec   → {FEATURE_SPEC_PATH}")

def main():
    X, y, phrases = build_dataset(DATA_DIR)
    print(f"Dataset: {X.shape[0]} sequences, feature_dim={X.shape[1]}, classes={len(phrases)}")
    train_and_save(X, y, phrases)

if __name__ == "__main__":
    main()
