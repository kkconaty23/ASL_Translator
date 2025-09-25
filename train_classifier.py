import os
import pickle
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ------------------ Config ------------------
STATIC_PICKLE = Path("data_static.pickle")      # from the unified extractor I sent
DYNAMIC_PICKLE = Path("data_dynamic_seq.pickle")
LEGACY_PICKLE = Path("data.pickle")             # your original format: {'data': [...], 'labels': [...]}
MODEL_OUT = Path("model_rf_336.p")
META_OUT  = Path("model_rf_336_meta.p")

TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_KW = dict(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)
# --------------------------------------------

def make_336_from_sequence(seq_Tx84: np.ndarray) -> np.ndarray:
    """
    Build 336-D features from a sequence of shape (T, 84):
      [ mean(84), std(84), last-first (84), mean_abs_diff(84) ]
    """
    if seq_Tx84.ndim != 2 or seq_Tx84.shape[1] != 84:
        raise ValueError(f"Expected (T,84), got {seq_Tx84.shape}")

    mean_ = seq_Tx84.mean(axis=0)
    std_  = seq_Tx84.std(axis=0)
    first = seq_Tx84[0]
    last  = seq_Tx84[-1]
    last_minus_first = last - first

    if seq_Tx84.shape[0] >= 2:
        diffs = np.diff(seq_Tx84, axis=0)
        mean_abs_diff = np.mean(np.abs(diffs), axis=0)
    else:
        mean_abs_diff = np.zeros_like(mean_)

    return np.concatenate([mean_, std_, last_minus_first, mean_abs_diff], axis=0)  # (336,)

def make_336_from_static(vec84: np.ndarray) -> np.ndarray:
    """
    Turn a single 84-D static vector into the same 336-D feature layout:
      mean=vec, std=0, last-first=0, mean_abs_diff=0
    """
    vec84 = np.asarray(vec84, dtype=np.float32).reshape(-1)
    assert vec84.shape[0] == 84
    zeros = np.zeros_like(vec84)
    return np.concatenate([vec84, zeros, zeros, zeros], axis=0)

def load_dataset():
    """
    Loads whichever files exist, builds X (336-D) and y (int labels),
    and returns (X, y, classes) with consistent class index mapping.
    Priority:
      1) data_static.pickle + data_dynamic_seq.pickle (if present)
      2) data_static.pickle only
      3) data.pickle (legacy)
    """
    X_parts, y_parts = [], []
    classes = None
    label_to_idx = None

    if STATIC_PICKLE.exists() or DYNAMIC_PICKLE.exists():
        # Prefer the newer format
        if STATIC_PICKLE.exists():
            d = pickle.load(open(STATIC_PICKLE, "rb"))
            Xs = d["X"]            # list of 84-d
            ys = d["y"]            # ints aligned to d['classes']
            classes = d["classes"]
            label_to_idx = d["label_to_idx"]
            X_parts.append(np.vstack([make_336_from_static(x) for x in Xs]))
            y_parts.append(np.asarray(ys, dtype=int))

        if DYNAMIC_PICKLE.exists():
            d = pickle.load(open(DYNAMIC_PICKLE, "rb"))
            Xseqs = d["X"]         # list of (T,84) arrays
            ys    = d["y"]         # ints aligned to d['classes']
            if classes is None:
                classes = d["classes"]
                label_to_idx = d["label_to_idx"]
            else:
                # Sanity: class list should match
                assert classes == d["classes"], "Class mismatch between static and dynamic pickles."

            X_dyn = np.vstack([make_336_from_sequence(seq) for seq in Xseqs])
            y_dyn = np.asarray(ys, dtype=int)
            X_parts.append(X_dyn)
            y_parts.append(y_dyn)

        X = np.vstack(X_parts) if X_parts else np.empty((0, 336), dtype=np.float32)
        y = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=int)

        if X.shape[0] == 0:
            raise RuntimeError("No samples found in data_static.pickle / data_dynamic_seq.pickle.")

        return X, y, classes, label_to_idx

    elif LEGACY_PICKLE.exists():
        # Legacy: strings in data['labels']
        d = pickle.load(open(LEGACY_PICKLE, "rb"))
        X84 = np.asarray(d["data"], dtype=np.float32)   # shape (N,84)
        labels = np.asarray(d["labels"])                # strings like "0","1","2" or "A","B",...

        # Build a deterministic label map
        classes = sorted(list(set(labels)))
        label_to_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([label_to_idx[s] for s in labels], dtype=int)

        X = np.vstack([make_336_from_static(x) for x in X84])  # (N,336)
        return X, y, classes, label_to_idx

    else:
        raise FileNotFoundError("No dataset found. Expected data_static.pickle / data_dynamic_seq.pickle or data.pickle.")

def main():
    X, y, classes, label_to_idx = load_dataset()
    print(f"[INFO] Samples: {X.shape[0]}, Feature dim: {X.shape[1]} (should be 336), Classes: {len(classes)} -> {classes}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train RF
    model = RandomForestClassifier(**RF_KW)
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")

    # --- Robust metrics ---
    import numpy as np
    labels_all = list(range(len(classes)))  # force 25 labels in order of `classes`

    # Confusion matrix with fixed label set
    cm = confusion_matrix(y_test, y_pred, labels=labels_all)
    print("\nConfusion Matrix (25x25 with fixed label order):\n", cm)

    # Which classes are missing from y_test?
    missing = sorted(set(labels_all) - set(np.unique(y_test)))
    if missing:
        print("Classes with NO test samples:",
              [classes[i] for i in missing])

    # Classification report with fixed labels & safe zero handling
    try:
        print("\nClassification Report:\n",
              classification_report(y_test, y_pred,
                                    labels=labels_all,
                                    target_names=classes,
                                    zero_division=0))
    except Exception as e:
        print("[WARN] classification_report failed:", e)

    # --- Always save, even if metrics threw earlier ---
    try:
        with open(MODEL_OUT, "wb") as f:
            pickle.dump({"model": model, "classes": classes}, f)
        with open(META_OUT, "wb") as f:
            pickle.dump(
                {"feature_layout": ["mean(84)", "std(84)", "last-first(84)", "mean_abs_diff(84)"],
                 "dim": 336,
                 "label_to_idx": label_to_idx,
                 "rf_params": RF_KW},
                f
            )
        print(f"\n[OK] Saved model to {MODEL_OUT.resolve()}")
        print(f"[OK] Saved meta  to {META_OUT.resolve()}")
    except Exception as e:
        print("[ERR] Failed to save model/meta:", e)


if __name__ == "__main__":
    main()
