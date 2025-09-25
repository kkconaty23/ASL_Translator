import os
import cv2
import time
from pathlib import Path
from datetime import datetime

# ============ CONFIG ============
DATA_DIR = Path("./data")

# Camera
CAMERA_INDEX = 0              # 0 default webcam; try 1/2 if needed
PREVIEW_MIRROR = True         # selfie preview

# Labels
STATIC_LABELS  = list("ABCDEFGHIKLMNOPQRSTUVWXY")
DYNAMIC_LABELS = list("JZ")

# Static capture
STILLS_PER_CLASS = 100        # how many still frames per static letter
FRAME_SKIP = 2                # save every Nth frame (variety). 1 = save every frame

# Dynamic capture (for J/Z)
SEQUENCES_PER_CLASS = 50      # how many sequences per dynamic letter
FRAMES_PER_SEQUENCE = 30      # frames per sequence (fixed length)
DYNAMIC_FRAME_GAP_MS = 33     # ~30 fps (1000/30) between frames

# UI
WINDOW_NAME = "ASL Collector"
COUNTDOWN_SEC = 2
# ===============================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_text(img, text, xy, scale=0.9, color=(0, 255, 0), thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    if PREVIEW_MIRROR:
        frame = cv2.flip(frame, 1)
    return frame

def wait_for_space_skip_quit(cap, header):
    """Show live preview until SPACE (start), N (skip), or Q (quit)."""
    while True:
        frame = read_frame(cap)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        draw_text(frame, header, (20, 40), 0.9, (0, 255, 255), 2)
        draw_text(frame, "SPACE=start   N=skip   Q=quit", (20, h - 20), 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            return "start"
        if k == ord('n'):
            return "skip"
        if k == ord('q'):
            return "quit"

def countdown(cap, seconds=COUNTDOWN_SEC):
    t0 = time.time()
    while True:
        frame = read_frame(cap)
        if frame is None:
            continue
        remain = max(0, seconds - int(time.time() - t0))
        draw_text(frame, f"Starting in: {remain}", (20, 60), 1.2, (0, 255, 255), 3)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        if time.time() - t0 >= seconds:
            return True

def collect_static_class(cap, label_dir: Path, label: str, stills: int, frame_skip: int):
    saved = 0
    seen = 0
    while saved < stills:
        frame = read_frame(cap)
        if frame is None:
            continue
        # show preview while capturing
        draw_text(frame, f"{label} (static) | saved {saved}/{stills}", (20, 40))
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return "quit"

        # save every Nth frame
        if seen % frame_skip == 0:
            out_path = label_dir / f"img_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        seen += 1
    return "done"

def collect_dynamic_sequence(cap, seq_dir: Path, frames_needed: int, gap_ms: int):
    ensure_dir(seq_dir)
    for i in range(frames_needed):
        t0 = time.time()
        frame = read_frame(cap)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        # simple progress bar
        prog = int(((i + 1) / frames_needed) * (w - 60))
        cv2.rectangle(frame, (20, h - 40), (20 + prog, h - 20), (0, 255, 0), -1)
        draw_text(frame, f"Recording {i+1}/{frames_needed}", (20, 40))
        cv2.imshow(WINDOW_NAME, frame)

        cv2.imwrite(str(seq_dir / f"frame_{i:03d}.jpg"), frame)

        # pacing
        elapsed_ms = int((time.time() - t0) * 1000)
        wait_ms = max(1, gap_ms - elapsed_ms)
        k = cv2.waitKey(wait_ms) & 0xFF
        if k == ord('q'):
            return "quit"
    return "done"

def main():
    ensure_dir(DATA_DIR)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Try another CAMERA_INDEX (0/1/2) and close other apps using the camera.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Iterate all labels
    all_labels = STATIC_LABELS + DYNAMIC_LABELS
    try:
        for label in all_labels:
            label_dir = DATA_DIR / label
            ensure_dir(label_dir)

            if label in STATIC_LABELS:
                header = f"Label {label} (STATIC) — target {STILLS_PER_CLASS} stills"
                action = wait_for_space_skip_quit(cap, header)
                if action == "quit":
                    break
                if action == "skip":
                    continue
                if not countdown(cap):
                    break
                status = collect_static_class(cap, label_dir, label, STILLS_PER_CLASS, FRAME_SKIP)
                if status == "quit":
                    break

            else:  # dynamic (J/Z)
                for seq_idx in range(SEQUENCES_PER_CLASS):
                    header = f"Label {label} (DYNAMIC) — sequence {seq_idx+1}/{SEQUENCES_PER_CLASS}"
                    action = wait_for_space_skip_quit(cap, header)
                    if action == "quit":
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    if action == "skip":
                        continue
                    if not countdown(cap):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    seq_dir = label_dir / f"seq_{seq_idx:03d}_{stamp}"
                    status = collect_dynamic_sequence(cap, seq_dir, FRAMES_PER_SEQUENCE, DYNAMIC_FRAME_GAP_MS)
                    if status == "quit":
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        print("Done.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
