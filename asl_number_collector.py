import os
import cv2
import time
from pathlib import Path
from datetime import datetime

# ============ CONFIG ============
DATA_DIR = Path("./data_numbers5")   # separate output folder for numbers

# Camera
CAMERA_INDEX = 0
PREVIEW_MIRROR = True

# Labels (numbers only)
NUMBER_LABELS = list("0123456789")

# Static capture
STILLS_PER_CLASS = 100
FRAME_SKIP = 2

# UI
WINDOW_NAME = "ASL Number Collector"
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
        draw_text(frame, f"{label} (number) | saved {saved}/{stills}", (20, 40))
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return "quit"
        if seen % frame_skip == 0:
            out_path = label_dir / f"img_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        seen += 1
    return "done"

def main():
    ensure_dir(DATA_DIR)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for label in NUMBER_LABELS:
            label_dir = DATA_DIR / label
            ensure_dir(label_dir)

            header = f"Number {label} — target {STILLS_PER_CLASS} stills"
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

        print("Done.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
