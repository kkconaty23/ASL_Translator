import os
import cv2
import time
import json
import re
from pathlib import Path
from datetime import datetime

# ============ CONFIG ============
DATA_DIR = Path("./phrases1")   # root for phrase clips

# Camera
CAMERA_INDEX = 0               # 0 default webcam; try 1/2 if needed
PREVIEW_MIRROR = True          # selfie preview
TARGET_WIDTH  = 1280
TARGET_HEIGHT = 720
TARGET_FPS    = 30

# Dynamic capture (phrases)
SEQUENCES_PER_PHRASE = 40      # how many takes per phrase
FRAMES_PER_SEQUENCE  = 30      # frames per take (fixed length)
FRAME_GAP_MS         = 33      # ~30 fps pacing (1000/30)

# UI
WINDOW_NAME   = "ASL Phrase Collector (Dynamic)"
COUNTDOWN_SEC = 2

# 20 common phrases
PHRASES = [
    # "MILK",
    # "MORE",
    # "ALL DONE",
    # "EAT",
    # "DRINK",
    # "SLEEP",
    # "DIAPER",
    # "BATH",
    # "MOM",
    # "DAD",
    # "PLEASE",
    # "THANK YOU",
    # "HELP",
    # "LOVE YOU",
    # "SORRY",
    # "PLAY",
    "BOOK",
    "BALL",
    "DOG",
    "MUSIC",
    "HEY, KEEP IT PG!"
]

# ===============================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    # filesystem-friendly folder name
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "phrase"

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

def collect_dynamic_sequence(cap, seq_dir: Path, frames_needed: int, gap_ms: int, overlay_msg: str = ""):
    ensure_dir(seq_dir)
    kept_frames = 0
    # approximate fps using pacing
    for i in range(frames_needed):
        t0 = time.time()
        frame = read_frame(cap)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        # progress bar
        prog = int(((i + 1) / frames_needed) * (w - 60))
        cv2.rectangle(frame, (20, h - 40), (20 + prog, h - 20), (0, 255, 0), -1)
        draw_text(frame, f"Recording {i+1}/{frames_needed}", (20, 40))
        if overlay_msg:
            draw_text(frame, overlay_msg, (20, 80), 0.9, (0, 200, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.imwrite(str(seq_dir / f"frame_{i:03d}.jpg"), frame)
        kept_frames += 1

        # pacing
        elapsed_ms = int((time.time() - t0) * 1000)
        wait_ms = max(1, gap_ms - elapsed_ms)
        k = cv2.waitKey(wait_ms) & 0xFF
        if k == ord('q'):
            return "quit", kept_frames

    return "done", kept_frames

def ask_keep_or_redo(cap, prompt="Keep this take?  K=keep   R=redo   Q=quit"):
    while True:
        frame = read_frame(cap)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        draw_text(frame, prompt, (20, h - 20), 0.9, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('k'), ord('K')):
            return "keep"
        if k in (ord('r'), ord('R')):
            return "redo"
        if k == ord('q'):
            return "quit"

def write_seq_metadata(seq_dir: Path, phrase: str, frames: int):
    meta = {
        "phrase": phrase,
        "slug": seq_dir.parent.name,
        "sequence_dir": seq_dir.name,
        "frames": frames,
        "frame_gap_ms": FRAME_GAP_MS,
        "intended_fps": round(1000 / max(1, FRAME_GAP_MS), 2),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    with open(seq_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    ensure_dir(DATA_DIR)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Try another CAMERA_INDEX (0/1/2) and close other apps using the camera.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for phrase_idx, phrase in enumerate(PHRASES, start=1):
            slug = slugify(phrase)
            phrase_dir = DATA_DIR / slug
            ensure_dir(phrase_dir)

            for seq_idx in range(SEQUENCES_PER_PHRASE):
                header = f"[{phrase_idx}/{len(PHRASES)}] Phrase: {phrase} — take {seq_idx+1}/{SEQUENCES_PER_PHRASE}"
                action = wait_for_space_skip_quit(cap, header)
                if action == "quit":
                    print("Quit requested.")
                    return
                if action == "skip":
                    # skip this take and continue to next take
                    continue
                if not countdown(cap, COUNTDOWN_SEC):
                    print("Quit during countdown.")
                    return

                while True:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    seq_dir = phrase_dir / f"seq_{seq_idx:03d}_{stamp}"
                    status, frames_saved = collect_dynamic_sequence(
                        cap,
                        seq_dir,
                        FRAMES_PER_SEQUENCE,
                        FRAME_GAP_MS,
                        overlay_msg=f'Phrase: "{phrase}"'
                    )
                    if status == "quit":
                        print("Quit during recording.")
                        return

                    choice = ask_keep_or_redo(cap)
                    if choice == "quit":
                        print("Quit after take.")
                        return
                    if choice == "keep":
                        write_seq_metadata(seq_dir, phrase, frames_saved)
                        break
                    if choice == "redo":
                        # remove the directory we just wrote and try again
                        try:
                            for f in seq_dir.glob("*"):
                                f.unlink()
                            seq_dir.rmdir()
                        except Exception:
                            pass
                        # loop to re-record

            # optional: phrase-level marker file with summary
            phrase_summary = {
                "phrase": phrase,
                "slug": slug,
                "sequences_target": SEQUENCES_PER_PHRASE,
                "frames_per_sequence": FRAMES_PER_SEQUENCE
            }
            with open(phrase_dir / "_phrase.json", "w", encoding="utf-8") as f:
                json.dump(phrase_summary, f, indent=2)

        print("All phrases completed. Goodbye!")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
