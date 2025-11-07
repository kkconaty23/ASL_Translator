import cv2, json, shutil
from pathlib import Path

# === CONFIG ===
DATA_DIR = Path("./phrases1")   # root of your collected data
IMG_GLOB = "frame_*.jpg"
SUFFIX   = "_mirrored"         # appended to sequence folder names

def mirror_sequence(seq_dir: Path):
    out_dir = seq_dir.parent / f"{seq_dir.name}{SUFFIX}"
    if out_dir.exists():
        print(f"[SKIP] {out_dir} already exists")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(seq_dir.glob(IMG_GLOB))
    if not frame_paths:
        print(f"[WARN] No frames in {seq_dir}")
        return

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        flipped = cv2.flip(img, 1)  # horizontal mirror
        out_path = out_dir / fp.name
        cv2.imwrite(str(out_path), flipped)

    # copy meta if present
    meta = seq_dir / "meta.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            data["augmented_from"] = seq_dir.name
            data["augmentation"] = "mirrored"
            (out_dir / "meta.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            shutil.copy2(meta, out_dir / "meta.json")

    print(f"[OK] Mirrored → {out_dir}")

def main():
    phrase_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    for phrase_dir in phrase_dirs:
        seq_dirs = [s for s in phrase_dir.iterdir() if s.is_dir() and s.name.startswith("seq_")]
        for seq_dir in seq_dirs:
            mirror_sequence(seq_dir)

if __name__ == "__main__":
    main()
