import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt  # used for visualization

mp_hands = mp.solutions.hands  # Hands model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Directory where the data is stored. Change this path if needed.

# Acceptable image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

data = []    # landmarks for all images
labels = []  # labels for all images

total_imgs_seen = 0
total_imgs_readable = 0
total_imgs_with_hands = 0

# --- NEW: hold one example (with landmarks drawn) per class ---
examples = {}  # key: class_name, value: (rgb_image_with_landmarks, img_path)

for dir_ in sorted(os.listdir(DATA_DIR)):  # class folders (e.g., "0", "1", "2")
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # skip files like .gitignore

    for img_name in sorted(os.listdir(dir_path)):
        img_path = os.path.join(dir_path, img_name)

        # Skip non-files / hidden / wrong extension
        if not os.path.isfile(img_path):
            continue
        if img_name.startswith('.'):
            continue
        _, ext = os.path.splitext(img_name)
        if ext.lower() not in IMAGE_EXTS:
            continue

        total_imgs_seen += 1

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        total_imgs_readable += 1

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Use ONLY the first detected hand so every sample has 42 features (21 landmarks * 2 coords)
            first = results.multi_hand_landmarks[0]

            xs = [lm.x for lm in first.landmark]
            ys = [lm.y for lm in first.landmark]
            min_x, min_y = min(xs), min(ys)

            data_aux = []
            for lm in first.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            # Optional sanity check:
            # assert len(data_aux) == 42

            data.append(data_aux)
            labels.append(dir_)


        # --- NEW: save first good visualization per class ---
        if dir_ not in examples:
            vis = img_rgb.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            examples[dir_] = (vis, img_path)

# Clean up mediapipe resources
hands.close()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("=== SUMMARY ===")
print(f"Classes found (that produced samples): {len(set(labels))}")
print(f"Images seen: {total_imgs_seen}")
print(f"Images readable: {total_imgs_readable}")
print(f"Images with hands: {total_imgs_with_hands}")
print(f"Samples saved: {len(data)} to data.pickle")

# --- NEW: visualize one example per class 0,1,2 (if available) ---
# If you want strictly the folders named "0","1","2", keep this order; otherwise display whatever exists.
desired_order = ["0", "1", "2"]
to_show = [(c, examples[c]) for c in desired_order if c in examples]

if to_show:
    n = len(to_show)
    plt.figure(figsize=(5 * n, 5))
    for i, (class_name, (img_vis, img_path)) in enumerate(to_show, start=1):
        plt.subplot(1, n, i)
        plt.imshow(img_vis)
        plt.title(f"Class {class_name}\n{os.path.basename(img_path)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] No examples to display (no detections found in classes 0/1/2).")
