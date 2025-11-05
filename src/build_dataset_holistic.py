# build_dataset_holistic.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from absl import logging as absl_logging
absl_logging.set_verbosity("error")

import cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from features_holistic import extract_features_bgr

DATASET_DIR = Path('resources/trainImage')
OUT_DIR     = Path('model_out/holistic_ds')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {'correcta': 1, 'incorrecta': 0}
VALID_EXT = {'.jpg', '.jpeg', '.png'}

def iter_images():
    items = []
    for cls, y in CLASSES.items():
        for p in (DATASET_DIR/cls).glob('**/*'):
            if p.suffix.lower() in VALID_EXT:
                items.append((p, y))
    return items

def main():
    items = iter_images()
    X, y = [], []
    dropped = 0

    mp_holistic = mp.solutions.holistic
    # model_complexity=0 acelera bastante sin perder mucho
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=0,
                              smooth_landmarks=False, enable_segmentation=False,
                              refine_face_landmarks=False) as holistic:
        for (path, label) in tqdm(items, desc="Extrayendo features", unit="img"):
            img = cv2.imread(str(path))
            if img is None:
                dropped += 1; continue
            try:
                feat = extract_features_bgr(img, holistic=holistic)
                if not np.isfinite(feat).all():
                    dropped += 1; continue
                X.append(feat); y.append(label)
            except Exception:
                dropped += 1

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUT_DIR/'X.npy', X)
    np.save(OUT_DIR/'y.npy', y)
    print(f"Guardado: X{X.shape}, y{y.shape}, descartados: {dropped}")

if __name__ == "__main__":
    main()
