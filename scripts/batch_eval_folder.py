# scripts/batch_eval_folder.py
from pathlib import Path
import cv2, numpy as np, json
from joblib import load
from features_holistic import extract_features_bgr

MODEL = load('model_out/holistic_models/holistic_lr.joblib')
THR   = json.loads(Path('model_out/holistic_models/threshold.json').read_text())["threshold"]
ROOT  = Path('resources/trainImage')  # correcta / incorrecta

def eval_dir(dir_path, true_label):
    ok, total = 0, 0
    bad = []
    for p in dir_path.rglob('*'):
        if p.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        img = cv2.imread(str(p));
        if img is None: continue
        f = extract_features_bgr(img).reshape(1,-1)
        prob = float(MODEL.predict_proba(f)[0,1])
        pred = 1 if prob>=THR else 0
        total += 1; ok += int(pred==true_label)
        if pred!=true_label:
            bad.append((str(p), prob))
    acc = ok/total if total else 0
    return acc, bad

acc_pos, bad_pos = eval_dir(ROOT/'correcta', 1)
acc_neg, bad_neg = eval_dir(ROOT/'incorrecta', 0)
print(f"Acc correcta: {acc_pos:.4f} | Acc incorrecta: {acc_neg:.4f}")
print(f"Falsos negativos: {len(bad_pos)} | Falsos positivos: {len(bad_neg)}")
