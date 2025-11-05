# scripts/select_threshold.py
import numpy as np, json
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score
from joblib import load

X = np.load('model_out/holistic_ds/X.npy')
y = np.load('model_out/holistic_ds/y.npy')
pipe = load('model_out/holistic_models/holistic_lr.joblib')

proba = pipe.predict_proba(X)[:,1]
prec, rec, thr = precision_recall_curve(y, proba)

# Ejemplo: umbral que maximiza F1
f1s = [f1_score(y, proba >= t) for t in thr]
best_t = float(thr[int(np.argmax(f1s))])

Path('model_out/holistic_models/threshold.json').write_text(json.dumps({
    "threshold": best_t
}, indent=2))
print("Umbral guardado:", best_t)
