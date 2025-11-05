# train_holistic.py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump
from pathlib import Path
# Ahora:
X_PATH = Path('model_out/holistic_ds/X.npy')
Y_PATH = Path('model_out/holistic_ds/y.npy')
OUT_DIR = Path('model_out/holistic_models')

OUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
POS_LABEL = 1

def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # seguridad por si hay NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE
        ))
    ])

    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    yproba = pipe.predict_proba(Xte)[:, 1]

    report = classification_report(yte, ypred, output_dict=True)
    auc = roc_auc_score(yte, yproba)

    print("Confusion matrix:\n", confusion_matrix(yte, ypred))
    print("AUC:", auc)
    print("Report:\n", json.dumps(report, indent=2))

    dump(pipe, OUT_DIR/'holistic_lr.joblib')
    print("Modelo guardado en:", OUT_DIR/'holistic_lr.joblib')

if __name__ == "__main__":
    main()
