# train_holistic_cv_full.py
import json
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from joblib import dump
from sklearn.metrics import roc_auc_score, make_scorer


# Rutas (ajusta si usaste otras)
X_PATH = Path('model_out/holistic_ds/X.npy')
Y_PATH = Path('model_out/holistic_ds/y.npy')
OUT_DIR = Path('model_out/holistic_models'); OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / 'holistic_lr.joblib'
METRICS_PATH = OUT_DIR / 'cv_metrics.json'

RANDOM_STATE = 42
N_SPLITS = 5  # 5-fold estratificado



def auc_from_proba(y_true, y_pred_proba):
    """Compute AUC from true labels and predicted probabilities"""
    return roc_auc_score(y_true, y_pred_proba)


def auc_scorer(estimator, X, y):
    """Evalúa AUC usando predict_proba de manera compatible."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)

def main():
    # --- carga del dataset completo ---
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- pipeline (sin fugas: scaler dentro del CV) ---
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE
        ))
    ])

    from sklearn.metrics import roc_auc_score, make_scorer

    def auc_scorer(estimator, X, y):
        """Evalúa AUC usando predict_proba de manera compatible."""
        y_prob = estimator.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_prob)

    scoring = {
        'auc': 'roc_auc',  # usa el scorer interno oficial
        'acc': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'prec': make_scorer(precision_score),
        'rec': make_scorer(recall_score),
    }


    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    # --- resumen CV ---
    summary = {k: {
        "mean": float(np.mean(v)),
        "std":  float(np.std(v))
    } for k, v in cv_res.items() if k.startswith('test_')}

    print("=== CV (StratifiedKFold) ===")
    for m, stats in summary.items():
        print(f"{m}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print("Métricas CV guardadas en:", METRICS_PATH)

    # --- entrenamiento final con el 100% del dataset ---
    pipe.fit(X, y)
    dump(pipe, MODEL_PATH)
    print("Modelo final entrenado con 100% del dataset y guardado en:", MODEL_PATH)

if __name__ == "__main__":
    main()
