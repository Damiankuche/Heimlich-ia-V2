# src/serve_inference.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from pathlib import Path
import base64, cv2, numpy as np, json, uuid
import uvicorn
from typing import List
from datetime import datetime

from .features_holistic import run_holistic, draw_holistic
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 1=oculta INFO, 2=oculta también WARNING, 3=oculta ERROR

# silenciar logs de absl (los que imprime MediaPipe)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# (opcional) por si tu runtime usa el logger estándar
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import mediapipe as mp  # <-- después de silenciar

OOD_MIN_PRESENCE = 1.0     # al menos 1 canal presente (pose o una mano)
OOD_MIN_NONZERO  = 10       # cantidad mínima de valores no-cero en features "geométricas"
OOD_MIN_STD      = 1e-4     # varianza mínima (evita vectores casi constantes)

MODEL_PATH   = 'model_out/holistic_models/holistic_lr.joblib'
THRESH_PATH  = 'model_out/holistic_models/threshold.json'  # opcional
SAVE_DIR     = Path('inference_logs')                      # carpeta base
SAVE_ORIGINAL = False                                      # cambia a True si querés guardar también la imagen original

app = FastAPI(title="Holistic Classifier + Save Overlay")

class B64Image(BaseModel):
    images: List[str]  # puede venir con o sin encabezado data:...

def b64_to_bgr(b64_str: str):
    raw = base64.b64decode(b64_str.split(',')[-1])
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Base64 inválido")
    return img

def is_ood_feature(feat: np.ndarray) -> bool:
    """
    Heurísticas baratas para detectar OOD / no detección:
    - presencia: últimas 3 posiciones del vector (pose, left, right) si ADD_PRESENCE_FLAGS=True
    - escasez de señal: muy pocos no-cero
    - varianza ínfima: vector 'plano'
    """
    # asumimos que las últimas 3 posiciones son presence flags [pose, left, right]
    presence = feat[-3:]
    if float(np.sum(presence)) < OOD_MIN_PRESENCE:
        return True

    core = feat[:-3]  # todo menos presence flags
    nonzero = int(np.count_nonzero(np.abs(core) > 1e-6))
    if nonzero < OOD_MIN_NONZERO:
        return True

    if float(np.std(core)) < OOD_MIN_STD:
        return True

    return False

def ensure_dirs():
    (SAVE_DIR / 'correcta').mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / 'incorrecta').mkdir(parents=True, exist_ok=True)

def safe_filename(label: str, prob: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{label}_{ts}_{prob:.3f}.jpg"

@app.on_event("startup")
def load_artifacts():
    global pipe, THRESH, holistic
    ensure_dirs()

    pipe = load(MODEL_PATH)
    THRESH = 0.7
    p = Path(THRESH_PATH)
    if p.exists():
        THRESH = json.loads(p.read_text()).get("threshold", 0.7)

    # Crear UNA instancia reutilizable de Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False
    )

@app.on_event("shutdown")
def close_holistic():
    try:
        holistic.close()
    except Exception:
        pass

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_one(payload: B64Image):
    try:
        # 1) decodificar
        img = b64_to_bgr(payload.images[0])

        # 2) features + resultados para dibujar
        feat, res = run_holistic(img, holistic=holistic)
        feat = feat.astype(np.float32).reshape(1, -1)

        # 3) REGLA OOD: si no reconoce landmarks / señal rara => incorrecta
        if is_ood_feature(feat[0]):
            annotated = draw_holistic(img, res)
            fname = safe_filename("incorrecta", 0.0)
            out_path = (SAVE_DIR / "incorrecta" / fname)
            ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                out_path.write_bytes(buf.tobytes())

            return {"prediction": "incorrecta", "score": 0.0, "reason": "ood"}

        # 4) predicción
        prob = float(pipe.predict_proba(feat)[0, 1])
        label = "correcta" if prob >= THRESH else "incorrecta"

        # 5) dibujar y guardar
        annotated = draw_holistic(img, res)
        fname = safe_filename(label, prob)
        out_path = (SAVE_DIR / label / fname)
        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("No se pudo codificar la imagen anotada")
        out_path.write_bytes(buf.tobytes())

        # (opcional) guardar original
        if SAVE_ORIGINAL:
            orig_name = fname.replace(".jpg", "_orig.jpg")
            orig_path = (SAVE_DIR / label / orig_name)
            ok2, buf2 = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok2:
                orig_path.write_bytes(buf2.tobytes())

        print("prediction: ",label, " score: ",prob)

        # 5) responder sin imagen
        return {
            "prediction": label,
            "score": prob
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
