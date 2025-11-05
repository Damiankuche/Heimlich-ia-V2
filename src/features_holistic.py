# src/features_holistic.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from absl import logging as absl_logging
absl_logging.set_verbosity("error")

import cv2
import numpy as np
import mediapipe as mp

USE_FACE = False
ADD_PRESENCE_FLAGS = True

mp_holistic = mp.solutions.holistic
mp_pose     = mp.solutions.pose
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils

POSE_LM = mp_pose.PoseLandmark

# ---------- core: extracción de features ----------
def _landmarks_to_np(landmarks):
    arr = []
    for lm in landmarks:
        arr.extend([lm.x, lm.y, getattr(lm, 'z', 0.0), getattr(lm, 'visibility', 1.0)])
    return np.array(arr, dtype=np.float32)

def _extract_raw(res):
    vecs, presence = [], []

    # Pose
    if res.pose_landmarks:
        pose = _landmarks_to_np(res.pose_landmarks.landmark)
        vecs.append(pose); presence.append(1.0)
    else:
        vecs.append(np.zeros(33*4, np.float32)); presence.append(0.0)

    # Mano izq
    if res.left_hand_landmarks:
        lhand = _landmarks_to_np(res.left_hand_landmarks.landmark)
        presence.append(1.0)
    else:
        lhand = np.zeros(21*4, np.float32); presence.append(0.0)
    vecs.append(lhand)

    # Mano der
    if res.right_hand_landmarks:
        rhand = _landmarks_to_np(res.right_hand_landmarks.landmark)
        presence.append(1.0)
    else:
        rhand = np.zeros(21*4, np.float32); presence.append(0.0)
    vecs.append(rhand)

    base = np.concatenate(vecs, axis=0)

    # Normalización por centro de caderas y escala hombro-hombro
    if res.pose_landmarks:
        try:
            l_sh = POSE_LM.LEFT_SHOULDER.value
            r_sh = POSE_LM.RIGHT_SHOULDER.value
            l_hp = POSE_LM.LEFT_HIP.value
            r_hp = POSE_LM.RIGHT_HIP.value
            lm = res.pose_landmarks.landmark
            cx = (lm[l_hp].x + lm[r_hp].x)/2; cy = (lm[l_hp].y + lm[r_hp].y)/2
            dist = ((lm[l_sh].x - lm[r_sh].x)**2 + (lm[l_sh].y - lm[r_sh].y)**2)**0.5
            scale = dist if dist > 1e-6 else 1.0
        except Exception:
            cx, cy, scale = 0.5, 0.5, 1.0
    else:
        cx, cy, scale = 0.5, 0.5, 1.0

    norm = base.copy()
    for i in range(0, len(norm), 4):
        norm[i]   = (norm[i]   - cx) / scale
        norm[i+1] = (norm[i+1] - cy) / scale
        norm[i+2] =  norm[i+2] / scale

    if ADD_PRESENCE_FLAGS:
        norm = np.concatenate([norm, np.array(presence, dtype=np.float32)], axis=0)

    return norm

def run_holistic(img_bgr, holistic):
    """
    Ejecuta holistic sobre una imagen BGR usando una instancia ya creada y
    devuelve (features_vector, results).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = holistic.process(img_rgb)
    vec = _extract_raw(res)
    return vec, res

# ---------- helpers de dibujo ----------
def draw_holistic(img_bgr, res):
    """
    Dibuja pose + manos sobre una copia de la imagen.
    Cara NO se dibuja.
    """
    out = img_bgr.copy()

    # Estilos de dibujo
    c_land = mp_draw.DrawingSpec(thickness=2, circle_radius=2)
    c_conn = mp_draw.DrawingSpec(thickness=2, circle_radius=2)

    # Pose
    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            image=out,
            landmark_list=res.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=c_land,
            connection_drawing_spec=c_conn
        )

    # Mano izquierda
    if res.left_hand_landmarks:
        mp_draw.draw_landmarks(
            image=out,
            landmark_list=res.left_hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=c_land,
            connection_drawing_spec=c_conn
        )

    # Mano derecha
    if res.right_hand_landmarks:
        mp_draw.draw_landmarks(
            image=out,
            landmark_list=res.right_hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=c_land,
            connection_drawing_spec=c_conn
        )

    return out

# Mantengo compatibilidad con el resto del proyecto
def extract_features_bgr(img_bgr, holistic=None):
    if holistic is None:
        # Uso temporal (menos eficiente): crear instancia ad-hoc
        with mp_holistic.Holistic(static_image_mode=True, model_complexity=1,
                                  smooth_landmarks=False, enable_segmentation=False,
                                  refine_face_landmarks=False) as h:
            res = h.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return _extract_raw(res)
    else:
        vec, _ = run_holistic(img_bgr, holistic)
        return vec
