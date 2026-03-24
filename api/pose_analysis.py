"""
pose_analysis.py
Extracts pose landmarks, joint angles, and engagement scores using MediaPipe.
No custom ML models needed — Claude handles pose classification.
"""

import math
import numpy as np
import mediapipe as mp

# ── MediaPipe setup (singleton) ───────────────────────────────────────────────
_mp_pose = mp.solutions.pose
_detector = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
_drawing = mp.solutions.drawing_utils

# MediaPipe landmark indices
LM = {
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow": 13,     "right_elbow": 14,
    "left_wrist": 15,     "right_wrist": 16,
    "left_hip": 23,       "right_hip": 24,
    "left_knee": 25,      "right_knee": 26,
    "left_ankle": 27,     "right_ankle": 28,
    "left_heel": 29,      "right_heel": 30,
}


def _angle(a, b, c) -> float:
    """Angle at point b formed by a-b-c in degrees (0-180)."""
    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(math.degrees(radians))
    return round(360 - angle if angle > 180 else angle, 1)


def _pt(lm, idx):
    return (lm[idx].x, lm[idx].y)


def _visible(lm, idx, threshold=0.5):
    return lm[idx].visibility > threshold


def is_full_body_visible(landmarks) -> bool:
    required = [
        LM["left_shoulder"], LM["right_shoulder"],
        LM["left_hip"],      LM["right_hip"],
        LM["left_knee"],     LM["right_knee"],
        LM["left_ankle"],    LM["right_ankle"],
    ]
    return all(_visible(landmarks, i) for i in required)


def extract_angles(lm) -> dict:
    return {
        "left_shoulder":  _angle(_pt(lm, LM["left_elbow"]),    _pt(lm, LM["left_shoulder"]),  _pt(lm, LM["left_hip"])),
        "right_shoulder": _angle(_pt(lm, LM["right_elbow"]),   _pt(lm, LM["right_shoulder"]), _pt(lm, LM["right_hip"])),
        "left_elbow":     _angle(_pt(lm, LM["left_shoulder"]), _pt(lm, LM["left_elbow"]),     _pt(lm, LM["left_wrist"])),
        "right_elbow":    _angle(_pt(lm, LM["right_shoulder"]),_pt(lm, LM["right_elbow"]),    _pt(lm, LM["right_wrist"])),
        "left_hip":       _angle(_pt(lm, LM["left_shoulder"]), _pt(lm, LM["left_hip"]),       _pt(lm, LM["left_knee"])),
        "right_hip":      _angle(_pt(lm, LM["right_shoulder"]),_pt(lm, LM["right_hip"]),      _pt(lm, LM["right_knee"])),
        "left_knee":      _angle(_pt(lm, LM["left_hip"]),      _pt(lm, LM["left_knee"]),      _pt(lm, LM["left_ankle"])),
        "right_knee":     _angle(_pt(lm, LM["right_hip"]),     _pt(lm, LM["right_knee"]),     _pt(lm, LM["right_ankle"])),
    }


def extract_engagement(lm) -> dict:
    # Shoulder: arm raise away from body
    ls = _angle(_pt(lm, LM["left_elbow"]),  _pt(lm, LM["left_shoulder"]),  _pt(lm, LM["left_hip"]))
    rs = _angle(_pt(lm, LM["right_elbow"]), _pt(lm, LM["right_shoulder"]), _pt(lm, LM["right_hip"]))
    shoulder = int(min(100, ((ls + rs) / 2) * 100 / 180))

    # Core: hip flexion deviation from upright
    lh = _angle(_pt(lm, LM["left_shoulder"]),  _pt(lm, LM["left_hip"]),  _pt(lm, LM["left_knee"]))
    rh = _angle(_pt(lm, LM["right_shoulder"]), _pt(lm, LM["right_hip"]), _pt(lm, LM["right_knee"]))
    core = int(min(100, abs(180 - (lh + rh) / 2) * 100 / 90))

    # Legs: knee bend
    lk = _angle(_pt(lm, LM["left_hip"]),  _pt(lm, LM["left_knee"]),  _pt(lm, LM["left_ankle"]))
    rk = _angle(_pt(lm, LM["right_hip"]), _pt(lm, LM["right_knee"]), _pt(lm, LM["right_ankle"]))
    legs = int(min(100, abs(180 - (lk + rk) / 2) * 100 / 90))

    return {"shoulder": shoulder, "core": core, "legs": legs}


def compute_similarity(current_lm, reference_lm) -> float:
    distances = [
        math.sqrt((c.x - r.x) ** 2 + (c.y - r.y) ** 2)
        for c, r in zip(current_lm, reference_lm)
    ]
    mean_dist = float(np.mean(distances))
    return round(max(1.0, min(10.0, 10.0 * (1.0 - mean_dist))), 2)


def landmarks_to_list(landmarks) -> list:
    return [
        {
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),
            "visibility": round(lm.visibility, 3),
        }
        for lm in landmarks
    ]


def process_frame(frame_bgr) -> dict:
    """
    Run MediaPipe on a BGR frame.
    Returns dict with: detected, full_body, landmarks, angles, engagement, _raw
    """
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = _detector.process(rgb)

    if not result.pose_landmarks:
        return {
            "detected": False, "full_body": False,
            "landmarks": None, "angles": None,
            "engagement": None, "_raw": None,
        }

    lm = result.pose_landmarks.landmark
    full_body = is_full_body_visible(lm)

    return {
        "detected": True,
        "full_body": full_body,
        "landmarks": landmarks_to_list(lm),
        "angles": extract_angles(lm) if full_body else None,
        "engagement": extract_engagement(lm) if full_body else None,
        "_raw": lm,
    }
