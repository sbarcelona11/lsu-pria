from __future__ import annotations

import numpy as np


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + 1e-6
    nbc = np.linalg.norm(bc) + 1e-6
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.arccos(cosang))


def extract_landmark_features(landmarks_norm: np.ndarray, handedness: str | None) -> np.ndarray:
    """
    landmarks_norm: (21,3) in normalized image coords (x,y,z).
    Returns a fixed-length feature vector with translation/scale normalization.
    """
    lm = landmarks_norm.astype(np.float32).copy()

    # Mirror right hand to reduce handedness variance (optional but helpful).
    # MediaPipe uses image coords; mirroring x for "Right" makes geometry closer.
    if handedness and handedness.lower().startswith("right"):
        lm[:, 0] = 1.0 - lm[:, 0]

    wrist = lm[0, :2]
    pts = lm[:, :2] - wrist[None, :]

    # Scale by wrist->middle_mcp distance (landmark 9).
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale

    # Pairwise distances to wrist for fingertips and MCPs.
    idxs = [4, 8, 12, 16, 20, 5, 9, 13, 17]
    dists = [np.linalg.norm(pts[i]) for i in idxs]

    # Finger joint angles (MCP-PIP-DIP and PIP-DIP-TIP for index/middle/ring/pinky)
    angles = []
    finger_chains = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
        "thumb": (1, 2, 3, 4),
    }
    for a, b, c, d in finger_chains.values():
        angles.append(_angle(pts[a], pts[b], pts[c]))
        angles.append(_angle(pts[b], pts[c], pts[d]))

    # Relative distances between fingertip pairs (shape cue)
    tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 12), (4, 20)]
    rel = [np.linalg.norm(pts[i] - pts[j]) for i, j in tip_pairs]

    feat = np.array(dists + angles + rel, dtype=np.float32)
    return feat


def _flatten_selected(points: np.ndarray, center: np.ndarray, scale: float, mirror_x: bool = False) -> np.ndarray:
    pts = points.astype(np.float32).copy()
    xy = pts[:, :2] - center[None, :]
    xy = xy / (scale + 1e-6)
    if mirror_x:
        xy[:, 0] *= -1.0
    z = pts[:, 2:3] / (scale + 1e-6)
    return np.concatenate([xy, z], axis=1).reshape(-1).astype(np.float32)


def _zero_like(size: int) -> np.ndarray:
    return np.zeros((size,), dtype=np.float32)


def extract_multimodal_frame_features(
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    pose: np.ndarray | None,
    face: np.ndarray | None,
) -> np.ndarray:
    shoulder_center = np.array([0.5, 0.5], dtype=np.float32)
    pose_scale = 1.0
    if pose is not None and pose.shape[0] >= 5:
        left_shoulder = pose[3, :2]
        right_shoulder = pose[4, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        pose_scale = float(np.linalg.norm(left_shoulder - right_shoulder) + 1e-6)

    face_center = face[0, :2] if face is not None and face.shape[0] > 0 else shoulder_center

    pose_feat = _flatten_selected(pose, shoulder_center, pose_scale) if pose is not None else _zero_like(11 * 3)
    face_feat = _flatten_selected(face, face_center, pose_scale) if face is not None else _zero_like(6 * 3)

    if left_hand is not None:
        left_feat = extract_landmark_features(left_hand, handedness=None)
    else:
        left_feat = _zero_like(25)

    if right_hand is not None:
        right_feat = extract_landmark_features(right_hand, handedness="Right")
    else:
        right_feat = _zero_like(25)

    presence = np.array(
        [
            1.0 if left_hand is not None else 0.0,
            1.0 if right_hand is not None else 0.0,
            1.0 if pose is not None else 0.0,
            1.0 if face is not None else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([left_feat, right_feat, pose_feat, face_feat, presence], axis=0).astype(np.float32)
