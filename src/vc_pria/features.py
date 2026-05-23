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

