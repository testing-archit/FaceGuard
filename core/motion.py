"""Constant-velocity Kalman filter for a face box centre.

Used only when a tracker has lost its face and coasting is on with
`coast_model = "kalman"` (DECISIONS D-030).

Concept: the state is x = [cx, cy, vx, vy]. Each frame,
- predict:  x <- F x,  P <- F P F^T + Q    (F moves the centre by the velocity)
- update:   K = P H^T (H P H^T + R)^-1,  x <- x + K (z - H x),  P <- (I - K H) P
where z is the measured centre (the tracker box). Q models unknown
acceleration (white-noise acceleration model, std `q`), R the measurement
noise (std `r`). The Kalman gain K weighs the prediction against the
measurement by their uncertainties, so velocity estimates are smoothed
without a hand-tuned decay. While no measurements arrive, P grows every
predict step: the position standard deviation sqrt(P) tells how far the
face may have moved, which sizes the lost mask.
"""

from __future__ import annotations

import numpy as np


class KalmanCV:
    def __init__(self, cx: float, cy: float, q: float, r: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([r * r, r * r, 100.0, 100.0])  # velocity unknown at birth
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        # Discrete white-noise acceleration, dt = 1 frame.
        g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        self.Q = g @ g.T * q * q
        self.R = np.eye(2) * r * r

    def predict(self) -> tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float) -> None:
        y = np.array([cx, cy]) - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4) - k @ self.H) @ self.P

    @property
    def position_std(self) -> float:
        """Largest position standard deviation, px."""
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
