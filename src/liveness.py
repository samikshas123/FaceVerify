"""
FaceVerify - Liveness Detection

Basic challenge-based liveness verification.

The user is asked to move their face while the
camera is running. The system checks whether
the detected face position changes across
multiple live frames.
"""

import numpy as np


class LivenessDetector:
    """
    Basic motion-based liveness detector.

    This is intended for an internship/demo project.
    It should not be considered a production-grade
    anti-spoofing system.
    """

    def __init__(
        self,
        minimum_movement=20.0,
        required_frames=8
    ):
        self.minimum_movement = minimum_movement
        self.required_frames = required_frames

        self.face_centers = []

    def reset(self):
        """Reset the liveness detector."""

        self.face_centers = []

    def add_face_center(self, bbox):
        """
        Add the center point of a detected face.

        bbox format:
        [x1, y1, x2, y2]
        """

        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        self.face_centers.append(
            (center_x, center_y)
        )

        # Keep recent points only
        if len(self.face_centers) > 50:
            self.face_centers.pop(0)

        return True

    def movement_distance(self):
        """
        Calculate total horizontal movement.
        """

        if len(self.face_centers) < 2:
            return 0.0

        x_positions = [
            point[0]
            for point in self.face_centers
        ]

        return max(x_positions) - min(x_positions)

    def is_live(self):
        """
        Determine whether sufficient movement
        has been detected.
        """

        if len(self.face_centers) < self.required_frames:
            return False

        movement = self.movement_distance()

        return movement >= self.minimum_movement

    def status(self):
        """
        Return current liveness status.
        """

        movement = self.movement_distance()

        return {
            "frames": len(self.face_centers),
            "movement": round(movement, 2),
            "live": self.is_live()
        }