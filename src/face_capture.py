"""
FaceVerify - Face Capture Utilities

Handles saving captured face images and
generating embeddings for registered candidates.
"""

from pathlib import Path

import cv2
import numpy as np


class FaceCaptureManager:
    """Manage candidate face images."""

    def __init__(
        self,
        base_directory="data/face_images"
    ):
        self.base_directory = Path(
            base_directory
        )

        self.base_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def candidate_directory(
        self,
        candidate_id
    ):
        """Return candidate image directory."""

        directory = (
            self.base_directory /
            candidate_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return directory

    def save_image(
        self,
        candidate_id,
        image,
        image_number
    ):
        """
        Save one face image.

        Returns saved file path.
        """

        directory = self.candidate_directory(
            candidate_id
        )

        filename = (
            f"face_{image_number:02d}.jpg"
        )

        file_path = directory / filename

        success = cv2.imwrite(
            str(file_path),
            image
        )

        if not success:
            raise IOError(
                f"Unable to save image: {file_path}"
            )

        return file_path

    def get_images(
        self,
        candidate_id
    ):
        """Return all saved candidate images."""

        directory = self.candidate_directory(
            candidate_id
        )

        return sorted(
            directory.glob("*.jpg")
        )