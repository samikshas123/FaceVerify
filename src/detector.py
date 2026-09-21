"""
Face detection module for FaceVerify.

Uses InsightFace to detect faces and extract face regions
from input images.
"""

from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """Detect faces using the InsightFace detection model."""

    def __init__(self, det_size=(640, 640)):
        self.det_size = det_size

        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.model.prepare(
            ctx_id=0,
            det_size=det_size
        )

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image.

        Returns:
            Image as a NumPy array.

        Raises:
            FileNotFoundError: If the image does not exist.
            ValueError: If OpenCV cannot read the image.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all faces in an image.

        Args:
            image: OpenCV BGR image.

        Returns:
            List containing information about detected faces.
        """

        faces = self.model.get(image)

        detected_faces = []

        for index, face in enumerate(faces):

            bbox = face.bbox.astype(int)

            detected_faces.append(
                {
                    "face_id": index + 1,
                    "bbox": bbox.tolist(),
                    "face_object": face
                }
            )

        return detected_faces

    def detect_from_file(
        self,
        image_path: str
    ) -> List[Dict[str, Any]]:
        """
        Detect faces directly from an image file.
        """

        image = self.load_image(image_path)

        return self.detect(image)