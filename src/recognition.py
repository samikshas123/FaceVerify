"""
Face recognition module for FaceVerify.

Identifies faces by comparing their embeddings against
the enrolled embedding database.
"""

from typing import Any, Dict, List

import numpy as np

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.matcher import FaceMatcher
from src.database import EmbeddingDatabase


class FaceRecognition:
    """Recognize faces using the enrolled database."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        matcher: FaceMatcher,
        database: EmbeddingDatabase
    ):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.database = database

    def recognize_image(
        self,
        image_path: str
    ) -> List[Dict[str, Any]]:
        """
        Recognize all faces in an image.

        Returns:
            List of recognition results.
        """

        image = self.detector.load_image(
            image_path
        )

        faces = self.detector.detect(
            image
        )

        database = self.database.load()

        results = []

        for face in faces:

            embedding = self.embedder.generate_from_face(
                face["face_object"]
            )

            name, similarity, is_match = (
                self.matcher.find_best_match(
                    embedding,
                    database
                )
            )

            results.append(
                {
                    "face_id": face["face_id"],
                    "bbox": face["bbox"],
                    "name": name,
                    "similarity": similarity,
                    "is_match": is_match
                }
            )

        return results

    def recognize_embedding(
        self,
        embedding: np.ndarray
    ) -> Dict[str, Any]:
        """
        Recognize an already-generated embedding.
        """

        database = self.database.load()

        name, similarity, is_match = (
            self.matcher.find_best_match(
                embedding,
                database
            )
        )

        return {
            "name": name,
            "similarity": similarity,
            "is_match": is_match
        }