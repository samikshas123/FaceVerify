"""
Face embedding generation module.

Extracts numerical face embeddings from detected faces.
"""

import numpy as np


class FaceEmbedder:
    """Generate normalized face embeddings."""

    @staticmethod
    def generate_embedding(face_object) -> np.ndarray:
        """
        Extract embedding from an InsightFace face object.

        Args:
            face_object: Face object returned by InsightFace.

        Returns:
            Normalized face embedding.
        """

        embedding = face_object.embedding

        if embedding is None:
            raise ValueError(
                "Face embedding could not be generated."
            )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError(
                "Invalid zero-length face embedding."
            )

        return embedding / norm

    @staticmethod
    def generate_from_face(face_object) -> np.ndarray:
        """
        Generate an embedding from a detected face.
        """

        return FaceEmbedder.generate_embedding(face_object)