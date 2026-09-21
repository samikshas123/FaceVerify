"""
Similarity-based face matching module.

Compares face embeddings using cosine similarity and
rejects low-confidence matches as UNKNOWN.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceMatcher:
    """Match query embeddings against enrolled embeddings."""

    def __init__(self, threshold: float = 0.50):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold

    def calculate_similarity(
        self,
        query_embedding: np.ndarray,
        database_embedding: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        database_embedding = np.asarray(
            database_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        similarity = cosine_similarity(
            query_embedding,
            database_embedding
        )[0][0]

        return float(similarity)

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        enrolled_embeddings: Dict[str, List[np.ndarray]]
    ) -> Tuple[str, float, bool]:
        """
        Find the closest enrolled person.

        Args:
            query_embedding: Embedding of the query face.
            enrolled_embeddings:
                Dictionary in the form:
                {
                    "Person Name": [embedding1, embedding2, ...]
                }

        Returns:
            Tuple containing:
                predicted_name,
                similarity_score,
                is_match
        """

        if not enrolled_embeddings:
            return "Unknown", 0.0, False

        best_name: Optional[str] = None
        best_similarity = -1.0

        for person_name, embeddings in enrolled_embeddings.items():

            for database_embedding in embeddings:

                similarity = self.calculate_similarity(
                    query_embedding,
                    database_embedding
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = person_name

        if (
            best_name is not None
            and best_similarity >= self.threshold
        ):
            return (
                best_name,
                best_similarity,
                True
            )

        return (
            "Unknown",
            best_similarity,
            False
        )