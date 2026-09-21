"""
Face enrollment module for FaceVerify.

Processes images of a person and stores their face
embeddings in the local database.
"""

from pathlib import Path
from typing import List

import numpy as np

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.database import EmbeddingDatabase


class FaceEnrollment:
    """Enroll people into the face recognition database."""

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        database: EmbeddingDatabase
    ):
        self.detector = detector
        self.embedder = embedder
        self.database = database

    def process_image(
        self,
        image_path: str
    ) -> np.ndarray:
        """
        Process one image and generate a face embedding.

        Exactly one face must be present in the image.
        """

        image = self.detector.load_image(image_path)

        faces = self.detector.detect(image)

        if len(faces) == 0:
            raise ValueError(
                f"No face detected in: {image_path}"
            )

        if len(faces) > 1:
            raise ValueError(
                f"Multiple faces detected in: {image_path}. "
                "Enrollment images must contain exactly one face."
            )

        face_object = faces[0]["face_object"]

        embedding = self.embedder.generate_from_face(
            face_object
        )

        return embedding

    def enroll_from_directory(
        self,
        person_name: str,
        image_directory: str
    ) -> int:
        """
        Enroll a person using all supported images
        inside a directory.

        Returns:
            Number of successfully processed images.
        """

        directory = Path(image_directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {image_directory}"
            )

        image_files = [
            path
            for path in directory.iterdir()
            if path.suffix.lower()
            in self.SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            raise ValueError(
                "No supported images found in the directory."
            )

        embeddings: List[np.ndarray] = []
        failed_images = []

        for image_path in sorted(image_files):

            try:
                embedding = self.process_image(
                    str(image_path)
                )

                embeddings.append(embedding)

            except (ValueError, FileNotFoundError) as error:

                failed_images.append(
                    f"{image_path.name}: {error}"
                )

        if not embeddings:
            raise ValueError(
                "No valid face embeddings could be generated."
            )

        self.database.add_person(
            person_name,
            embeddings
        )

        print(
            f"Successfully enrolled {person_name} "
            f"with {len(embeddings)} image(s)."
        )

        if failed_images:
            print("\nImages skipped:")

            for error in failed_images:
                print(f"  - {error}")

        return len(embeddings)