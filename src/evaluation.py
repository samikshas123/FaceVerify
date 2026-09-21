"""
Evaluation module for FaceVerify.

Evaluates face recognition performance on a labelled test dataset.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


class FaceRecognitionEvaluator:
    """Evaluate the face recognition system."""

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    def __init__(self, recognizer):
        self.recognizer = recognizer

    def collect_test_images(
        self,
        test_directory: str
    ) -> List[Dict[str, str]]:
        """
        Collect labelled test images.

        Expected structure:

        test/
        ├── Samiksha/
        │   ├── test1.jpg
        │   └── test2.jpg
        │
        ├── Pooja/
        │   ├── test1.jpg
        │   └── test2.jpg
        │
        └── Unknown/
            └── unknown1.jpg

        The folder name is treated as the expected identity.
        """

        test_path = Path(test_directory)

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test directory not found: {test_directory}"
            )

        samples = []

        for person_directory in sorted(test_path.iterdir()):

            if not person_directory.is_dir():
                continue

            expected_name = person_directory.name

            for image_path in sorted(
                person_directory.iterdir()
            ):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in self.SUPPORTED_EXTENSIONS
                ):
                    samples.append(
                        {
                            "image_path": str(image_path),
                            "expected_name": expected_name
                        }
                    )

        if not samples:
            raise ValueError(
                "No test images were found."
            )

        return samples

    def evaluate(
        self,
        test_directory: str
    ) -> pd.DataFrame:
        """
        Run recognition on all test images.

        Returns:
            DataFrame containing predictions and scores.
        """

        samples = self.collect_test_images(
            test_directory
        )

        results = []

        for sample in samples:

            image_path = sample["image_path"]
            expected_name = sample["expected_name"]

            try:

                predictions = self.recognizer.recognize_image(
                    image_path
                )

                if not predictions:

                    results.append(
                        {
                            "image": Path(image_path).name,
                            "expected": expected_name,
                            "predicted": "No Face",
                            "similarity": 0.0,
                            "correct": False,
                            "status": "No Face Detected"
                        }
                    )

                    continue

                # Evaluation images should normally contain one face.
                prediction = predictions[0]

                predicted_name = prediction["name"]
                similarity = prediction["similarity"]

                correct = (
                    predicted_name.lower()
                    == expected_name.lower()
                )

                results.append(
                    {
                        "image": Path(image_path).name,
                        "expected": expected_name,
                        "predicted": predicted_name,
                        "similarity": round(
                            similarity,
                            4
                        ),
                        "correct": correct,
                        "status": (
                            "Match"
                            if prediction["is_match"]
                            else "Unknown"
                        )
                    }
                )

            except Exception as error:

                results.append(
                    {
                        "image": Path(image_path).name,
                        "expected": expected_name,
                        "predicted": "Error",
                        "similarity": 0.0,
                        "correct": False,
                        "status": str(error)
                    }
                )

        return pd.DataFrame(results)

    def calculate_metrics(
        self,
        results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        """

        if results.empty:
            return {}

        y_true = results["expected"]
        y_pred = results["predicted"]

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

    def save_results(
        self,
        results: pd.DataFrame,
        output_path: str
    ) -> None:
        """Save detailed evaluation results."""

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        results.to_csv(
            output,
            index=False
        )

    def create_confusion_matrix(
        self,
        results: pd.DataFrame,
        output_path: str
    ) -> None:
        """Create and save a confusion matrix."""

        if results.empty:
            return

        labels = sorted(
            set(results["expected"])
            | set(results["predicted"])
        )

        matrix = confusion_matrix(
            results["expected"],
            results["predicted"],
            labels=labels
        )

        plt.figure(
            figsize=(10, 8)
        )

        plt.imshow(
            matrix,
            interpolation="nearest"
        )

        plt.title(
            "Face Recognition Confusion Matrix"
        )

        plt.colorbar()

        plt.xticks(
            range(len(labels)),
            labels,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(labels)),
            labels
        )

        plt.xlabel("Predicted Identity")
        plt.ylabel("Actual Identity")

        plt.tight_layout()

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def create_similarity_distribution(
        self,
        results: pd.DataFrame,
        output_path: str
    ) -> None:
        """Create a similarity-score distribution plot."""

        if results.empty:
            return

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            results["similarity"],
            bins=10
        )

        plt.title(
            "Face Similarity Score Distribution"
        )

        plt.xlabel(
            "Cosine Similarity"
        )

        plt.ylabel(
            "Number of Images"
        )

        plt.tight_layout()

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()