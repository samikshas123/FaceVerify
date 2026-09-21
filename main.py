"""
FaceVerify - Face Recognition & Identification System

Main application entry point for enrollment and evaluation.
"""

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.database import EmbeddingDatabase
from src.enrollment import FaceEnrollment
from src.matcher import FaceMatcher
from src.recognition import FaceRecognition
from src.evaluation import FaceRecognitionEvaluator


# Configuration
THRESHOLD = 0.50

ENROLLED_DIRECTORY = "data/enrolled"
TEST_DIRECTORY = "data/test"

DATABASE_PATH = "embeddings/face_embeddings.pkl"

RESULTS_PATH = "results/evaluation_results.csv"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"
SIMILARITY_PATH = "results/similarity_distribution.png"


def build_system():
    """Create all FaceVerify components."""

    print("Loading face detection model...")

    detector = FaceDetector()

    embedder = FaceEmbedder()

    database = EmbeddingDatabase(
        DATABASE_PATH
    )

    matcher = FaceMatcher(
        threshold=THRESHOLD
    )

    enrollment = FaceEnrollment(
        detector=detector,
        embedder=embedder,
        database=database
    )

    recognition = FaceRecognition(
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        database=database
    )

    return (
        enrollment,
        recognition,
        database
    )


def enroll_people(
    enrollment,
    database
):
    """Enroll all people from the enrolled directory."""

    import os

    enrolled_path = os.path.abspath(
        ENROLLED_DIRECTORY
    )

    if not os.path.exists(enrolled_path):
        raise FileNotFoundError(
            f"Enrollment directory not found: "
            f"{enrolled_path}"
        )

    print("\n========== ENROLLMENT ==========")

    for person_directory in sorted(
        os.listdir(enrolled_path)
    ):

        person_path = os.path.join(
            enrolled_path,
            person_directory
        )

        if not os.path.isdir(person_path):
            continue

        print(
            f"\nProcessing: {person_directory}"
        )

        try:

            count = enrollment.enroll_from_directory(
                person_name=person_directory,
                image_directory=person_path
            )

            print(
                f"Enrolled {count} image(s) "
                f"for {person_directory}."
            )

        except Exception as error:

            print(
                f"Could not enroll "
                f"{person_directory}: {error}"
            )

    print("\nEnrolled people:")

    for person in database.list_people():
        print(
            f"  - {person} "
            f"({database.count_embeddings(person)} "
            f"embeddings)"
        )


def run_evaluation(
    recognition
):
    """Run the evaluation pipeline."""

    print("\n========== EVALUATION ==========")

    evaluator = FaceRecognitionEvaluator(
        recognition
    )

    results = evaluator.evaluate(
        TEST_DIRECTORY
    )

    metrics = evaluator.calculate_metrics(
        results
    )

    evaluator.save_results(
        results,
        RESULTS_PATH
    )

    evaluator.create_confusion_matrix(
        results,
        CONFUSION_MATRIX_PATH
    )

    evaluator.create_similarity_distribution(
        results,
        SIMILARITY_PATH
    )

    print("\nEvaluation Results")
    print("----------------------------")

    for metric, value in metrics.items():

        print(
            f"{metric.replace('_', ' ').title()}: "
            f"{value * 100:.2f}%"
        )

    print(
        f"\nDetailed results: {RESULTS_PATH}"
    )

    print(
        f"Confusion matrix: "
        f"{CONFUSION_MATRIX_PATH}"
    )

    print(
        f"Similarity distribution: "
        f"{SIMILARITY_PATH}"
    )

    print("\nDetailed predictions:")
    print(results.to_string(index=False))


def main():
    """Main execution function."""

    print("=" * 60)
    print("FACEVERIFY")
    print("Face Recognition & Identification System")
    print("=" * 60)

    print(
        f"\nMatching threshold: {THRESHOLD}"
    )

    enrollment, recognition, database = (
        build_system()
    )

    enroll_people(
        enrollment,
        database
    )

    run_evaluation(
        recognition
    )


if __name__ == "__main__":
    main()