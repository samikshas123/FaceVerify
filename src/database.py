from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np


class EmbeddingDatabase:
    """Manage the local face embedding database."""

    def __init__(
        self,
        database_path: str = "embeddings/face_embeddings.pkl"
    ):
        self.database_path = Path(database_path)

        # Create the parent directory if it doesn't exist.
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self) -> Dict[str, List[np.ndarray]]:
        """
        Load embeddings from the database.

        Returns:
            Dictionary containing person names and embeddings.
        """

        if not self.database_path.exists():
            return {}

        try:
            with open(
                self.database_path,
                "rb"
            ) as file:
                database = pickle.load(file)

            return database

        except (pickle.PickleError, EOFError):
            raise ValueError(
                "The embedding database is corrupted."
            )

    def save(
        self,
        database: Dict[str, List[np.ndarray]]
    ) -> None:
        """
        Save embeddings to the database.
        """

        with open(
            self.database_path,
            "wb"
        ) as file:
            pickle.dump(
                database,
                file
            )

    def add_person(
        self,
        person_name: str,
        embeddings: List[np.ndarray]
    ) -> None:
        """
        Add a person and their embeddings.

        If the person already exists, the new embeddings
        are added to the existing embeddings.
        """

        database = self.load()

        if person_name not in database:
            database[person_name] = []

        database[person_name].extend(embeddings)

        self.save(database)

    def remove_person(
        self,
        person_name: str
    ) -> bool:
        """
        Remove a person from the database.

        Returns:
            True if the person existed and was removed.
        """

        database = self.load()

        if person_name not in database:
            return False

        del database[person_name]

        self.save(database)

        return True

    def list_people(self) -> List[str]:
        """
        Return the names of all enrolled people.
        """

        database = self.load()

        return sorted(database.keys())

    def count_embeddings(
        self,
        person_name: str
    ) -> int:
        """
        Return the number of embeddings stored for a person.
        """

        database = self.load()

        return len(
            database.get(person_name, [])
        )