import json
from pathlib import Path
from datetime import datetime


class CandidateDatabase:

    def __init__(
        self,
        database_path="data/candidates/candidates.json"
    ):
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.database_path.exists():

            self._save([])


    # ========================================================
    # LOAD DATABASE
    # ========================================================

    def _load(self):

        try:

            with open(
                self.database_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if isinstance(data, list):
                    return data

                return []

        except (
            json.JSONDecodeError,
            FileNotFoundError
        ):

            return []


    # ========================================================
    # SAVE DATABASE
    # ========================================================

    def _save(self, candidates):

        with open(
            self.database_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                candidates,
                file,
                indent=4
            )


    # ========================================================
    # GENERATE CANDIDATE ID
    # ========================================================

    def generate_candidate_id(self):

        candidates = self._load()

        highest_number = 0

        for candidate in candidates:

            candidate_id = candidate.get(
                "candidate_id",
                ""
            )

            if candidate_id.startswith("FV"):

                try:

                    number = int(
                        candidate_id[2:]
                    )

                    highest_number = max(
                        highest_number,
                        number
                    )

                except ValueError:

                    pass

        return f"FV{highest_number + 1:04d}"


    # ========================================================
    # ADD CANDIDATE
    # ========================================================

    def add_candidate(
        self,
        name,
        email,
        mobile,
        date_of_birth="",
        gender="",
        password_salt="",
        password_hash=""
    ):

        candidates = self._load()

        # ----------------------------------------------------
        # CHECK EMAIL
        # ----------------------------------------------------

        for candidate in candidates:

            existing_email = candidate.get(
                "email",
                ""
            )

            if (
                existing_email.lower()
                == email.strip().lower()
            ):

                raise ValueError(
                    "A candidate with this email already exists."
                )

        # ----------------------------------------------------
        # CREATE ID
        # ----------------------------------------------------

        candidate_id = (
            self.generate_candidate_id()
        )

        # ----------------------------------------------------
        # CREATE CANDIDATE
        # ----------------------------------------------------

        candidate = {

            "candidate_id":
                candidate_id,

            "name":
                name.strip(),

            "email":
                email.strip(),

            "mobile":
                mobile.strip(),

            "date_of_birth":
                date_of_birth,

            "gender":
                gender,

            "registered_at":
                datetime.now().isoformat(),

            # -----------------------------------------------
            # PASSWORD SECURITY
            # -----------------------------------------------

            "password_salt":
                password_salt,

            "password_hash":
                password_hash,

            # -----------------------------------------------
            # FACE DATA
            # -----------------------------------------------

            "face_images":
                [],

            "embedding_count":
                0,

            "embedding_file":
                "",

            "registration_status":
                "pending",

            "liveness_verified":
                False
        }

        candidates.append(
            candidate
        )

        self._save(
            candidates
        )

        return candidate


    # ========================================================
    # GET ALL CANDIDATES
    # ========================================================

    def get_all_candidates(self):

        return self._load()


    # ========================================================
    # GET ONE CANDIDATE
    # ========================================================

    def get_candidate(
        self,
        candidate_id
    ):

        candidates = self._load()

        for candidate in candidates:

            if (
                candidate.get("candidate_id")
                == candidate_id
            ):

                return candidate

        return None


    # ========================================================
    # FIND BY EMAIL
    # ========================================================

    def get_candidate_by_email(
        self,
        email
    ):

        candidates = self._load()

        email = email.strip().lower()

        for candidate in candidates:

            candidate_email = candidate.get(
                "email",
                ""
            ).strip().lower()

            if candidate_email == email:

                return candidate

        return None


    # ========================================================
    # UPDATE CANDIDATE
    # ========================================================

    def update_candidate(
        self,
        candidate_id,
        updates
    ):

        candidates = self._load()

        for candidate in candidates:

            if (
                candidate.get("candidate_id")
                == candidate_id
            ):

                candidate.update(
                    updates
                )

                self._save(
                    candidates
                )

                return True

        return False