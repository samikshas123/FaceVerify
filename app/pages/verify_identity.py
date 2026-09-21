import streamlit as st
import cv2
import numpy as np
from pathlib import Path

from insightface.app import FaceAnalysis

from src.liveness import LivenessDetector
from src.password_security import verify_password
from src.candidate_database import CandidateDatabase


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Verify Identity",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DB = (
    PROJECT_ROOT
    / "data"
    / "candidates"
    / "candidates.json"
)

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "embeddings"
)


# ============================================================
# DATABASE
# ============================================================

candidate_database = CandidateDatabase(
    database_path=str(CANDIDATE_DB)
)


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

@st.cache_resource
def load_face_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return model


face_model = load_face_model()


# ============================================================
# SESSION STATE
# ============================================================

if "verification_started" not in st.session_state:
    st.session_state.verification_started = False

if "verification_result" not in st.session_state:
    st.session_state.verification_result = None

if "verification_message" not in st.session_state:
    st.session_state.verification_message = ""

if "verification_similarity" not in st.session_state:
    st.session_state.verification_similarity = None

if "liveness_detector" not in st.session_state:
    st.session_state.liveness_detector = LivenessDetector(
        minimum_movement=20.0,
        required_frames=8
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .verify-title {
        font-size: 38px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .verify-subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        background: white;
        border: 1px solid #e4e7ec;
        margin-top: 20px;
        text-align: center;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

    .result-description {
        color: #667085;
        font-size: 16px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="verify-title">🔍 Verify Identity</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="verify-subtitle">'
    'Live facial recognition and biometric identity verification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# CHECK LOGIN
# ============================================================

logged_in = st.session_state.get(
    "logged_in",
    False
)

logged_candidate_id = st.session_state.get(
    "candidate_id",
    None
)


if not logged_in or not logged_candidate_id:

    st.warning(
        "Please log in before starting face verification."
    )

    st.stop()


# ============================================================
# GET LOGGED-IN CANDIDATE
# ============================================================

candidate = candidate_database.get_candidate(
    logged_candidate_id
)

if candidate is None:

    st.error(
        "Logged-in candidate could not be found."
    )

    st.stop()


# ============================================================
# CHECK REGISTERED EMBEDDING
# ============================================================

embedding_file = candidate.get(
    "embedding_file",
    ""
)

if not embedding_file:

    st.error(
        "No registered face embedding was found "
        "for this account."
    )

    st.stop()


embedding_path = Path(
    embedding_file
)

if not embedding_path.is_absolute():

    embedding_path = (
        PROJECT_ROOT
        / embedding_path
    )


if not embedding_path.exists():

    st.error(
        "Registered face embedding file does not exist."
    )

    st.stop()


try:

    registered_embedding = np.load(
        embedding_path
    ).astype(np.float32)

except Exception as error:

    st.error(
        f"Unable to load registered face embedding: {error}"
    )

    st.stop()


# ============================================================
# NORMALIZE REGISTERED EMBEDDING
# ============================================================

registered_norm = np.linalg.norm(
    registered_embedding
)

if registered_norm == 0:

    st.error(
        "Registered face embedding is invalid."
    )

    st.stop()

registered_embedding = (
    registered_embedding
    / registered_norm
)


# ============================================================
# USER INFORMATION
# ============================================================

st.subheader(
    "Identity Verification"
)

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write(
        f"**Name:** {candidate.get('name', 'Unknown')}"
    )

with info_col2:

    st.write(
        f"**Candidate ID:** "
        f"{candidate.get('candidate_id', 'Unknown')}"
    )


st.divider()


# ============================================================
# VERIFICATION INSTRUCTIONS
# ============================================================

st.info(
    "Look directly at the camera. "
    "Move your face slightly from side to side so that "
    "the system can verify liveness."
)


# ============================================================
# CAMERA
# ============================================================

st.subheader(
    "Live Face Scan"
)

camera_image = st.camera_input(
    "Camera",
    key="verification_camera"
)


# ============================================================
# START SCAN
# ============================================================

if camera_image is not None:

    if st.button(
        "🔍 Scan My Face",
        use_container_width=True,
        type="primary"
    ):

        # ----------------------------------------------------
        # READ CAMERA IMAGE
        # ----------------------------------------------------

        image_bytes = camera_image.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            st.error(
                "Unable to read the camera image."
            )

            st.stop()

        # ----------------------------------------------------
        # DETECT FACE
        # ----------------------------------------------------

        try:

            faces = face_model.get(
                frame
            )

        except Exception as error:

            st.error(
                f"Face detection failed: {error}"
            )

            st.stop()

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if len(faces) == 0:

            st.session_state.verification_result = (
                "NO FACE DETECTED"
            )

            st.session_state.verification_message = (
                "No face was detected. "
                "Please position your face clearly "
                "in front of the camera."
            )

            st.session_state.verification_similarity = None

        # ----------------------------------------------------
        # MULTIPLE FACES
        # ----------------------------------------------------

        elif len(faces) > 1:

            st.session_state.verification_result = (
                "MULTIPLE FACES DETECTED"
            )

            st.session_state.verification_message = (
                "More than one face was detected. "
                "Only one person should be visible."
            )

            st.session_state.verification_similarity = None

        # ----------------------------------------------------
        # ONE FACE
        # ----------------------------------------------------

        else:

            face = faces[0]

            # ------------------------------------------------
            # LIVENESS
            # ------------------------------------------------

            bbox = face.bbox.astype(int)

            st.session_state.liveness_detector.add_face_center(
                bbox
            )

            liveness_status = (
                st.session_state
                .liveness_detector
                .status()
            )

            # ------------------------------------------------
            # CHECK LIVENESS
            # ------------------------------------------------

            if not liveness_status["live"]:

                st.session_state.verification_result = (
                    "LIVENESS NOT VERIFIED"
                )

                st.session_state.verification_message = (
                    "Please move your face slightly and "
                    "scan again."
                )

                st.session_state.verification_similarity = None

            else:

                # ------------------------------------------------
                # GET EMBEDDING
                # ------------------------------------------------

                if face.embedding is None:

                    st.session_state.verification_result = (
                        "VERIFICATION ERROR"
                    )

                    st.session_state.verification_message = (
                        "Unable to generate a face embedding."
                    )

                    st.session_state.verification_similarity = None

                else:

                    live_embedding = np.asarray(
                        face.embedding,
                        dtype=np.float32
                    ).reshape(-1)

                    # --------------------------------------------
                    # NORMALIZE
                    # --------------------------------------------

                    live_norm = np.linalg.norm(
                        live_embedding
                    )

                    if live_norm == 0:

                        st.session_state.verification_result = (
                            "VERIFICATION ERROR"
                        )

                        st.session_state.verification_message = (
                            "Invalid face embedding."
                        )

                        st.session_state.verification_similarity = None

                    else:

                        live_embedding = (
                            live_embedding
                            / live_norm
                        )

                        # ----------------------------------------
                        # COSINE SIMILARITY
                        # ----------------------------------------

                        similarity = float(
                            np.dot(
                                registered_embedding,
                                live_embedding
                            )
                        )

                        st.session_state.verification_similarity = (
                            similarity
                        )

                        # ----------------------------------------
                        # MATCH THRESHOLD
                        # ----------------------------------------

                        MATCH_THRESHOLD = 0.50

                        if similarity >= MATCH_THRESHOLD:

                            st.session_state.verification_result = (
                                "SAME FACE"
                            )

                            st.session_state.verification_message = (
                                "Face matched with the registered "
                                "identity."
                            )

                        else:

                            st.session_state.verification_result = (
                                "UNKNOWN"
                            )

                            st.session_state.verification_message = (
                                "The scanned face does not match "
                                "the registered identity."
                            )


# ============================================================
# SHOW LIVENESS STATUS
# ============================================================

liveness_status = (
    st.session_state
    .liveness_detector
    .status()
)

st.write("")

st.subheader(
    "Liveness Status"
)

live_col1, live_col2 = st.columns(2)

with live_col1:

    st.metric(
        "Frames",
        liveness_status["frames"]
    )

with live_col2:

    st.metric(
        "Movement",
        f"{liveness_status['movement']:.2f}"
    )


# ============================================================
# SHOW VERIFICATION RESULT
# ============================================================

result = st.session_state.verification_result

if result:

    st.divider()

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-title">{result}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-description">'
        f'{st.session_state.verification_message}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    similarity = (
        st.session_state.verification_similarity
    )

    if similarity is not None:

        st.write("")

        st.metric(
            "Face Similarity",
            f"{similarity:.4f}"
        )

        st.caption(
            "Matching threshold: 0.50"
        )


# ============================================================
# RESET VERIFICATION
# ============================================================

st.write("")

if st.button(
    "🔄 Start New Scan",
    use_container_width=True
):

    st.session_state.verification_result = None

    st.session_state.verification_message = ""

    st.session_state.verification_similarity = None

    st.session_state.liveness_detector.reset()

    st.rerun()