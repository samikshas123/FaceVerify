import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from src.candidate_database import CandidateDatabase
from src.liveness import LivenessDetector


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FaceVerify - Verify Identity",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #f7f9fc;
    }

    .verify-header {
        padding: 10px 0 20px 0;
    }

    .verify-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .verify-subtitle {
        color: #667085;
        font-size: 16px;
    }

    .status-box {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        background: white;
        margin-bottom: 12px;
    }

    .status-title {
        font-size: 13px;
        color: #667085;
        margin-bottom: 5px;
    }

    .status-value {
        font-size: 21px;
        font-weight: 700;
    }

    .result-box {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        background: white;
        border: 1px solid #e4e7ec;
        margin-top: 20px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
    }

    .result-description {
        color: #667085;
        font-size: 15px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD INSIGHTFACE MODEL
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


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(embedding_1, embedding_2):

    embedding_1 = np.asarray(
        embedding_1,
        dtype=np.float32
    )

    embedding_2 = np.asarray(
        embedding_2,
        dtype=np.float32
    )

    norm_1 = np.linalg.norm(embedding_1)
    norm_2 = np.linalg.norm(embedding_2)

    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    return float(
        np.dot(embedding_1, embedding_2)
        / (norm_1 * norm_2)
    )


# ============================================================
# LIVE VIDEO PROCESSOR
# ============================================================

class VerificationProcessor(VideoProcessorBase):

    def __init__(
        self,
        face_model,
        registered_embedding
    ):

        self.face_model = face_model

        self.registered_embedding = (
            registered_embedding
        )

        self.liveness = LivenessDetector(
            minimum_movement=20.0,
            required_frames=8
        )

        self.lock = threading.Lock()

        self.status = "STARTING CAMERA"

        self.liveness_status = "WAITING"

        self.face_count = 0

        self.frames = 0

        self.movement = 0.0

        self.similarity = 0.0

        self.final_result = None

        self.processing_verification = False


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        with self.lock:

            self.liveness.reset()

            self.status = "WAITING FOR FACE"

            self.liveness_status = "WAITING"

            self.face_count = 0

            self.frames = 0

            self.movement = 0.0

            self.similarity = 0.0

            self.final_result = None

            self.processing_verification = False


    # --------------------------------------------------------
    # GET CURRENT STATE
    # --------------------------------------------------------

    def get_state(self):

        with self.lock:

            return {
                "status": self.status,
                "liveness_status": self.liveness_status,
                "face_count": self.face_count,
                "frames": self.frames,
                "movement": self.movement,
                "similarity": self.similarity,
                "final_result": self.final_result
            }


    # --------------------------------------------------------
    # PROCESS CAMERA FRAME
    # --------------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        display_image = image.copy()

        try:

            faces = self.face_model.get(
                image
            )

            face_count = len(faces)

            # ==================================================
            # NO FACE
            # ==================================================

            if face_count == 0:

                with self.lock:

                    self.face_count = 0

                    self.status = "NO FACE DETECTED"

                    self.liveness_status = "WAITING"

                    self.frames = 0

                    self.movement = 0.0

                    if self.final_result is None:
                        self.liveness.reset()

                cv2.putText(
                    display_image,
                    "NO FACE DETECTED",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

                return av.VideoFrame.from_ndarray(
                    display_image,
                    format="bgr24"
                )


            # ==================================================
            # MULTIPLE FACES
            # ==================================================

            if face_count > 1:

                with self.lock:

                    self.face_count = face_count

                    self.status = (
                        "MULTIPLE FACES DETECTED"
                    )

                    self.liveness_status = "WAITING"

                    self.frames = 0

                    self.movement = 0.0

                    if self.final_result is None:
                        self.liveness.reset()

                cv2.putText(
                    display_image,
                    "MULTIPLE FACES DETECTED",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    2
                )

                return av.VideoFrame.from_ndarray(
                    display_image,
                    format="bgr24"
                )


            # ==================================================
            # ONE FACE
            # ==================================================

            face = faces[0]

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                display_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ==================================================
            # ALREADY VERIFIED
            # ==================================================

            with self.lock:

                if self.final_result is not None:

                    result = self.final_result

                else:

                    result = None


            if result is not None:

                cv2.putText(
                    display_image,
                    result,
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0)
                    if result == "SAME FACE"
                    else (0, 0, 255),
                    2
                )

                return av.VideoFrame.from_ndarray(
                    display_image,
                    format="bgr24"
                )


            # ==================================================
            # LIVENESS
            # ==================================================

            self.liveness.add_face_center(
                bbox
            )

            live_status = (
                self.liveness.status()
            )

            with self.lock:

                self.face_count = 1

                self.frames = live_status["frames"]

                self.movement = live_status["movement"]

                self.status = "CHECKING LIVENESS"

                self.liveness_status = (
                    "LIVE"
                    if live_status["live"]
                    else "MOVE FACE SLIGHTLY"
                )


            cv2.putText(
                display_image,
                "CHECKING LIVENESS",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )


            # ==================================================
            # LIVENESS PASSED
            # ==================================================

            if live_status["live"]:

                with self.lock:

                    self.liveness_status = (
                        "LIVENESS VERIFIED"
                    )

                    self.status = (
                        "VERIFYING IDENTITY"
                    )

                # ----------------------------------------------
                # Compare registered embedding
                # ----------------------------------------------

                current_embedding = face.embedding

                similarity = cosine_similarity(
                    current_embedding,
                    self.registered_embedding
                )

                with self.lock:

                    self.similarity = similarity


                # ----------------------------------------------
                # Face match threshold
                # ----------------------------------------------

                MATCH_THRESHOLD = 0.50


                if similarity >= MATCH_THRESHOLD:

                    with self.lock:

                        self.final_result = (
                            "SAME FACE"
                        )

                        self.status = (
                            "IDENTITY VERIFIED"
                        )

                else:

                    with self.lock:

                        self.final_result = (
                            "UNKNOWN"
                        )

                        self.status = (
                            "IDENTITY NOT MATCHED"
                        )


            return av.VideoFrame.from_ndarray(
                display_image,
                format="bgr24"
            )


        except Exception as error:

            with self.lock:

                self.status = (
                    "VERIFICATION ERROR"
                )

            cv2.putText(
                display_image,
                "VERIFICATION ERROR",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                display_image,
                format="bgr24"
            )


# ============================================================
# LOGIN CHECK
# ============================================================

if not st.session_state.get(
    "logged_in",
    False
):

    st.warning(
        "Please login before verifying your identity."
    )

    st.stop()


candidate_id = st.session_state.get(
    "candidate_id"
)


if not candidate_id:

    st.error(
        "Logged-in candidate information was not found."
    )

    st.stop()


# ============================================================
# DATABASE
# ============================================================

database = CandidateDatabase()

candidate = database.get_candidate(
    candidate_id
)


if candidate is None:

    st.error(
        "Candidate account could not be found."
    )

    st.stop()


# ============================================================
# REGISTERED EMBEDDING
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


if not embedding_path.exists():

    st.error(
        f"Embedding file not found: {embedding_path}"
    )

    st.stop()


registered_embedding = np.load(
    embedding_path
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="verify-header">

        <div class="verify-title">
            🔍 Verify Identity
        </div>

        <div class="verify-subtitle">
            Secure biometric verification using your registered face.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CANDIDATE INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="status-box">

            <div class="status-title">
                CANDIDATE
            </div>

            <div class="status-value">
                {candidate.get("name", "Candidate")}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="status-box">

            <div class="status-title">
                CANDIDATE ID
            </div>

            <div class="status-value">
                {candidate.get("candidate_id", "-")}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="status-box">

            <div class="status-title">
                VERIFICATION
            </div>

            <div class="status-value">
                Face + Liveness
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CAMERA VERSION
# ============================================================

if "verification_camera_version" not in st.session_state:

    st.session_state.verification_camera_version = 0


camera_key = (
    "faceverify_camera_"
    + str(
        st.session_state.verification_camera_version
    )
)


# ============================================================
# CAMERA
# ============================================================

st.subheader("Live Face Verification")

st.info(
    "Allow camera access and keep only your face inside "
    "the camera view. Move your face slightly from side "
    "to side when requested."
)


face_model = load_face_model()


ctx = webrtc_streamer(
    key=camera_key,

    mode=WebRtcMode.SENDRECV,

    video_processor_factory=lambda:
        VerificationProcessor(
            face_model,
            registered_embedding
        ),

    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 640
            },
            "height": {
                "ideal": 480
            }
        },
        "audio": False
    },

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    async_processing=True
)


# ============================================================
# LIVE STATUS
# ============================================================

st.divider()

st.subheader("Verification Status")


@st.fragment(run_every="500ms")
def show_live_status():

    processor = ctx.video_processor

    if processor is None:

        st.info(
            "Start the camera to begin verification."
        )

        return


    state = processor.get_state()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Face Detection",
            (
                "1 FACE"
                if state["face_count"] == 1
                else str(state["status"])
            )
        )


    with col2:

        st.metric(
            "Liveness",
            state["liveness_status"]
        )


    with col3:

        if state["similarity"] > 0:

            st.metric(
                "Face Similarity",
                f'{state["similarity"]:.3f}'
            )

        else:

            st.metric(
                "Face Similarity",
                "Waiting"
            )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = state["final_result"]


    if result == "SAME FACE":

        st.markdown(
            """
            <div class="result-box">

                <div class="result-title">
                    ✅ SAME FACE
                </div>

                <div class="result-description">
                    Liveness verified and the face matches
                    the registered candidate.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    elif result == "UNKNOWN":

        st.markdown(
            """
            <div class="result-box">

                <div class="result-title">
                    ⚠ UNKNOWN
                </div>

                <div class="result-description">
                    The live face was detected, but it does
                    not match the registered candidate.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    elif state["status"] == "NO FACE DETECTED":

        st.error(
            "NO FACE DETECTED"
        )


    elif state["status"] == "MULTIPLE FACES DETECTED":

        st.error(
            "MULTIPLE FACES DETECTED"
        )


    elif state["liveness_status"] == "MOVE FACE SLIGHTLY":

        st.info(
            "Move your face slightly from side to side."
        )


show_live_status()


# ============================================================
# SCAN AGAIN
# ============================================================

processor = ctx.video_processor

if processor is not None:

    current_state = processor.get_state()

    if current_state["final_result"] is not None:

        st.write("")

        if st.button(
            "🔄 Scan Again",
            use_container_width=True
        ):

            st.session_state.verification_camera_version += 1

            st.rerun()