from pathlib import Path
import sys
import io

import streamlit as st
import streamlit.components.v2 as components
import numpy as np
import cv2
from PIL import Image

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_database import CandidateDatabase
from src.password_security import verify_password


st.set_page_config(
    page_title="FaceVerify",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)



CANDIDATE_DB = PROJECT_ROOT / "data" / "candidates" / "candidates.json"

EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"


candidate_database = CandidateDatabase(
    database_path=str(CANDIDATE_DB)
)

DEFAULT_SESSION_VALUES = {
    "logged_in": False,
    "logged_in_candidate": None,

    # Verification state
    "verification_step": 0,
    "verification_captures": [],
    "verification_faces": [],
    "verification_embeddings": [],
    "verification_result": None,
    "verification_liveness": None,
    "verification_similarity": None,
    "verification_error": None,

    # Camera widget version
    "verification_camera_version": 0,

    # Live biometric verification during login
    "pending_login_candidate": None,
    "login_face_result": None,
    "login_face_similarity": None,
    "login_liveness": False,
    "login_liveness_centers": [],
    "login_camera_version": 0,
}


for key, value in DEFAULT_SESSION_VALUES.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       Global
       -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(56, 189, 248, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(139, 92, 246, 0.10),
                transparent 30%
            ),
            #07111f;

        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* --------------------------------------------------------
       Hide Streamlit default elements
       -------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* --------------------------------------------------------
       Typography
       -------------------------------------------------------- */

    h1,
    h2,
    h3,
    h4,
    p,
    label {
        color: #f8fafc !important;
    }


    /* --------------------------------------------------------
       Buttons
       -------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        min-height: 48px;

        border-radius: 12px;

        border: 1px solid rgba(148, 163, 184, 0.25);

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #7c3aed
            );

        color: white;

        font-weight: 700;
        font-size: 15px;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 10px 30px rgba(37, 99, 235, 0.25);
    }


    /* --------------------------------------------------------
       Input boxes
       -------------------------------------------------------- */

    div[data-baseweb="input"] {
        background: #0f1b2d;
        border-radius: 10px;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] {
        background: #0f1b2d;
    }


    /* --------------------------------------------------------
       Cards
       -------------------------------------------------------- */

    .fv-card {

        background:
            linear-gradient(
                145deg,
                rgba(15, 27, 45, 0.95),
                rgba(8, 18, 32, 0.95)
            );

        border: 1px solid rgba(148, 163, 184, 0.16);

        border-radius: 22px;

        padding: 30px;

        margin-bottom: 20px;

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.25);
    }


    .fv-small-card {

        background: rgba(15, 27, 45, 0.90);

        border: 1px solid rgba(148, 163, 184, 0.15);

        border-radius: 16px;

        padding: 22px;

        margin-bottom: 15px;
    }


    /* --------------------------------------------------------
       Hero
       -------------------------------------------------------- */

    .hero-title {

        font-size: 58px;

        font-weight: 800;

        line-height: 1.05;

        margin-bottom: 18px;

        background:
            linear-gradient(
                90deg,
                #60a5fa,
                #a78bfa,
                #f0abfc
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .hero-subtitle {

        font-size: 19px;

        line-height: 1.7;

        color: #cbd5e1 !important;

        max-width: 760px;
    }


    /* --------------------------------------------------------
       Badge
       -------------------------------------------------------- */

    .fv-badge {

        display: inline-block;

        padding: 8px 14px;

        border-radius: 999px;

        background: rgba(37, 99, 235, 0.12);

        border: 1px solid rgba(96, 165, 250, 0.25);

        color: #93c5fd !important;

        font-size: 13px;

        font-weight: 700;

        margin-bottom: 18px;
    }


    /* --------------------------------------------------------
       Verification instruction
       -------------------------------------------------------- */

    .capture-instruction {

        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.12),
                rgba(124, 58, 237, 0.10)
            );

        border: 1px solid rgba(96, 165, 250, 0.20);

        border-radius: 16px;

        padding: 20px;

        margin: 15px 0 20px 0;

        text-align: center;
    }


    .capture-number {

        font-size: 14px;

        font-weight: 700;

        color: #93c5fd !important;

        margin-bottom: 6px;
    }


    .capture-title {

        font-size: 23px;

        font-weight: 800;

        color: #ffffff !important;

        margin-bottom: 5px;
    }


    .capture-description {

        font-size: 14px;

        color: #cbd5e1 !important;
    }


    /* --------------------------------------------------------
       Verification status
       -------------------------------------------------------- */

    .status-box {

        border-radius: 18px;

        padding: 25px;

        text-align: center;

        margin: 20px 0;
    }


    .status-success {

        background: rgba(34, 197, 94, 0.10);

        border: 1px solid rgba(34, 197, 94, 0.35);
    }


    .status-danger {

        background: rgba(239, 68, 68, 0.10);

        border: 1px solid rgba(239, 68, 68, 0.35);
    }


    .status-warning {

        background: rgba(245, 158, 11, 0.10);

        border: 1px solid rgba(245, 158, 11, 0.35);
    }


    .status-title {

        font-size: 28px;

        font-weight: 800;

        margin-bottom: 8px;
    }


    .status-text {

        font-size: 15px;

        color: #cbd5e1 !important;
    }


    /* --------------------------------------------------------
       Profile
       -------------------------------------------------------- */

    .profile-name {

        font-size: 30px;

        font-weight: 800;

        color: #ffffff !important;

        margin-bottom: 5px;
    }


    .profile-id {

        color: #94a3b8 !important;

        font-size: 14px;
    }


    /* --------------------------------------------------------
       Camera area
       -------------------------------------------------------- */

    div[data-testid="stCameraInput"] {

        border-radius: 18px;

        overflow: hidden;

        border: 1px solid rgba(148, 163, 184, 0.22);

        background: #020617;

        padding: 4px;
    }


    /* --------------------------------------------------------
       Progress
       -------------------------------------------------------- */

    .progress-text {

        text-align: center;

        color: #94a3b8 !important;

        font-size: 13px;

        margin-top: 8px;
    }



    /* ========================================================
       FACEVERIFY PREMIUM UI
       ======================================================== */
    .main .block-container {
        max-width: 1400px;
        padding-top: 1.15rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
        padding-bottom: 3rem;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 18%, rgba(0, 153, 255, .12), transparent 25%),
            radial-gradient(circle at 92% 15%, rgba(112, 60, 255, .14), transparent 27%),
            radial-gradient(circle at 70% 85%, rgba(0, 225, 190, .06), transparent 24%),
            #030b18;
    }

    .fv-nav {
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 10px 6px 16px;
        margin-bottom: 12px;
        border-bottom:1px solid rgba(93, 156, 255,.12);
    }
    .fv-brand { display:flex; align-items:center; gap:11px; }
    .fv-logo {
        width:38px; height:38px; border-radius:12px;
        display:flex; align-items:center; justify-content:center;
        background:linear-gradient(145deg,#146cff,#7a3cff);
        box-shadow:0 0 28px rgba(73,105,255,.28);
        font-size:20px;
    }
    .fv-brand-name { font-size:22px; font-weight:850; letter-spacing:-.5px; color:#fff; }
    .fv-brand-name span { color:#6ea7ff; }
    .fv-trust { color:#8298b9; font-size:12px; }

    .fv-hero-grid { display:grid; grid-template-columns:1.05fr .9fr 1.15fr; gap:20px; align-items:center; margin:24px 0 18px; }
    .fv-hero-copy { padding:22px 8px 22px 4px; }
    .fv-kicker { display:inline-flex; gap:7px; align-items:center; padding:8px 13px; border:1px solid rgba(66,151,255,.28); border-radius:999px; color:#79b8ff; background:rgba(28,106,255,.10); font-size:12px; font-weight:750; }
    .fv-hero-title { font-size:56px; line-height:1.02; letter-spacing:-2.4px; font-weight:900; margin:20px 0 12px; color:#fff; }
    .fv-gradient { background:linear-gradient(90deg,#f8fbff 10%,#60a5ff 48%,#9b72ff 92%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .fv-hero-text { color:#9db0ca !important; font-size:15px; line-height:1.75; max-width:570px; }

    .fv-face-orb { min-height:310px; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; border-radius:30px; border:1px solid rgba(73,139,255,.18); background:radial-gradient(circle,rgba(30,111,255,.15),rgba(3,11,24,.2) 60%); }
    .fv-face-orb:before { content:''; width:230px; height:230px; border-radius:50%; border:1px solid rgba(45,165,255,.35); box-shadow:0 0 80px rgba(0,126,255,.14), inset 0 0 45px rgba(78,96,255,.08); position:absolute; }
    .fv-face-icon { font-size:112px; filter:drop-shadow(0 0 24px rgba(44,158,255,.45)); z-index:1; }
    .fv-scan-line { position:absolute; width:230px; height:2px; background:linear-gradient(90deg,transparent,#24d7ff,transparent); box-shadow:0 0 16px #24d7ff; animation:fvscan 2.8s ease-in-out infinite; z-index:2; }
    @keyframes fvscan { 0%,100%{transform:translateY(-105px);opacity:.35} 50%{transform:translateY(105px);opacity:1} }

    .fv-feature { min-height:170px; padding:23px; border-radius:20px; border:1px solid rgba(93,151,235,.16); background:linear-gradient(145deg,rgba(13,31,58,.92),rgba(6,17,33,.92)); box-shadow:0 18px 50px rgba(0,0,0,.22); position:relative; }
    .fv-feature-icon { width:44px; height:44px; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:20px; margin-bottom:16px; background:linear-gradient(145deg,#623cff,#267eff); box-shadow:0 8px 25px rgba(75,76,255,.25); }
    .fv-feature h3 { font-size:17px; margin:0 0 9px; }
    .fv-feature p { color:#91a5c1 !important; font-size:13px; line-height:1.55; margin:0; }
    .fv-arrow { position:absolute; right:18px; bottom:17px; width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#12386f; color:#bcd9ff; }

    .fv-section-title { font-size:25px; font-weight:850; margin:34px 0 16px; letter-spacing:-.5px; }
    .fv-panel { border:1px solid rgba(83,146,237,.17); border-radius:24px; background:linear-gradient(145deg,rgba(12,28,51,.92),rgba(5,14,27,.94)); box-shadow:0 22px 65px rgba(0,0,0,.25); padding:28px; }

    .fv-login-layout { display:grid; grid-template-columns: .85fr 1.15fr; gap:22px; margin-top:22px; }
    .fv-login-header { display:flex; gap:13px; align-items:center; margin-bottom:20px; }
    .fv-login-icon { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:linear-gradient(145deg,#1b7cff,#6940ff); font-size:21px; }
    .fv-login-title { font-size:25px; font-weight:850; margin:0; }
    .fv-login-sub { color:#8499b6 !important; font-size:12px; margin:4px 0 0; }
    .fv-secure-line { margin-top:18px; padding:11px 13px; border-radius:12px; background:rgba(29,110,255,.08); color:#8eacd3 !important; font-size:11px; border:1px solid rgba(61,137,255,.12); }
    .fv-camera-panel { min-height:410px; }
    .fv-steps { border:1px solid rgba(73,143,240,.14); border-radius:16px; padding:16px; background:rgba(3,13,27,.45); }
    .fv-step { display:flex; align-items:center; gap:9px; padding:10px 0; color:#8197b6; font-size:12px; border-bottom:1px solid rgba(90,130,180,.08); }
    .fv-step:last-child { border-bottom:0; }
    .fv-step-dot { width:20px; height:20px; border-radius:50%; border:1px solid #31527b; display:flex; align-items:center; justify-content:center; font-size:10px; }
    .fv-step.active { color:#59d8ff; }
    .fv-step.active .fv-step-dot { border-color:#16c8ff; box-shadow:0 0 12px rgba(22,200,255,.28); }

    .fv-footer-note { text-align:center; color:#536b8e !important; font-size:11px; margin:26px 0 4px; letter-spacing:.3px; }

    /* Streamlit inputs */
    div[data-baseweb="input"] { background:rgba(5,16,31,.72) !important; border:1px solid rgba(87,142,220,.22) !important; border-radius:12px !important; }
    div[data-baseweb="input"]:focus-within { border-color:#4c8dff !important; box-shadow:0 0 0 3px rgba(63,125,255,.09); }
    .stTextInput label { color:#a9bad1 !important; font-size:12px !important; font-weight:650 !important; }
    .stButton > button { border-radius:12px; min-height:46px; background:linear-gradient(90deg,#1d68ff,#713cff); border:1px solid rgba(126,174,255,.24); box-shadow:0 10px 30px rgba(48,79,255,.18); }
    .stButton > button:hover { box-shadow:0 14px 34px rgba(48,79,255,.28); transform:translateY(-1px); }

    @media (max-width: 1000px) {
        .fv-hero-grid, .fv-login-layout { grid-template-columns:1fr; }
        .fv-face-orb { min-height:240px; }
        .fv-hero-title { font-size:44px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_face_model():

    """
    Load InsightFace once and reuse it.
    """

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    model.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    return model


def image_bytes_to_bgr(image_bytes):

    """
    Convert Streamlit camera bytes to OpenCV BGR image.
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    rgb = np.array(image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    return bgr


def detect_faces(image_bgr):

    """
    Detect faces in an image.
    """

    model = load_face_model()

    faces = model.get(image_bgr)

    return faces


def get_face_embedding(face):

    """
    Extract normalized 512-dimensional embedding.
    """

    embedding = getattr(
        face,
        "embedding",
        None,
    )

    if embedding is None:
        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


def face_center(face):

    """
    Return center x/y of detected face.
    """

    bbox = np.asarray(
        face.bbox,
        dtype=np.float32,
    )

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    width = max(
        float(x2 - x1),
        1.0,
    )

    height = max(
        float(y2 - y1),
        1.0,
    )

    return (
        center_x,
        center_y,
        width,
        height,
    )


def cosine_similarity(a, b):

    """
    Calculate cosine similarity.
    """

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b)
        /
        (a_norm * b_norm)
    )


def load_registered_embedding(candidate):

    """
    Load the candidate's registered face embedding.
    """

    candidate_id = candidate.get(
        "candidate_id",
        "",
    )

    embedding_file = candidate.get(
        "embedding_file",
        "",
    )

    possible_paths = []

    if embedding_file:

        embedding_path = Path(
            embedding_file
        )

        if embedding_path.is_absolute():
            possible_paths.append(
                embedding_path
            )
        else:
            possible_paths.append(
                PROJECT_ROOT / embedding_path
            )

    if candidate_id:

        possible_paths.append(
            EMBEDDINGS_DIR / f"{candidate_id}.npy"
        )

    for path in possible_paths:

        if path.exists():

            try:

                embedding = np.load(
                    str(path)
                )

                embedding = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

                norm = np.linalg.norm(
                    embedding
                )

                if norm == 0:
                    continue

                return embedding / norm

            except Exception:
                continue

    return None


def calculate_liveness(face_records):

    """
    Motion-based liveness using the three still captures.

    We calculate horizontal movement relative to face width.
    This makes the check less dependent on how close the
    person is to the camera.

    This is a demo-level liveness check, not production
    anti-spoofing.
    """

    if len(face_records) < 3:
        return False, 0.0

    centers = []

    for face in face_records:

        cx, cy, width, height = face_center(face)

        centers.append(
            {
                "x": cx,
                "y": cy,
                "width": width,
                "height": height,
            }
        )

    x_values = [
        item["x"]
        for item in centers
    ]

    movement_pixels = (
        max(x_values)
        -
        min(x_values)
    )

    average_width = np.mean(
        [
            item["width"]
            for item in centers
        ]
    )

    if average_width <= 0:
        return False, 0.0

    normalized_movement = (
        movement_pixels
        /
        average_width
    )

    # Require approximately 12% of face width
    # movement across the captures.
    live = normalized_movement >= 0.12

    return (
        bool(live),
        float(normalized_movement),
    )


def reset_verification():

    """
    Completely reset verification state.
    """

    st.session_state.verification_step = 0

    st.session_state.verification_captures = []

    st.session_state.verification_faces = []

    st.session_state.verification_embeddings = []

    st.session_state.verification_result = None

    st.session_state.verification_liveness = None

    st.session_state.verification_similarity = None

    st.session_state.verification_error = None

    st.session_state.verification_camera_version += 1


def process_verification_capture(uploaded_image):

    """
    Process one still camera capture.

    Returns:
        success, message
    """

    try:

        image_bytes = uploaded_image.getvalue()

        image_bgr = image_bytes_to_bgr(
            image_bytes
        )

    except Exception as error:

        return (
            False,
            f"Unable to read camera image: {error}",
        )


    try:

        faces = detect_faces(
            image_bgr
        )

    except Exception as error:

        return (
            False,
            f"Face detection failed: {error}",
        )

    if len(faces) == 0:

        return (
            False,
            "NO FACE DETECTED. Please position your face clearly inside the camera.",
        )

    if len(faces) > 1:

        return (
            False,
            "MULTIPLE FACES DETECTED. Only one person should be visible.",
        )


    face = faces[0]

    detection_score = float(
        getattr(
            face,
            "det_score",
            1.0,
        )
    )

    if detection_score < 0.40:

        return (
            False,
            "The face is not clear enough. Please move closer and capture again.",
        )


    _, _, width, height = face_center(
        face
    )

    image_height, image_width = (
        image_bgr.shape[:2]
    )

    face_area_ratio = (
        width * height
    ) / (
        image_width * image_height
    )


    # Reject extremely tiny faces.
    if face_area_ratio < 0.025:

        return (
            False,
            "Your face is too far from the camera. Please move closer and capture again.",
        )

    embedding = get_face_embedding(
        face
    )

    if embedding is None:

        return (
            False,
            "Unable to generate a face embedding. Please capture again.",
        )

    st.session_state.verification_captures.append(
        image_bytes
    )

    st.session_state.verification_faces.append(
        face
    )

    st.session_state.verification_embeddings.append(
        embedding
    )

    return (
        True,
        "Face captured successfully.",
    )


LIVE_DETECTOR_PATH = MODEL_DIR / "det_10g.onnx"
LIVE_RECOGNITION_PATH = MODEL_DIR / "w600k_r50.onnx"

MATCH_THRESHOLD = 0.50


@st.cache_resource
def load_live_detector():
    detector = get_model(
        str(LIVE_DETECTOR_PATH),
        providers=["CPUExecutionProvider"],
    )
    detector.prepare(
        ctx_id=-1,
        input_size=(320, 320),
    )
    return detector


@st.cache_resource
def load_live_recognition_model():
    recognition_model = get_model(
        str(LIVE_RECOGNITION_PATH),
        providers=["CPUExecutionProvider"],
    )
    recognition_model.prepare(ctx_id=-1)
    return recognition_model


def generate_live_embedding(image_bgr, landmarks):
    """Generate the 512-D ArcFace embedding from one temporary live frame."""
    recognition_model = load_live_recognition_model()

    input_size = getattr(
        recognition_model,
        "input_size",
        (112, 112),
    )

    aligned_face = face_align.norm_crop(
        image_bgr,
        landmark=landmarks,
        image_size=input_size[0],
        mode="arcface",
    )

    embedding = recognition_model.get_feat(aligned_face)
    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


def reset_login_face_verification():
    """Reset only the live face-verification state."""
    st.session_state.login_face_result = None
    st.session_state.login_face_similarity = None
    st.session_state.login_liveness = False
    st.session_state.login_liveness_centers = []
    st.session_state.login_camera_version += 1


def update_login_liveness(bbox):
    """
    Demo-level motion liveness.

    Temporary live frames are used only in memory.
    No verification photo is saved.
    """
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]

    center_x = (x1 + x2) / 2.0
    face_width = max(x2 - x1, 1.0)

    centers = st.session_state.login_liveness_centers

    centers.append(
        {
            "x": center_x,
            "width": face_width,
        }
    )

    # Keep a small rolling window.
    if len(centers) > 8:
        centers.pop(0)

    if len(centers) < 3:
        return False, 0.0

    movement = (
        max(item["x"] for item in centers)
        - min(item["x"] for item in centers)
    )

    average_width = float(
        np.mean(
            [item["width"] for item in centers]
        )
    )

    normalized_movement = movement / max(
        average_width,
        1.0,
    )

    # Require visible movement before face matching.
    live = normalized_movement >= 0.08

    return bool(live), float(normalized_movement)


def process_live_login_frame(frame_data, candidate):
    """
    Process one temporary browser-camera frame.

    The frame is decoded and processed in memory only.
    It is never written to disk.
    """
    try:
        if not frame_data:
            return

        import base64

        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]

        raw_bytes = base64.b64decode(frame_data)

        image = Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")

        rgb_image = np.asarray(image)

        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        detector = load_live_detector()

        bboxes, kpss = detector.detect(
            bgr_image,
            max_num=0,
        )


        if bboxes is None or len(bboxes) == 0:
            st.session_state.login_face_result = "NO FACE"
            st.session_state.login_face_similarity = None
            st.session_state.login_liveness = False
            st.session_state.login_liveness_centers = []
            return

        if len(bboxes) > 1:
            st.session_state.login_face_result = "MULTIPLE FACES"
            st.session_state.login_face_similarity = None
            st.session_state.login_liveness = False
            return

        bbox = bboxes[0]
        landmarks = kpss[0]

        detection_confidence = float(bbox[4])

        if detection_confidence < 0.40:
            st.session_state.login_face_result = "NO FACE"
            st.session_state.login_face_similarity = None
            return

        live, _ = update_login_liveness(bbox)

        st.session_state.login_liveness = live

        if not live:
            st.session_state.login_face_result = "LIVENESS CHECK"
            st.session_state.login_face_similarity = None
            return

        registered_embedding = load_registered_embedding(
            candidate
        )

        if registered_embedding is None:
            st.session_state.login_face_result = "ERROR"
            st.session_state.login_face_similarity = None
            return

        live_embedding = generate_live_embedding(
            bgr_image,
            landmarks,
        )

        if live_embedding is None:
            st.session_state.login_face_result = "ERROR"
            st.session_state.login_face_similarity = None
            return


        similarity = cosine_similarity(
            registered_embedding,
            live_embedding,
        )

        st.session_state.login_face_similarity = similarity

        if similarity >= MATCH_THRESHOLD:
            st.session_state.login_face_result = "FACE DETECTED"
        else:
            st.session_state.login_face_result = "UNKNOWN FACE"

    except Exception as error:
        st.session_state.login_face_result = "ERROR"
        st.session_state.login_face_similarity = None
        st.session_state.verification_error = str(error)

live_login_camera_component = components.component(
    "faceverify_live_login_camera",
    html="""
    <div class="fv-live-camera">
        <video id="fv-login-video" autoplay playsinline muted></video>
        <div id="fv-login-camera-status">Starting camera...</div>
    </div>
    """,
    css="""
    .fv-live-camera {
        width: 100%;
        max-width: 760px;
        margin: 0 auto;
        background: #020617;
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 22px;
        overflow: hidden;
        position: relative;
        aspect-ratio: 4 / 3;
    }

    #fv-login-video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        transform: scaleX(-1);
        background: #020617;
    }

    #fv-login-camera-status {
        position: absolute;
        left: 16px;
        bottom: 16px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(2,6,23,.75);
        color: #e2e8f0;
        font: 600 13px sans-serif;
    }
    """,
    js="""
    export default function(component) {
        const { setStateValue } = component;

        const video =
            component.parentElement.querySelector("#fv-login-video");

        const status =
            component.parentElement.querySelector("#fv-login-camera-status");

        let stream = null;
        let running = true;

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: "user"
                    },
                    audio: false
                });

                video.srcObject = stream;
                await video.play();

                status.textContent = "LIVE CAMERA";
                setStateValue("camera_status", "live");

                async function sendFrame() {
                    if (!running || video.readyState < 2) {
                        return;
                    }

                    const canvas =
                        document.createElement("canvas");

                    canvas.width = 480;
                    canvas.height = 360;

                    const context =
                        canvas.getContext("2d");

                    context.drawImage(
                        video,
                        0,
                        0,
                        canvas.width,
                        canvas.height
                    );

                    const frame =
                        canvas.toDataURL(
                            "image/jpeg",
                            0.45
                        );

                    setStateValue("frame", frame);
                }

                // Give the browser time to initialize the camera.
                setTimeout(sendFrame, 1500);

                // Send one temporary frame every 3 seconds.
                setInterval(sendFrame, 3000);

            } catch (error) {
                status.textContent = "Camera unavailable";

                setStateValue(
                    "camera_status",
                    "error"
                );

                setStateValue(
                    "camera_error",
                    String(error)
                );
            }
        }

        startCamera();

        return () => {
            running = false;

            if (stream) {
                stream.getTracks().forEach(
                    track => track.stop()
                );
            }
        };
    }
    """,
)


def render_live_login_camera():
    camera_key = (
        f"login_live_camera_"
        f"{st.session_state.login_camera_version}"
    )

    return live_login_camera_component(
        default={
            "camera_status": "starting",
            "camera_error": "",
            "frame": "",
        },
        on_camera_status_change=lambda: None,
        on_camera_error_change=lambda: None,
        on_frame_change=lambda: None,
        key=camera_key,
    )


def show_live_login_verification(candidate):
    """Show the registered details and then perform live verification."""
    from html import escape

    st.markdown(
        '<div class="fv-badge">LIVE BIOMETRIC VERIFICATION</div>',
        unsafe_allow_html=True,
    )

    name = escape(str(candidate.get("name", "")))
    email = escape(str(candidate.get("email", "")))
    mobile = escape(str(candidate.get("mobile", "")))
    date_of_birth = escape(str(candidate.get("date_of_birth", "")))
    gender = escape(str(candidate.get("gender", "")))

    st.html(
        f"""
        <div class="fv-panel" style="margin-bottom:20px;">
            <div class="fv-kicker">◈ REGISTERED DETAILS</div>
            <div style="
                font-size:26px;
                font-weight:800;
                color:#f8fafc;
                margin:4px 0 20px;
            ">
                Your Registration Information
            </div>

            <div style="
                display:grid;
                grid-template-columns:repeat(2,minmax(0,1fr));
                gap:12px;
            ">
                <div style="
                    background:rgba(255,255,255,.035);
                    border:1px solid rgba(148,163,184,.12);
                    border-radius:14px;
                    padding:16px;
                ">
                    <div style="color:#8fa4c2;font-size:11px;text-transform:uppercase;letter-spacing:.8px;">Name</div>
                    <div style="color:#f8fafc;font-size:15px;font-weight:700;margin-top:7px;">{name}</div>
                </div>

                <div style="
                    background:rgba(255,255,255,.035);
                    border:1px solid rgba(148,163,184,.12);
                    border-radius:14px;
                    padding:16px;
                ">
                    <div style="color:#8fa4c2;font-size:11px;text-transform:uppercase;letter-spacing:.8px;">Email</div>
                    <div style="color:#f8fafc;font-size:15px;font-weight:700;margin-top:7px;word-break:break-word;">{email}</div>
                </div>

                <div style="
                    background:rgba(255,255,255,.035);
                    border:1px solid rgba(148,163,184,.12);
                    border-radius:14px;
                    padding:16px;
                ">
                    <div style="color:#8fa4c2;font-size:11px;text-transform:uppercase;letter-spacing:.8px;">Mobile</div>
                    <div style="color:#f8fafc;font-size:15px;font-weight:700;margin-top:7px;">{mobile}</div>
                </div>

                <div style="
                    background:rgba(255,255,255,.035);
                    border:1px solid rgba(148,163,184,.12);
                    border-radius:14px;
                    padding:16px;
                ">
                    <div style="color:#8fa4c2;font-size:11px;text-transform:uppercase;letter-spacing:.8px;">Date of Birth</div>
                    <div style="color:#f8fafc;font-size:15px;font-weight:700;margin-top:7px;">{date_of_birth}</div>
                </div>

                <div style="
                    background:rgba(255,255,255,.035);
                    border:1px solid rgba(148,163,184,.12);
                    border-radius:14px;
                    padding:16px;
                    grid-column:1 / -1;
                ">
                    <div style="color:#8fa4c2;font-size:11px;text-transform:uppercase;letter-spacing:.8px;">Gender</div>
                    <div style="color:#f8fafc;font-size:15px;font-weight:700;margin-top:7px;">{gender}</div>
                </div>
            </div>
        </div>
        """,        
    )

    st.markdown(
        """
        <div class="fv-card">
            <h1>Verify Your Face</h1>
            <p>
                Keep only your face in front of the camera.
                Move your head slightly left and right.
                The system checks liveness first and then
                compares the live face with your registered face.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.login_face_result
    similarity = st.session_state.login_face_similarity
    live = st.session_state.login_liveness

    if result == "FACE DETECTED":
        st.markdown(
            """
            <div class="status-box status-success">
                <div class="status-title">✅ FACE DETECTED</div>
                <div class="status-text">
                    Liveness passed and the live face matches the registered identity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if similarity is not None:
            st.metric("Face Similarity", f"{similarity:.4f}")

        st.success("Identity verification completed successfully.")

        if st.button("Continue to Profile", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.logged_in_candidate = candidate
            st.session_state.pending_login_candidate = None
            reset_login_face_verification()
            st.rerun()

        return

    if result == "NO MATCH":
        st.markdown(
            """
            <div class="status-box status-danger">
                <div class="status-title">❌ NO MATCH</div>
                <div class="status-text">
                    The live face does not match the registered identity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if similarity is not None:
            st.metric("Face Similarity", f"{similarity:.4f}")

    elif result == "MULTIPLE FACES":
        st.markdown(
            """
            <div class="status-box status-warning">
                <div class="status-title">⚠ MULTIPLE FACES</div>
                <div class="status-text">
                    Only one person should be visible during verification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif result == "NO FACE":
        st.markdown(
            """
            <div class="status-box status-warning">
                <div class="status-title">⚠ NO FACE</div>
                <div class="status-text">
                    No face is currently visible. Please look at the camera.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif result == "LIVENESS CHECK":
        st.info(
            "🔄 Liveness check in progress. Move your head slightly left and right."
        )

    elif result == "ERROR":
        st.error(
            "Face verification encountered an error. Please restart the scan."
        )

    else:
        st.info(
            "🟢 Live camera is ready. Look at the camera and move your head slightly."
        )

    if live:
        st.success("✓ Liveness detected")

    camera_state = render_live_login_camera()

    if isinstance(camera_state, dict):
        camera_status = camera_state.get("camera_status", "starting")
        frame = camera_state.get("frame", "")

        if camera_status == "live" and frame:
            process_live_login_frame(frame, candidate)
            st.rerun()

    st.caption(
        "Live camera frames are processed temporarily in memory. "
        "No verification photo is saved to disk."
    )

    if st.button("🔄 Restart Face Verification", use_container_width=True):
        reset_login_face_verification()
        st.rerun()

    if st.button("← Back to Login", use_container_width=True):
        st.session_state.pending_login_candidate = None
        reset_login_face_verification()
        st.rerun()

def landing_page():
    st.markdown("""
    <div class="fv-nav">
      <div class="fv-brand"><div class="fv-logo">🛡️</div><div class="fv-brand-name">Face<span>Verify</span></div></div>
      <div class="fv-trust">◈ Secure • Smart • Reliable</div>
    </div>
    <div class="fv-hero-grid">
      <div class="fv-hero-copy">
        <div class="fv-kicker">◈ BIOMETRIC IDENTITY PLATFORM</div>
        <div class="fv-hero-title">Secure <span class="fv-gradient">Identity Verification</span></div>
        <div class="fv-hero-text">Advanced face recognition, biometric embeddings and liveness detection — all in one professional platform. Verify that the person in front of the camera is really the registered user.</div>
      </div>
      <div class="fv-face-orb"><div class="fv-face-icon">🧑‍💻</div><div class="fv-scan-line"></div></div>
      <div style="display:grid;gap:14px;">
        <div class="fv-feature"><div class="fv-feature-icon">👤</div><h3>Face Registration</h3><p>Register your identity and capture multiple face samples for biometric verification.</p><div class="fv-arrow">›</div></div>
        <div class="fv-feature"><div class="fv-feature-icon">🛡️</div><h3>Liveness Detection</h3><p>Verify that the face presented during verification belongs to a live person.</p><div class="fv-arrow">›</div></div>
        <div class="fv-feature"><div class="fv-feature-icon">⌕</div><h3>Face Matching</h3><p>Compare the live face with the registered biometric embedding.</p><div class="fv-arrow">›</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,1.2,1])
    with c2:
        if st.button("→  Get Started", use_container_width=True): st.switch_page(home_page)
    st.markdown('<div class="fv-footer-note">FACEVERIFY  •  Secure Today  •  Safer Tomorrow</div>', unsafe_allow_html=True)


def home_page_function():
    st.markdown("""
    <div class="fv-nav">
      <div class="fv-brand"><div class="fv-logo">🛡️</div><div class="fv-brand-name">Face<span>Verify</span></div></div>
      <div style="display:flex;gap:28px;align-items:center;color:#a8b8cf;font-size:13px;"><span style="color:#fff;">Home</span><span>Login</span><span>Register</span><span>About</span></div>
      <div class="fv-trust">◈ Secure • Smart • Reliable</div>
    </div>
    <div class="fv-hero-copy" style="padding:28px 0 14px;">
      <div class="fv-kicker">◈ FACEVERIFY</div>
      <div class="fv-hero-title" style="font-size:48px;">Your Identity, <span class="fv-gradient">Verified.</span></div>
      <div class="fv-hero-text">Register your identity, sign in securely and verify your face using biometric face recognition with liveness detection.</div>
    </div>
    """, unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='large')
    with c1:
        st.markdown('<div class="fv-panel"><div class="fv-login-header"><div class="fv-login-icon">👤</div><div><div class="fv-login-title">New User?</div><div class="fv-login-sub">Create your secure FaceVerify identity</div></div></div><p style="color:#93a8c4!important;font-size:13px;line-height:1.65;">Create your account and register your face using ten clear samples to build your biometric profile.</p></div>',unsafe_allow_html=True)
        if st.button("Create Account  →",use_container_width=True,key="home_create_account"): st.switch_page(register_page)
    with c2:
        st.markdown('<div class="fv-panel"><div class="fv-login-header"><div class="fv-login-icon">🔐</div><div><div class="fv-login-title">Existing User?</div><div class="fv-login-sub">Secure access with password + live face</div></div></div><p style="color:#93a8c4!important;font-size:13px;line-height:1.65;">Sign in with your email and password. FaceVerify then activates live biometric verification before access is granted.</p></div>',unsafe_allow_html=True)
        if st.button("Login Securely  →",use_container_width=True,key="home_login"): st.switch_page(login_page)
    st.markdown('<div class="fv-footer-note">FACEVERIFY  •  Biometric security designed for a safer digital experience</div>',unsafe_allow_html=True)

def login_page_function():
    if st.session_state.logged_in:
        candidate=st.session_state.logged_in_candidate
        if candidate is None:
            st.session_state.logged_in=False; st.rerun()
        st.markdown('<div class="fv-nav"><div class="fv-brand"><div class="fv-logo">🛡️</div><div class="fv-brand-name">Face<span>Verify</span></div></div><div class="fv-trust">◈ Secure • Smart • Reliable</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="fv-panel"><div class="fv-kicker">◈ ACCOUNT VERIFIED</div><div class="fv-hero-title" style="font-size:40px;">Welcome, <span class="fv-gradient">{candidate.get("name","User")}</span></div><p style="color:#91a5c1!important;">Candidate ID: {candidate.get("candidate_id","N/A")}</p></div>',unsafe_allow_html=True)
        a,b=st.columns(2)
        with a:
            if st.button("📷 Scan My Face",use_container_width=True,key="profile_scan"): st.session_state.pending_login_candidate=candidate; st.session_state.logged_in=False; st.session_state.logged_in_candidate=None; reset_login_face_verification(); st.rerun()
        with b:
            if st.button("Logout",use_container_width=True,key="profile_logout"): st.session_state.logged_in=False; st.session_state.logged_in_candidate=None; st.session_state.pending_login_candidate=None; reset_login_face_verification(); st.rerun()
        return

    pending_candidate=st.session_state.pending_login_candidate
    if pending_candidate is not None:
        st.markdown('<div class="fv-nav"><div class="fv-brand"><div class="fv-logo">🛡️</div><div class="fv-brand-name">Face<span>Verify</span></div></div><div class="fv-trust">◉ Live biometric verification</div></div>',unsafe_allow_html=True)
        show_live_login_verification(pending_candidate)
        return

    st.markdown('<div class="fv-nav"><div class="fv-brand"><div class="fv-logo">🛡️</div><div class="fv-brand-name">Face<span>Verify</span></div></div><div class="fv-trust">◈ Secure • Smart • Reliable</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="fv-hero-copy" style="padding:22px 0 8px;"><div class="fv-kicker">◈ WELCOME BACK</div><div class="fv-hero-title" style="font-size:46px;">Secure <span class="fv-gradient">Login</span></div><div class="fv-hero-text">Sign in to your account. After your password is verified, FaceVerify activates live biometric verification.</div></div>',unsafe_allow_html=True)

    left,right=st.columns([0.88,1.12],gap="large")
    with left:
        st.markdown('<div class="fv-panel"><div class="fv-login-header"><div class="fv-login-icon">🔐</div><div><div class="fv-login-title">Welcome Back</div><div class="fv-login-sub">Secure access to your FaceVerify account</div></div></div>',unsafe_allow_html=True)
        email=st.text_input("Email Address",placeholder="Enter your registered email",key="login_email")
        password=st.text_input("Password",type="password",placeholder="Enter your password",key="login_password")
        st.markdown('<div class="fv-secure-line">🛡️ Your credentials are checked securely before biometric verification begins.</div>',unsafe_allow_html=True)
        st.write("")
        login_clicked=st.button("→  Login Securely",use_container_width=True,key="secure_login_button")
        st.markdown('<div style="text-align:center;color:#6f86a7;font-size:11px;margin-top:14px;">Don’t have an account? Use the Register page to create one.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with right:
        st.markdown('<div class="fv-panel fv-camera-panel"><div class="fv-login-header"><div class="fv-login-icon">📷</div><div><div class="fv-login-title">Live Biometric Verification</div><div class="fv-login-sub">Camera activates only after successful login</div></div></div><div style="height:190px;border-radius:18px;border:1px solid rgba(77,143,235,.16);background:radial-gradient(circle at center,rgba(31,111,255,.13),rgba(2,10,22,.72));display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;"><div style="font-size:76px;opacity:.75;filter:drop-shadow(0 0 20px rgba(44,158,255,.4));">🧑</div><div style="position:absolute;width:150px;height:190px;border:1px solid rgba(67,180,255,.4);border-radius:75px;box-shadow:0 0 35px rgba(0,137,255,.12);"></div><div style="position:absolute;width:70%;height:1px;background:linear-gradient(90deg,transparent,#1ed8ff,transparent);box-shadow:0 0 12px #1ed8ff;"></div></div><div class="fv-steps" style="margin-top:15px;"><div class="fv-step active"><span class="fv-step-dot">✓</span> 1. Password verification</div><div class="fv-step"><span class="fv-step-dot">2</span> 2. Live camera</div><div class="fv-step"><span class="fv-step-dot">3</span> 3. Liveness check</div><div class="fv-step"><span class="fv-step-dot">4</span> 4. Face matching</div></div><div class="fv-secure-line">🔒 No verification photo is saved. Live frames are processed temporarily in memory.</div></div>',unsafe_allow_html=True)

    if not login_clicked: return
    if not email.strip(): st.error("Please enter your email address."); return
    if not password: st.error("Please enter your password."); return
    candidate=candidate_database.get_candidate_by_email(email.strip())
    if candidate is None: st.error("Invalid email or password."); return
    stored_salt=candidate.get("password_salt",""); stored_hash=candidate.get("password_hash","")
    if not stored_salt or not stored_hash: st.error("This account does not contain valid authentication credentials."); return
    try: password_valid=verify_password(password,stored_salt,stored_hash)
    except Exception: password_valid=False
    if not password_valid: st.error("Invalid email or password."); return
    if candidate.get("registration_status")!="completed": st.warning("Your account exists, but face registration has not been completed."); return
    fresh_candidate=candidate_database.get_candidate(candidate.get("candidate_id")) or candidate
    st.session_state.pending_login_candidate=fresh_candidate
    st.session_state.logged_in=False
    st.session_state.logged_in_candidate=None
    reset_login_face_verification()
    st.rerun()

def verify_page_function():

    if not st.session_state.logged_in:

        st.warning(
            "Please login before starting face verification."
        )

        if st.button(
            "Go to Login",
            use_container_width=True,
        ):

            st.switch_page(
                login_page
            )

        return


    candidate = (
        st.session_state.logged_in_candidate
    )


    if candidate is None:

        st.error(
            "Unable to identify the logged-in account."
        )

        return

    st.markdown(
        '<div class="fv-badge">BIOMETRIC VERIFICATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fv-card">

            <h1>Scan My Face</h1>

            <p>
                Live camera verification is used here.
                Temporary camera frames are checked for liveness
                and face matching. No verification photo is saved.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        st.session_state.verification_result
        is not None
    ):

        show_verification_result(
            candidate
        )

        return

    step = (
        st.session_state.verification_step
    )


    if step >= 3:

        perform_final_verification(
            candidate
        )

        show_verification_result(
            candidate
        )

        return

    instructions = [

        (
            "Capture 1 of 3",
            "Look Straight",
            "Keep your face straight and centered in the camera."
        ),

        (
            "Capture 2 of 3",
            "Move Slightly Left",
            "Move your head slightly toward your left and capture."
        ),

        (
            "Capture 3 of 3",
            "Move Slightly Right",
            "Move your head slightly toward your right and capture."
        ),
    ]


    number_text, title, description = (
        instructions[step]
    )


    st.markdown(
        f"""
        <div class="capture-instruction">

            <div class="capture-number">
                {number_text}
            </div>

            <div class="capture-title">
                {title}
            </div>

            <div class="capture-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_value = (
        step / 3
    )

    st.progress(
        progress_value
    )

    st.markdown(
        f"""
        <div class="progress-text">
            {step} of 3 captures completed
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.write("")

    camera_key = (
        f"verification_camera_"
        f"{st.session_state.verification_camera_version}_"
        f"{step}"
    )


    uploaded_image = st.camera_input(
        "Capture Face",
        key=camera_key,
    )

    if uploaded_image is not None:

        # Avoid processing the same widget repeatedly.
        processed_key = (
            f"_processed_{camera_key}"
        )

        image_identifier = (
            uploaded_image.getvalue()
        )

        current_hash = hash(
            image_identifier
        )

        previous_hash = (
            st.session_state.get(
                processed_key
            )
        )


        if previous_hash != current_hash:

            st.session_state[
                processed_key
            ] = current_hash


            success, message = (
                process_verification_capture(
                    uploaded_image
                )
            )


            if success:

                st.session_state.verification_step += 1

                st.success(
                    message
                )

                # Automatically move to the next
                # capture after a successful photo.
                st.rerun()

            else:

                st.error(
                    message
                )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 Restart Scan",
            use_container_width=True,
        ):

            reset_verification()

            st.rerun()


    with col2:

        if st.button(
            "← Back to Profile",
            use_container_width=True,
        ):

            reset_verification()

            st.switch_page(
                login_page
            )

def perform_final_verification(
    candidate
):

    """
    Perform:

        1. Liveness detection
        2. Face embedding comparison
        3. Final verification result
    """
    faces = (
        st.session_state.verification_faces
    )

    embeddings = (
        st.session_state.verification_embeddings
    )

    if len(faces) != 3:

        st.session_state.verification_result = (
            "NO FACE DETECTED"
        )

        st.session_state.verification_error = (
            "Three valid face captures are required."
        )

        return


    if len(embeddings) != 3:

        st.session_state.verification_result = (
            "UNKNOWN"
        )

        st.session_state.verification_error = (
            "Unable to generate face embeddings."
        )

        return

    live, movement = (
        calculate_liveness(
            faces
        )
    )


    st.session_state.verification_liveness = (
        live
    )


    # Store normalized movement value
    st.session_state.verification_error = (
        f"Liveness movement: {movement:.2f}"
    )

    if not live:

        st.session_state.verification_result = (
            "LIVENESS FAILED"
        )

        st.session_state.verification_similarity = (
            None
        )

        return

    registered_embedding = (
        load_registered_embedding(
            candidate
        )
    )


    if registered_embedding is None:

        st.session_state.verification_result = (
            "UNKNOWN"
        )

        st.session_state.verification_similarity = (
            None
        )

        st.session_state.verification_error = (
            "Registered face embedding could not be found."
        )

        return

    similarities = []

    for embedding in embeddings:

        similarity = cosine_similarity(
            registered_embedding,
            embedding,
        )

        similarities.append(
            similarity
        )


    # Use the average similarity for final decision.
    average_similarity = float(
        np.mean(similarities)
    )


    st.session_state.verification_similarity = (
        average_similarity
    )


    MATCH_THRESHOLD = 0.50


    if average_similarity >= MATCH_THRESHOLD:

        st.session_state.verification_result = (
            "SAME FACE"
        )

    else:

        st.session_state.verification_result = (
            "UNKNOWN"
        )

def show_verification_result(
    candidate
):

    result = (
        st.session_state.verification_result
    )

    liveness = (
        st.session_state.verification_liveness
    )

    similarity = (
        st.session_state.verification_similarity
    )

    if result == "SAME FACE":

        st.markdown(
            """
            <div class="status-box status-success">

                <div class="status-title">
                    ✅ SAME FACE
                </div>

                <div class="status-text">
                    Liveness verification passed and the
                    captured face matches the registered identity.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        if similarity is not None:

            st.metric(
                "Face Similarity",
                f"{similarity:.3f}",
            )


        st.success(
            "Identity verification completed successfully."
        )

    elif result == "UNKNOWN":

        st.markdown(
            """
            <div class="status-box status-danger">

                <div class="status-title">
                    ❌ UNKNOWN
                </div>

                <div class="status-text">
                    A live face was detected, but it does not
                    match the registered identity.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        if similarity is not None:

            st.metric(
                "Face Similarity",
                f"{similarity:.3f}",
            )

    elif result == "NO FACE DETECTED":

        st.markdown(
            """
            <div class="status-box status-warning">

                <div class="status-title">
                    ⚠ NO FACE DETECTED
                </div>

                <div class="status-text">
                    A valid face could not be detected during
                    verification.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    elif result == "LIVENESS FAILED":

        st.markdown(
            """
            <div class="status-box status-danger">

                <div class="status-title">
                    ❌ LIVENESS VERIFICATION FAILED
                </div>

                <div class="status-text">
                    The required movement between the three
                    captures was not detected.
                    Please perform the scan again.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    elif result == "MULTIPLE FACES DETECTED":

        st.markdown(
            """
            <div class="status-box status-warning">

                <div class="status-title">
                    ⚠ MULTIPLE FACES DETECTED
                </div>

                <div class="status-text">
                    Only one person should be visible during
                    face verification.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    if liveness is True:

        st.success(
            "✓ Liveness detection passed"
        )

    elif liveness is False:

        st.error(
            "✗ Liveness detection failed"
        )


    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 Scan Again",
            use_container_width=True,
        ):

            reset_verification()

            st.rerun()


    with col2:

        if st.button(
            "👤 Back to Profile",
            use_container_width=True,
        ):

            reset_verification()

            st.switch_page(
                login_page
            )

def about_page_function():

    st.markdown(
        '<div class="fv-badge">ABOUT FACEVERIFY</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fv-card">

            <h1>About FaceVerify</h1>

            <p>
                FaceVerify is a biometric identity verification
                application developed using Python and computer
                vision technologies.
            </p>

            <h3>Technology</h3>

            <p>
                The system uses InsightFace for face detection
                and biometric embedding generation.
            </p>

            <h3>Face Registration</h3>

            <p>
                During registration, multiple face samples are
                captured and combined to create a registered
                biometric representation.
            </p>

            <h3>Face Verification</h3>

            <p>
                During login verification, a live camera stream
                is used. Temporary frames are checked for liveness
                and the live face is compared with the registered
                face embedding. Verification photos are not saved.
            </p>

            <h3>Verification Results</h3>

            <p>
                The system can identify FACE DETECTED, NO MATCH,
                NO FACE and MULTIPLE FACES conditions.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


landing_page_obj = st.Page(
    landing_page,
    title="Welcome",
    icon="🔐",
    default=True,
)


home_page = st.Page(
    home_page_function,
    title="Home",
    icon="🏠",
)


register_page = st.Page(
    "pages/register_candidate.py",
    title="Register",
    icon="👤",
)


login_page = st.Page(
    login_page_function,
    title="Login",
    icon="🔐",
)


verify_page = st.Page(
    verify_page_function,
    title="Verify Identity",
    icon="📷",
)


about_page = st.Page(
    about_page_function,
    title="About",
    icon="ℹ️",
)


pg = st.navigation(
    [
        landing_page_obj,
        home_page,
        register_page,
        login_page,
        verify_page,
        about_page,
    ],
    position="hidden",
)
pg.run()