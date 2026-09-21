import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from src.candidate_database import CandidateDatabase
from src.password_security import hash_password
from src.face_capture import FaceCaptureManager
from datetime import date

st.set_page_config(
    page_title="FaceVerify - Face Registration",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DB = (
    PROJECT_ROOT
    / "data"
    / "candidates"
    / "candidates.json"
)

FACE_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "face_images"
)

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "embeddings"
)

INSIGHTFACE_MODEL_DIR = (
    Path.home()
    / ".insightface"
    / "models"
    / "buffalo_l"
)

DETECTION_MODEL_PATH = (
    INSIGHTFACE_MODEL_DIR
    / "det_10g.onnx"
)


FACE_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

candidate_database = CandidateDatabase(
    database_path=str(CANDIDATE_DB)
)

capture_manager = FaceCaptureManager(
    base_directory=str(FACE_IMAGE_DIR)
)

@st.cache_resource
def load_fast_detector():

    if not DETECTION_MODEL_PATH.exists():

        raise FileNotFoundError(
            "InsightFace detection model was not found:\n"
            f"{DETECTION_MODEL_PATH}"
        )

    detector = get_model(
        str(DETECTION_MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    detector.prepare(
        ctx_id=0,
        input_size=(320, 320)
    )

    return detector


fast_detector = load_fast_detector()


@st.cache_resource
def load_embedding_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return model

if "registration_started" not in st.session_state:
    st.session_state.registration_started = False

if "candidate_id" not in st.session_state:
    st.session_state.candidate_id = None

if "registration_data" not in st.session_state:
    st.session_state.registration_data = {}

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

if "registration_finished" not in st.session_state:
    st.session_state.registration_finished = False

if "processing_faces" not in st.session_state:
    st.session_state.processing_faces = False


st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    .title {
        font-size: 38px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #667085;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        color: #172033;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .camera-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e4e7ec;
        box-shadow: 0 4px 18px rgba(16,24,40,0.06);
    }

    .progress-number {
        font-size: 34px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    .progress-text {
        text-align: center;
        color: #667085;
    }

    .capture-help {
        color: #667085;
        font-size: 14px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title">Create Your FaceVerify Account</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Create your account and securely register your facial identity.'
    '</div>',
    unsafe_allow_html=True
)


if st.session_state.registration_finished:

    st.success(
        "Face registration completed successfully."
    )

    st.markdown(
        f"""
        ### Registration Complete

        **Candidate ID:** {st.session_state.candidate_id}

        **Face samples:** 10

        Your facial identity has been registered successfully.
        """
    )

    if st.button(
        "Go to Login",
        use_container_width=True
    ):

        st.switch_page(
            "streamlit_app.py"
        )

    st.stop()


if not st.session_state.registration_started:

    st.markdown(
        '<div class="section-title">Personal Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        name = st.text_input(
            "Full Name",
            placeholder="Enter your full name"
        )

        email = st.text_input(
            "Email Address",
            placeholder="example@gmail.com"
        )

        mobile = st.text_input(
            "Mobile Number",
            placeholder="10 digit mobile number"
        )

        date_of_birth = st.date_input(
            "Date of Birth",
             min_value=date(1900, 1, 1),
             max_value=date.today(),
             value=date(2000, 1, 1),
             format="DD/MM/YYYY"
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Select",
                "Female",
                "Male",
                "Other",
                "Prefer not to say"
            ]
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimum 8 characters"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter password"
        )

    st.write("")

    if st.button(
        "Continue to Face Registration",
        use_container_width=True,
        type="primary"
    ):


        if not name.strip():

            st.error(
                "Please enter your full name."
            )
            st.stop()

        if not email.strip():

            st.error(
                "Please enter your email address."
            )
            st.stop()

        if "@" not in email or "." not in email:

            st.error(
                "Please enter a valid email address."
            )
            st.stop()

        cleaned_mobile = (
            mobile
            .replace(" ", "")
            .replace("-", "")
        )

        if not cleaned_mobile.isdigit():

            st.error(
                "Mobile number must contain only digits."
            )
            st.stop()

        if len(cleaned_mobile) != 10:

            st.error(
                "Mobile number must contain exactly 10 digits."
            )
            st.stop()

        if gender == "Select":

            st.error(
                "Please select your gender."
            )
            st.stop()

        if len(password) < 8:

            st.error(
                "Password must contain at least 8 characters."
            )
            st.stop()

        if password != confirm_password:

            st.error(
                "Passwords do not match."
            )
            st.stop()

 
        candidates = (
            candidate_database.get_all_candidates()
        )

        email_exists = any(
            candidate.get("email", "").strip().lower()
            == email.strip().lower()
            for candidate in candidates
        )

        if email_exists:

            st.error(
                "An account with this email already exists."
            )
            st.stop()
        try:

            password_salt, password_hash = hash_password(
                password
            )

            candidate = candidate_database.add_candidate(
                name=name.strip(),
                email=email.strip(),
                mobile=cleaned_mobile,
                date_of_birth=str(date_of_birth),
                gender=gender,
                password_salt=password_salt,
                password_hash=password_hash
            )

        except Exception as error:

            st.error(
                f"Unable to create account: {error}"
            )
            st.stop()

        st.session_state.registration_data = {
            "name": name.strip(),
            "email": email.strip(),
            "mobile": cleaned_mobile,
            "date_of_birth": str(date_of_birth),
            "gender": gender
        }

        st.session_state.candidate_id = (
            candidate["candidate_id"]
        )

        st.session_state.registration_started = True

        st.rerun()

    st.stop()

st.markdown(
    '<div class="section-title">Register Your Face</div>',
    unsafe_allow_html=True
)

st.info(
    "Position your face clearly in the camera and press "
    "Capture Face. You need 10 face samples. "
    "Liveness verification is not required during registration."
)


camera_col, progress_col = st.columns(
    [2, 1],
    gap="large"
)


with camera_col:

    st.markdown(
        '<div class="camera-card">',
        unsafe_allow_html=True
    )

    st.subheader("Face Camera")

    camera_image = st.camera_input(
        "Camera",
        key="face_registration_camera"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

with progress_col:

    st.markdown(
        '<div class="camera-card">',
        unsafe_allow_html=True
    )

    count = len(
        st.session_state.captured_images
    )

    st.markdown(
        f"""
        <div class="progress-number">
            {count}/10
        </div>

        <div class="progress-text">
            Face samples captured
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.progress(
        count / 10
    )

    st.write("")

    if count < 10:

        st.info(
            f"{10 - count} samples remaining."
        )

    else:

        st.success(
            "All 10 samples captured."
        )

    st.markdown(
        """
        <div class="capture-help">
        <br>
        For better registration:
        <br>• Look directly at the camera
        <br>• Keep your face clearly visible
        <br>• Change your angle slightly between samples
        <br>• Only one face should be visible
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

if (
    camera_image is not None
    and len(st.session_state.captured_images) < 10
):

    st.write("")

    if st.button(
        f"📷 Capture Face "
        f"{len(st.session_state.captured_images) + 1}/10",
        use_container_width=True,
        type="primary"
    ):

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

        try:

            bboxes, keypoints = (
                fast_detector.detect(
                    frame,
                    max_num=0
                )
            )

        except Exception as error:

            st.error(
                f"Face detection failed: {error}"
            )
            st.stop()
        if bboxes is None or len(bboxes) == 0:

            st.error(
                "NO FACE DETECTED. "
                "Please position your face clearly and try again."
            )

        elif len(bboxes) > 1:

            st.error(
                "MULTIPLE FACES DETECTED. "
                "Only one person should be visible."
            )

        else:

            bbox = bboxes[0]

            x1, y1, x2, y2 = (
                bbox[:4].astype(int)
            )

            height, width = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            face_width = x2 - x1
            face_height = y2 - y1

            if (
                face_width < 100
                or face_height < 100
            ):

                st.error(
                    "Face is too small. "
                    "Move closer to the camera."
                )

            else:

                pad_x = int(
                    face_width * 0.25
                )

                pad_y = int(
                    face_height * 0.25
                )

                crop_x1 = max(
                    0,
                    x1 - pad_x
                )

                crop_y1 = max(
                    0,
                    y1 - pad_y
                )

                crop_x2 = min(
                    width,
                    x2 + pad_x
                )

                crop_y2 = min(
                    height,
                    y2 + pad_y
                )

                face_crop = frame[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2
                ]

                if face_crop.size == 0:

                    st.error(
                        "Unable to crop the detected face."
                    )

                else:

                    sample_number = (
                        len(
                            st.session_state.captured_images
                        )
                        + 1
                    )

                    candidate_id = (
                        st.session_state.candidate_id
                    )

                    saved_path = (
                        capture_manager.save_image(
                            candidate_id,
                            face_crop,
                            sample_number
                        )
                    )

                    st.session_state.captured_images.append(
                        str(saved_path)
                    )

                    st.success(
                        f"Face sample "
                        f"{sample_number}/10 captured successfully."
                    )

                    st.rerun()


if st.session_state.captured_images:

    st.write("")

    st.markdown(
        '<div class="section-title">Captured Face Samples</div>',
        unsafe_allow_html=True
    )

    image_columns = st.columns(5)

    for index, image_path in enumerate(
        st.session_state.captured_images
    ):

        with image_columns[index % 5]:

            try:

                image = Image.open(
                    image_path
                )

                st.image(
                    image,
                    caption=f"Face {index + 1}",
                    width="stretch"
                )

            except Exception:

                st.warning(
                    f"Unable to display Face {index + 1}"
                )


if len(
    st.session_state.captured_images
) == 10:

    st.write("")

    st.success(
        "All 10 face samples have been captured."
    )

    st.info(
        "Capture is complete. "
        "The recognition model will now generate your "
        "facial embedding."
    )

    if st.button(
        "⚡ Process Faces & Complete Registration",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.processing_faces = True

        status_box = st.empty()

        progress_bar = st.progress(0)


        status_box.info(
            "Loading face recognition model..."
        )

        embedding_model = load_embedding_model()

        embeddings = []

        total_images = len(
            st.session_state.captured_images
        )

        for index, image_path in enumerate(
            st.session_state.captured_images
        ):

            status_box.info(
                f"Generating embedding "
                f"{index + 1}/{total_images}..."
            )

            image = cv2.imread(
                image_path
            )

            if image is None:

                st.session_state.processing_faces = False

                st.error(
                    f"Unable to read Face {index + 1}."
                )
                st.stop()

 
            faces = embedding_model.get(
                image
            )

            if len(faces) == 0:

                st.session_state.processing_faces = False

                st.error(
                    f"No face detected while processing "
                    f"Face {index + 1}."
                )
                st.stop()

            if len(faces) > 1:

                st.session_state.processing_faces = False

                st.error(
                    f"Multiple faces detected while processing "
                    f"Face {index + 1}."
                )
                st.stop()

            face = faces[0]

            if face.embedding is None:

                st.session_state.processing_faces = False

                st.error(
                    f"Could not generate an embedding for "
                    f"Face {index + 1}."
                )
                st.stop()

            embedding = np.asarray(
                face.embedding,
                dtype=np.float32
            ).reshape(-1)

 
            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:

                st.session_state.processing_faces = False

                st.error(
                    f"Invalid embedding generated for "
                    f"Face {index + 1}."
                )
                st.stop()

            embedding = (
                embedding / norm
            )

            embeddings.append(
                embedding
            )

            progress_bar.progress(
                (index + 1) / total_images
            )

 
        embeddings_array = np.vstack(
            embeddings
        ).astype(np.float32)

        final_embedding = np.mean(
            embeddings_array,
            axis=0
        )

 
        final_norm = np.linalg.norm(
            final_embedding
        )

        if final_norm == 0:

            st.session_state.processing_faces = False

            st.error(
                "Unable to create final face embedding."
            )
            st.stop()

        final_embedding = (
            final_embedding / final_norm
        ).astype(np.float32)


        candidate_id = (
            st.session_state.candidate_id
        )

        embedding_path = (
            EMBEDDING_DIR
            / f"{candidate_id}.npy"
        )

        np.save(
            embedding_path,
            final_embedding
        )

 
        candidate_database.update_candidate(
            candidate_id,
            {
                "face_images":
                    st.session_state.captured_images,

                "embedding_count":
                    10,

                "embedding_file":
                    str(embedding_path),

                "registration_status":
                    "completed",

                # Liveness is intentionally FALSE because
                # registration does not perform liveness.
                "liveness_verified":
                    False
            }
        )

        progress_bar.progress(1.0)

        status_box.success(
            "Face embeddings generated successfully."
        )

        st.session_state.processing_faces = False

        st.session_state.registration_finished = True

        st.rerun()