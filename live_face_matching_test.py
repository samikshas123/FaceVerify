import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v2 as components

from PIL import Image

from insightface.model_zoo import get_model
from insightface.app.common import Face
from insightface.utils import face_align


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FaceVerify - Live Face Matching",
    page_icon="🔐",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🔐 FaceVerify")
st.subheader("Live Face Matching Test")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


CANDIDATE_FILE = (
    BASE_DIR
    / "data"
    / "candidates"
    / "candidates.json"
)


MODEL_DIR = (
    Path.home()
    / ".insightface"
    / "models"
    / "buffalo_l"
)


DETECTION_MODEL_PATH = (
    MODEL_DIR
    / "det_10g.onnx"
)


RECOGNITION_MODEL_PATH = (
    MODEL_DIR
    / "w600k_r50.onnx"
)


# ============================================================
# SETTINGS
# ============================================================

MATCH_THRESHOLD = 0.50

FRAME_WIDTH = 480
FRAME_HEIGHT = 360

JPEG_QUALITY = 0.45

FRAME_INTERVAL_MS = 3000


# ============================================================
# CAMERA HTML
# ============================================================

CAMERA_HTML = """
<div id="camera-wrapper">

    <video
        id="camera"
        autoplay
        playsinline
        muted
    ></video>

    <canvas id="canvas"></canvas>

    <div id="camera-status">
        Starting camera...
    </div>

</div>
"""


# ============================================================
# CAMERA CSS
# ============================================================

CAMERA_CSS = """
#camera-wrapper {

    width: 100%;

    max-width: 760px;

    margin: auto;

    position: relative;

    background: #111;

    border-radius: 18px;

    overflow: hidden;

    border: 2px solid #333;
}


#camera {

    width: 100%;

    display: block;

    background: #111;

    transform: scaleX(-1);
}


#canvas {

    display: none;
}


#camera-status {

    position: absolute;

    left: 16px;

    bottom: 16px;

    padding: 8px 14px;

    border-radius: 20px;

    background: rgba(0, 0, 0, 0.75);

    color: white;

    font-size: 14px;
}
"""


# ============================================================
# CAMERA JAVASCRIPT
# ============================================================

CAMERA_JS = """
export default function(component) {

    const {
        setStateValue,
        parentElement
    } = component;


    const video =
        parentElement.querySelector("#camera");


    const canvas =
        parentElement.querySelector("#canvas");


    const status =
        parentElement.querySelector("#camera-status");


    async function startCamera() {

        try {

            const stream =
                await navigator.mediaDevices.getUserMedia({

                    video: {

                        width: {
                            ideal: 480
                        },

                        height: {
                            ideal: 360
                        },

                        facingMode: "user"

                    },

                    audio: false

                });


            video.srcObject = stream;


            await video.play();


            status.textContent =
                "🟢 LIVE CAMERA IS WORKING";


            setStateValue(
                "camera_status",
                "live"
            );


            canvas.width = 480;

            canvas.height = 360;


            const ctx =
                canvas.getContext("2d");


            function sendFrame() {

                if (
                    !video.videoWidth ||
                    !video.videoHeight
                ) {

                    return;

                }


                ctx.drawImage(

                    video,

                    0,

                    0,

                    480,

                    360

                );


                const imageData =
                    canvas.toDataURL(

                        "image/jpeg",

                        0.45

                    );


                setStateValue(

                    "frame",

                    imageData

                );

            }


            setTimeout(

                () => {

                    sendFrame();


                    setInterval(

                        sendFrame,

                        3000

                    );

                },

                1500

            );


        }

        catch (error) {

            console.error(error);


            status.textContent =
                "🔴 CAMERA ERROR";


            setStateValue(

                "camera_status",

                "error"

            );

        }

    }


    startCamera();

}
"""


# ============================================================
# REGISTER CAMERA COMPONENT
# ============================================================

live_camera_component = components.component(

    "faceverify_live_matching_camera",

    html=CAMERA_HTML,

    css=CAMERA_CSS,

    js=CAMERA_JS,

)


# ============================================================
# LOAD CANDIDATES
# ============================================================

@st.cache_data
def load_candidates():

    if not CANDIDATE_FILE.exists():

        return []


    try:

        with open(

            CANDIDATE_FILE,

            "r",

            encoding="utf-8"

        ) as file:

            data = json.load(file)


        if isinstance(data, list):

            return data


        return []


    except Exception:

        return []


# ============================================================
# GET COMPLETED CANDIDATE
# ============================================================

def get_completed_candidate():

    candidates = load_candidates()


    for candidate in candidates:

        if (
            candidate.get(
                "registration_status"
            )
            == "completed"
        ):

            return candidate


    return None


# ============================================================
# GET REGISTERED CANDIDATE
# ============================================================

candidate = get_completed_candidate()


if candidate is None:

    st.error(
        "❌ No completed registered account was found."
    )

    st.info(
        "Please register an account and capture "
        "10 face samples first."
    )

    st.stop()


candidate_id = candidate.get(
    "candidate_id",
    ""
)


candidate_name = candidate.get(
    "name",
    ""
)


embedding_file = candidate.get(
    "embedding_file",
    ""
)


# ============================================================
# CHECK EMBEDDING
# ============================================================

if not embedding_file:

    st.error(
        "❌ Registered account has no embedding file."
    )

    st.stop()


embedding_path = (
    BASE_DIR
    / embedding_file
)


if not embedding_path.exists():

    st.error(
        "❌ Registered embedding file was not found:\n"
        f"{embedding_path}"
    )

    st.stop()


# ============================================================
# LOAD REGISTERED EMBEDDING
# ============================================================

try:

    registered_embedding = np.load(
        str(embedding_path)
    ).astype(np.float32)


    registered_embedding = (
        registered_embedding
        /
        (
            np.linalg.norm(
                registered_embedding
            )
            + 1e-10
        )
    )


except Exception as error:

    st.error(
        "❌ Could not load registered embedding."
    )

    st.exception(error)

    st.stop()


# ============================================================
# REGISTERED ACCOUNT
# ============================================================

st.success(
    f"Registered account found: "
    f"**{candidate_name}** ({candidate_id})"
)


st.write(
    "The live camera will compare the face "
    "with this registered face."
)


st.caption(
    "🔒 No verification photo is saved to disk."
)


# ============================================================
# CHECK MODELS
# ============================================================

if not DETECTION_MODEL_PATH.exists():

    st.error(
        "❌ Detection model not found:\n"
        f"{DETECTION_MODEL_PATH}"
    )

    st.stop()


if not RECOGNITION_MODEL_PATH.exists():

    st.error(
        "❌ Recognition model not found:\n"
        f"{RECOGNITION_MODEL_PATH}"
    )

    st.stop()


# ============================================================
# LOAD DETECTOR
# ============================================================

@st.cache_resource
def load_detector():

    detector = get_model(

        str(DETECTION_MODEL_PATH),

        providers=[
            "CPUExecutionProvider"
        ],

    )


    detector.prepare(

        ctx_id=-1,

        input_size=(320, 320),

    )


    return detector


# ============================================================
# LOAD RECOGNITION MODEL
# ============================================================

@st.cache_resource
def load_recognition_model():

    model = get_model(

        str(RECOGNITION_MODEL_PATH),

        providers=[
            "CPUExecutionProvider"
        ],

    )


    model.prepare(

        ctx_id=-1

    )


    return model


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(

    embedding1,

    embedding2

):

    embedding1 = np.asarray(

        embedding1,

        dtype=np.float32

    ).reshape(-1)


    embedding2 = np.asarray(

        embedding2,

        dtype=np.float32

    ).reshape(-1)


    norm1 = np.linalg.norm(
        embedding1
    )


    norm2 = np.linalg.norm(
        embedding2
    )


    if norm1 <= 1e-10:

        return 0.0


    if norm2 <= 1e-10:

        return 0.0


    embedding1 = (
        embedding1 / norm1
    )


    embedding2 = (
        embedding2 / norm2
    )


    return float(
        np.dot(
            embedding1,
            embedding2
        )
    )


# ============================================================
# START LIVE CAMERA
# ============================================================

live_camera = live_camera_component(

    default={

        "camera_status": "starting",

        "frame": "",

    },

    on_camera_status_change=lambda: None,

    on_frame_change=lambda: None,

    key="faceverify_live_matching_camera",

)


camera_status = live_camera.get(
    "camera_status",
    "starting"
)


frame_data = live_camera.get(
    "frame",
    ""
)


# ============================================================
# CAMERA STATUS
# ============================================================

if camera_status == "live":

    st.success(
        "🟢 LIVE CAMERA IS WORKING"
    )


elif camera_status == "error":

    st.error(
        "🔴 CAMERA ERROR"
    )


# ============================================================
# PROCESS LIVE FRAME
# ============================================================

if frame_data:

    st.success(
        "🟢 LIVE FRAME RECEIVED BY PYTHON"
    )


    try:

        # ====================================================
        # STEP 1
        # DECODE LIVE FRAME
        # ====================================================

        st.info(
            "Step 1: Decoding live frame..."
        )


        if "," in frame_data:

            encoded_image = (
                frame_data.split(
                    ",",
                    1
                )[1]
            )

        else:

            encoded_image = frame_data


        image_bytes = base64.b64decode(
            encoded_image
        )


        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        bgr_image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )


        st.success(
            "Step 1 successful: "
            "Live frame decoded."
        )


        # ====================================================
        # DISPLAY TEMPORARY FRAME
        # ====================================================

        st.image(
            pil_image,
            caption="Latest Live Camera Frame",
            width=520,
        )


        st.caption(
            "This frame is held temporarily in memory. "
            "It is NOT saved as a verification photo."
        )


        # ====================================================
        # STEP 2
        # DETECTOR
        # ====================================================

        with st.spinner(
            "Loading face detector..."
        ):

            detector = load_detector()


        st.success(
            "Step 2 successful: "
            "Face detector is ready."
        )


        # ====================================================
        # STEP 3
        # DETECTION
        # ====================================================

        st.info(
            "Step 3: Detecting faces..."
        )


        detection_start = (
            time.perf_counter()
        )


        bboxes, kpss = detector.detect(
            bgr_image,
            max_num=0
        )


        detection_time = (
            time.perf_counter()
            - detection_start
        )


        st.write(
            f"Detection time: "
            f"**{detection_time:.2f} seconds**"
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if (
            bboxes is None
            or len(bboxes) == 0
        ):

            st.warning(
                "🟡 NO FACE"
            )


            st.info(
                "No person is currently "
                "detected in front of the camera."
            )


        # ====================================================
        # MULTIPLE FACES
        # ====================================================

        elif len(bboxes) > 1:

            st.warning(
                "🟠 MULTIPLE FACES"
            )


            st.info(
                f"{len(bboxes)} faces were detected. "
                "Only one person should be in front "
                "of the camera."
            )


        # ====================================================
        # ONE FACE
        # ====================================================

        else:

            st.success(
                "🟢 ONE FACE DETECTED"
            )


            # =================================================
            # BOUNDING BOX
            # =================================================

            bbox = bboxes[0]


            x1, y1, x2, y2 = (
                bbox[:4].astype(int)
            )


            # =================================================
            # DETECTION CONFIDENCE
            # =================================================

            confidence = float(
                bbox[4]
            )


            st.write(
                f"Detection confidence: "
                f"**{confidence:.3f}**"
            )


            # =================================================
            # LANDMARKS
            # =================================================

            if (
                kpss is None
                or len(kpss) == 0
            ):

                st.error(
                    "❌ Face landmarks were not returned."
                )

                st.stop()


            face_landmarks = np.asarray(
                kpss[0],
                dtype=np.float32
            )


            st.write(
                "Facial landmarks detected: "
                f"**{face_landmarks.shape}**"
            )


            # =================================================
            # DRAW FACE BOX
            # =================================================

            display_image = (
                bgr_image.copy()
            )


            cv2.rectangle(
                display_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )


            display_image_rgb = cv2.cvtColor(
                display_image,
                cv2.COLOR_BGR2RGB
            )


            st.image(
                display_image_rgb,
                caption="Detected Face",
                width=520,
            )


            # =================================================
            # STEP 4
            # LOAD RECOGNITION MODEL
            # =================================================

            st.info(
                "Step 4: Loading direct face "
                "recognition model..."
            )


            model_start = (
                time.perf_counter()
            )


            recognition_model = (
                load_recognition_model()
            )


            model_load_time = (
                time.perf_counter()
                - model_start
            )


            st.success(
                "Step 4 successful: "
                "Direct face recognition model is ready."
            )


            st.write(
                f"Recognition model load time: "
                f"**{model_load_time:.2f} seconds**"
            )


            # =================================================
            # STEP 5
            # FACE ALIGNMENT
            # =================================================

            st.info(
                "Step 5: Aligning face for recognition..."
            )


            alignment_start = (
                time.perf_counter()
            )


            # ArcFace expects a 112 x 112 aligned face.
            #
            # We use the five landmarks detected by SCRFD.
            #

            input_size = (
                int(
                    recognition_model.input_size[0]
                )
            )


            aligned_face = (
                face_align.norm_crop(
                    bgr_image,
                    landmark=face_landmarks,
                    image_size=input_size,
                    mode="arcface",
                )
            )


            alignment_time = (
                time.perf_counter()
                - alignment_start
            )


            st.success(
                "Step 5 successful: "
                "Face alignment completed."
            )


            st.write(
                f"Alignment time: "
                f"**{alignment_time:.4f} seconds**"
            )


            # =================================================
            # SHOW ALIGNED FACE
            # =================================================

            aligned_face_rgb = cv2.cvtColor(
                aligned_face,
                cv2.COLOR_BGR2RGB
            )


            st.image(
                aligned_face_rgb,
                caption="Aligned Face Used for Recognition",
                width=224,
            )


            # =================================================
            # STEP 6
            # DIRECT EMBEDDING
            # =================================================

            st.info(
                "Step 6: Generating live face embedding..."
            )


            embedding_start = (
                time.perf_counter()
            )


            # IMPORTANT:
            #
            # Do NOT use:
            #
            # recognition_model.get(...)
            #
            # Instead we directly call get_feat().
            #
            # This sends the already-aligned 112x112
            # face directly to the ArcFace ONNX model.
            #

            live_embedding = (
                recognition_model.get_feat(
                    aligned_face
                )
            )


            embedding_time = (
                time.perf_counter()
                - embedding_start
            )


            st.write(
                f"Embedding generation time: "
                f"**{embedding_time:.4f} seconds**"
            )


            # =================================================
            # CHECK EMBEDDING
            # =================================================

            if live_embedding is None:

                st.error(
                    "❌ Live face embedding was not generated."
                )

                st.stop()


            live_embedding = np.asarray(
                live_embedding,
                dtype=np.float32
            ).reshape(-1)


            if live_embedding.size == 0:

                st.error(
                    "❌ Live face embedding is empty."
                )

                st.stop()


            # =================================================
            # NORMALIZE
            # =================================================

            live_norm = np.linalg.norm(
                live_embedding
            )


            if live_norm <= 1e-10:

                st.error(
                    "❌ Live face embedding is invalid."
                )

                st.stop()


            live_embedding = (
                live_embedding / live_norm
            )


            st.success(
                "Step 6 successful: "
                "Live face embedding generated."
            )


            st.write(
                f"Embedding dimensions: "
                f"**{live_embedding.shape[0]}**"
            )


            # =================================================
            # STEP 7
            # COMPARE FACES
            # =================================================

            st.info(
                "Step 7: Comparing with registered face..."
            )


            comparison_start = (
                time.perf_counter()
            )


            similarity = cosine_similarity(
                registered_embedding,
                live_embedding
            )


            comparison_time = (
                time.perf_counter()
                - comparison_start
            )


            st.write(
                f"Comparison time: "
                f"**{comparison_time:.4f} seconds**"
            )


            st.write(
                f"Face similarity: "
                f"**{similarity:.4f}**"
            )


            st.write(
                f"Matching threshold: "
                f"**{MATCH_THRESHOLD:.2f}**"
            )


            # =================================================
            # FINAL RESULT
            # =================================================

            if similarity >= MATCH_THRESHOLD:

                st.success(
                    "✅ FACE DETECTED"
                )


                st.write(
                    "The live face matches "
                    "the registered face."
                )


            else:

                st.error(
                    "❌ NO MATCH"
                )


                st.write(
                    "The live face does not match "
                    "the registered face."
                )


    except Exception as error:

        st.error(
            "❌ Error while processing "
            "the live camera frame."
        )


        st.exception(error)


# ============================================================
# PRIVACY
# ============================================================

st.divider()


st.caption(
    "🔒 Privacy: The live verification frame "
    "is processed temporarily in memory. "
    "No verification photo is saved to disk."
)