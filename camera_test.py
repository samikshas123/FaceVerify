import streamlit as st
import cv2
import numpy as np
from PIL import Image
from camera_input_live import camera_input_live

from insightface.app import FaceAnalysis


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="FaceVerify Camera Test",
    page_icon="📷",
    layout="centered"
)

st.title("FaceVerify - Live Face Detection Test")
st.write("Live camera test only. No verification photos are saved.")


# --------------------------------------------------
# LOAD INSIGHTFACE
# --------------------------------------------------

@st.cache_resource
def load_face_model():
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(
        ctx_id=0,
        det_size=(320, 320)
    )

    return model


face_model = load_face_model()


# --------------------------------------------------
# LIVE CAMERA
# --------------------------------------------------

image = camera_input_live()


if image is None:

    st.info("Please allow camera access.")

else:

    # Convert camera image to OpenCV format
    pil_image = Image.open(image).convert("RGB")

    frame = np.array(pil_image)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    # Detect faces
    faces = face_model.get(frame)


    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    if len(faces) == 0:

        st.warning("NO FACE DETECTED")

    elif len(faces) > 1:

        st.error("MULTIPLE FACES DETECTED")

    else:

        st.success("FACE DETECTED")

        face = faces[0]

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Convert back to RGB
        display_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            display_frame,
            caption="Live Face Detection",
            use_container_width=True
        )