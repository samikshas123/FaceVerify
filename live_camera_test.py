import base64
from io import BytesIO

import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="FaceVerify Live Frame Test",
    page_icon="🔐",
    layout="centered",
)


# --------------------------------------------------
# HTML
# --------------------------------------------------

HTML = """
<div class="camera-wrapper">

    <div class="camera-header">

        <div>
            <div class="camera-title">FaceVerify</div>
            <div class="camera-subtitle">
                Live Biometric Camera
            </div>
        </div>

        <div id="status" class="status waiting">
            CAMERA STARTING
        </div>

    </div>


    <div class="camera-frame">

        <video
            id="camera"
            autoplay
            playsinline
            muted>
        </video>

        <div class="scan-line"></div>

        <div class="corner top-left"></div>
        <div class="corner top-right"></div>
        <div class="corner bottom-left"></div>
        <div class="corner bottom-right"></div>

        <div id="message" class="camera-message">
            Starting camera...
        </div>

    </div>

</div>
"""


# --------------------------------------------------
# CSS
# --------------------------------------------------

CSS = """
.camera-wrapper {
    width: 100%;
    max-width: 720px;
    margin: 20px auto;
    font-family: Arial, sans-serif;
}

.camera-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.camera-title {
    font-size: 26px;
    font-weight: 700;
}

.camera-subtitle {
    font-size: 14px;
    opacity: 0.65;
    margin-top: 3px;
}

.status {
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.status.waiting {
    background: #eeeeee;
    color: #555555;
}

.status.live {
    background: #e8f7ed;
    color: #16803c;
}

.status.error {
    background: #fdeaea;
    color: #c62828;
}

.camera-frame {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    overflow: hidden;
    border-radius: 22px;
    background: #111111;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.25);
}

.camera-frame video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1);
    display: block;
}

.camera-message {
    position: absolute;
    left: 50%;
    bottom: 25px;
    transform: translateX(-50%);
    padding: 9px 16px;
    border-radius: 20px;
    background: rgba(0, 0, 0, 0.65);
    color: white;
    font-size: 13px;
    white-space: nowrap;
}

.scan-line {
    position: absolute;
    left: 12%;
    right: 12%;
    top: 50%;
    height: 2px;
    background: rgba(255, 255, 255, 0.55);
    pointer-events: none;
}

.corner {
    position: absolute;
    width: 45px;
    height: 45px;
    border-color: white;
    border-style: solid;
    opacity: 0.9;
}

.top-left {
    top: 30px;
    left: 30px;
    border-width: 3px 0 0 3px;
}

.top-right {
    top: 30px;
    right: 30px;
    border-width: 3px 3px 0 0;
}

.bottom-left {
    bottom: 30px;
    left: 30px;
    border-width: 0 0 3px 3px;
}

.bottom-right {
    bottom: 30px;
    right: 30px;
    border-width: 0 3px 3px 0;
}
"""


# --------------------------------------------------
# JAVASCRIPT
# --------------------------------------------------

JS = """
export default function(component) {

    const {
        parentElement,
        setStateValue
    } = component;


    const video =
        parentElement.querySelector("#camera");

    const status =
        parentElement.querySelector("#status");

    const message =
        parentElement.querySelector("#message");


    let stream = null;
    let captureTimer = null;


    async function startCamera() {

        try {

            if (
                !navigator.mediaDevices ||
                !navigator.mediaDevices.getUserMedia
            ) {
                throw new Error(
                    "Camera API is not available."
                );
            }


            stream =
                await navigator.mediaDevices.getUserMedia({

                    video: {
                        facingMode: "user",
                        width: {
                            ideal: 640
                        },
                        height: {
                            ideal: 480
                        }
                    },

                    audio: false

                });


            video.srcObject = stream;


            status.textContent =
                "CAMERA LIVE";

            status.className =
                "status live";


            message.textContent =
                "Live camera active";


            setStateValue(
                "camera_status",
                "live"
            );


            /*
             * Wait until the browser knows the
             * actual video dimensions.
             */
            video.onloadedmetadata = () => {

                video.play();


                /*
                 * Temporary canvas.
                 *
                 * This does NOT save an image.
                 * It only creates a frame in memory.
                 */
                const canvas =
                    document.createElement("canvas");


                canvas.width = 320;
                canvas.height = 240;


                const context =
                    canvas.getContext("2d");


                /*
                 * Send one frame approximately
                 * every second.
                 */
                captureTimer =
                    setInterval(() => {

                        if (
                            video.readyState <
                            HTMLMediaElement.HAVE_CURRENT_DATA
                        ) {
                            return;
                        }


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
                                0.60
                            );


                        setStateValue(
                            "frame",
                            frame
                        );

                    }, 1000);

            };

        }

        catch (error) {

            console.error(
                "FaceVerify camera error:",
                error
            );


            status.textContent =
                "CAMERA ERROR";

            status.className =
                "status error";


            message.textContent =
                "Please allow camera access";


            setStateValue(
                "camera_status",
                "error"
            );

        }

    }


    startCamera();


    return () => {

        if (captureTimer) {
            clearInterval(captureTimer);
        }


        if (stream) {

            stream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

        }

    };

}
"""


# --------------------------------------------------
# CREATE COMPONENT
# --------------------------------------------------

live_camera = st.components.v2.component(
    "faceverify_live_frame_test",
    html=HTML,
    css=CSS,
    js=JS,
)


# --------------------------------------------------
# COMPONENT CALLBACKS
# --------------------------------------------------

def camera_status_changed():
    pass


def frame_changed():
    pass


# --------------------------------------------------
# MOUNT COMPONENT
# --------------------------------------------------

result = live_camera(

    default={
        "camera_status": "starting",
        "frame": "",
    },

    on_camera_status_change=camera_status_changed,
    on_frame_change=frame_changed,

    key="faceverify_live_camera",
)


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.title("FaceVerify Live Frame Test")

st.caption(
    "Live camera test. Frames are processed in memory "
    "and are not saved."
)


# --------------------------------------------------
# CAMERA STATUS
# --------------------------------------------------

if result.camera_status == "live":

    st.success(
        "LIVE CAMERA IS WORKING"
    )

elif result.camera_status == "error":

    st.error(
        "Camera access failed."
    )

else:

    st.info(
        "Starting live camera..."
    )


# --------------------------------------------------
# FRAME RECEIVED
# --------------------------------------------------

frame_data = result.frame


if frame_data:

    try:

        # Remove:
        # data:image/jpeg;base64,
        base64_data = frame_data.split(
            ",",
            1
        )[1]


        image_bytes = base64.b64decode(
            base64_data
        )


        image = Image.open(
            BytesIO(image_bytes)
        )


        st.success(
            "LIVE FRAME RECEIVED BY PYTHON"
        )


        st.image(
            image,
            caption="Temporary live frame",
            width=320,
        )


    except Exception as error:

        st.error(
            f"Frame received, but could not decode it: {error}"
        )

else:

    st.info(
        "Waiting for the first live frame..."
    )