import streamlit as st
from pathlib import Path

from src.candidate_database import CandidateDatabase
from src.password_security import verify_password


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FaceVerify - Login",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DB = (
    PROJECT_ROOT
    / "data"
    / "candidates"
    / "candidates.json"
)


# ============================================================
# DATABASE
# ============================================================

candidate_database = CandidateDatabase(
    database_path=str(CANDIDATE_DB)
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "logged_in_candidate" not in st.session_state:
    st.session_state.logged_in_candidate = None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    .login-title {
        font-size: 38px;
        font-weight: 700;
        color: #172033;
        text-align: center;
        margin-top: 50px;
    }

    .login-subtitle {
        text-align: center;
        color: #667085;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .login-card {
        background-color: white;
        padding: 35px;
        border-radius: 20px;
        border: 1px solid #e4e7ec;
        box-shadow: 0 8px 30px rgba(16,24,40,0.08);
    }

    .security-text {
        text-align: center;
        color: #667085;
        font-size: 14px;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ALREADY LOGGED IN
# ============================================================

if st.session_state.logged_in:

    candidate = st.session_state.logged_in_candidate

    st.markdown(
        '<div class="login-title">Welcome Back</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-subtitle">'
        'You have successfully logged in to FaceVerify.'
        '</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"Welcome, {candidate['name']}!"
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.markdown(
            f"""
            ### Account Information

            **Name:** {candidate["name"]}

            **Email:** {candidate["email"]}

            **Candidate ID:** {candidate["candidate_id"]}

            **Registration Status:** {candidate.get(
                "registration_status",
                "Unknown"
            )}

            **Face Samples:** {candidate.get(
                "embedding_count",
                0
            )}
            """
        )

        st.write("")

        if st.button(
            "📷 Scan My Face",
            use_container_width=True,
            type="primary"
        ):

            st.switch_page(
                "pages/4_Face_Verification.py"
            )

        st.write("")

        if st.button(
            "Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.logged_in_candidate = None

            st.rerun()

    st.stop()


# ============================================================
# LOGIN HEADER
# ============================================================

st.markdown(
    '<div class="login-title">Welcome to FaceVerify</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="login-subtitle">'
    'Sign in securely to access your FaceVerify account.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOGIN FORM
# ============================================================

left, center, right = st.columns(
    [1, 1.2, 1]
)

with center:

    st.markdown(
        '<div class="login-card">',
        unsafe_allow_html=True
    )

    st.subheader("Secure Login")

    email = st.text_input(
        "Email Address",
        placeholder="Enter your registered email"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password"
    )

    st.write("")

    login_clicked = st.button(
        "🔐 Login",
        use_container_width=True,
        type="primary"
    )

    st.markdown(
        '<div class="security-text">'
        'Your password is securely verified using a salted '
        'password hash.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# LOGIN PROCESS
# ============================================================

if login_clicked:

    if not email.strip():

        st.error(
            "Please enter your email address."
        )
        st.stop()

    if not password:

        st.error(
            "Please enter your password."
        )
        st.stop()

    # --------------------------------------------------------
    # FIND ACCOUNT
    # --------------------------------------------------------

    candidate = (
        candidate_database.get_candidate_by_email(
            email
        )
    )

    if candidate is None:

        st.error(
            "Invalid email or password."
        )
        st.stop()

    # --------------------------------------------------------
    # CHECK PASSWORD DATA
    # --------------------------------------------------------

    stored_salt = candidate.get(
        "password_salt",
        ""
    )

    stored_hash = candidate.get(
        "password_hash",
        ""
    )

    if not stored_salt or not stored_hash:

        st.error(
            "This account does not have valid "
            "authentication credentials."
        )
        st.stop()

    # --------------------------------------------------------
    # VERIFY PASSWORD
    # --------------------------------------------------------

    try:

        password_valid = verify_password(
            password,
            stored_salt,
            stored_hash
        )

    except Exception:

        password_valid = False

    if not password_valid:

        st.error(
            "Invalid email or password."
        )
        st.stop()

    # --------------------------------------------------------
    # CHECK FACE REGISTRATION
    # --------------------------------------------------------

    if candidate.get(
        "registration_status"
    ) != "completed":

        st.warning(
            "Your account has not completed "
            "face registration yet."
        )
        st.stop()

    # --------------------------------------------------------
    # LOGIN SUCCESS
    # --------------------------------------------------------

    st.session_state.logged_in = True

    st.session_state.logged_in_candidate = candidate

    st.success(
        "Login successful."
    )

    st.rerun()