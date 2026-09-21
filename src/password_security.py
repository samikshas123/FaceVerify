import hashlib
import hmac
import secrets


ITERATIONS = 310000


def hash_password(password):
    """
    Create a secure salted password hash.
    """

    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ITERATIONS
    )

    return (
        salt.hex(),
        password_hash.hex()
    )


def verify_password(
    password,
    salt_hex,
    stored_hash_hex
):
    """
    Verify a password against
    its stored salted hash.
    """

    salt = bytes.fromhex(
        salt_hex
    )

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ITERATIONS
    )

    return hmac.compare_digest(
        password_hash.hex(),
        stored_hash_hex
    )