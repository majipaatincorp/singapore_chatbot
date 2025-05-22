import hashlib
import base64
import json
import hmac
from app.logger import logger


def verify_auth(payload: dict, nonce: str, timestamp: str, signature_b64: str, secret: str):
    """
    Verifies HMAC-SHA256 signature using nonce, timestamp, and full JSON payload.

    Args:
        payload (dict): The entire request body (e.g., {"message": ..., "history": [...]})
        nonce (str): Unique request nonce.
        timestamp (str): Timestamp string.
        signature_b64 (str): Base64-encoded HMAC-SHA256 signature.
        secret (str): Shared HMAC secret key.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        # Deterministically serialize JSON (sorted keys, no whitespace)
        serialized_payload  = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        concatenated_payload = nonce+timestamp+serialized_payload

        expected_signature  = hmac.new(
            key=secret.encode(),
            msg=concatenated_payload.encode(),
            digestmod=hashlib.sha256
        ).digest()

        received_signature  = base64.b64decode(signature_b64)

        return hmac.compare_digest(received_signature , expected_signature)
    except Exception as e:
        logger.error(f"Error verifying HMAC signature: {e}", exc_info=True)
        return False
