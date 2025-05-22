import hashlib
import base64
import json
import hmac



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
        message = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        concatenated_payload = f"{nonce}:{timestamp}:{message}"

        expected_hmac = hmac.new(
            key=secret.encode(),
            msg=concatenated_payload.encode(),
            digestmod=hashlib.sha256
        ).digest()

        received_hmac = base64.b64decode(signature_b64)

        return hmac.compare_digest(received_hmac, expected_hmac)
    except Exception:
        return False
