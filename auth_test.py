import hashlib
import base64
import json
import hmac
import time

# ------------------- Your verify_auth function -------------------
def verify_auth(payload: dict, nonce: str, timestamp: str, signature_b64: str, secret: str):
    try:
        message = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        concatenated_payload = nonce+timestamp+message

        expected_hmac = hmac.new(
            key=secret.encode(),
            msg=concatenated_payload.encode(),
            digestmod=hashlib.sha256
        ).digest()

        received_hmac = base64.b64decode(signature_b64)

        return hmac.compare_digest(received_hmac, expected_hmac)
    except Exception as e:
        print("Exception:", e)
        return False

# ------------------- Sample Test Client -------------------
# Shared secret key
API_SECRET = "5z1OCulNuL8MA4qTzr9g9xRvM3VwiJSPHdmqXOgAqWM0mcYotxsWJQQJ99BEACHYHv6XJ3w3AAAAACOGm8RX" 

# Fake request body (as Pydantic would model_dump())
payload = {
    "history": [
        {
            "user_type": "bot",
            "text": "Hello! I’m Sophie. I can help with",
            "timestamp": "2025-05-22T14:27:44.954Z",
            "delay": 500,
            "contact_owner": "Chatbot"
        },
        {
            "user_type": "bot",
            "text": "hello!",
            "timestamp": "2025-05-22T14:27:44.954Z",
            "delay": 2000,
            "contact_owner": "Chatbot"
        }
    ],
    "message": "Hey sophie!!@#$$%^&*(){}[]:;'></é“\n"
}

 

# Header fields just for testing
nonce = "c2a6028e63cdbc74fe5cc6f537d452a3"  # Should be unique per request
timestamp = str(int(1747923485124)) # UNIX timestamp

# Serialize and concatenate payload
message = json.dumps(payload, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
print(message)
concatenated = nonce+timestamp+message

# Create signature
signature = hmac.new(
    key=API_SECRET.encode(),
    msg=concatenated.encode(),
    digestmod=hashlib.sha256
).digest()

signature_b64 = base64.b64encode(signature).decode()

# ------------------- Test Verification -------------------
print("Generated Signature (Base64):", signature_b64)

is_valid = verify_auth(
    payload=payload,
    nonce=nonce,
    timestamp=timestamp,
    signature_b64=signature_b64,
    secret=API_SECRET
)

print("✅ Is Signature Valid?", is_valid)
