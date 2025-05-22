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
  "message": "i want to start a business after 30 dayus, my budget is 10000 dollars and i am the decision maker but dont call me",
  "history": [
    {
      "user_type": "bot",
      "text": "Hello! I’m Sophie. I can help with:\n\n📌 Company Formation & Compliance \n📊 Accounting & Taxation \n📑 Corporate Secretarial Services \n💰 Finance, HR & Payroll \n✈️ Immigration Services \n🛡️ Risk Management \n🔏 IP, Trademarks & Business Advisory\n\nHow can I assist you today?",
      "timestamp": "2025-05-22T03:47:01.001Z",
      "delay": 500,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "Hi",
      "timestamp": "2025-05-22T03:47:04.387Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot",
      "text": "Hello! How can I assist you today?",
      "timestamp": "2025-05-22T03:47:04.387Z",
      "delay": 100,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "Need Dell amc quote in Singapore.",
      "timestamp": "2025-05-22T03:47:11.467Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot",
      "text": "Could you please provide more details about the Dell AMC you need a quote for in Singapore, such as the product model and service duration? This will help us offer you an accurate and tailored quote.",
      "timestamp": "2025-05-22T03:47:11.467Z",
      "delay": 100,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "Let me know if you support us",
      "timestamp": "2025-05-22T03:47:20.782Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot",
      "text": "To help you get started, may I have your name?",
      "timestamp": "2025-05-22T03:47:20.782Z",
      "delay": 2000,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "Could you please share email id I'll share Requirement details with you",
      "timestamp": "2025-05-22T03:47:30.946Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot",
      "text": "Thank you Could you please share email id I'll share Requirement details with you, next I would need your email address in case this chat disconnects.",
      "timestamp": "2025-05-22T03:47:30.946Z",
      "delay": 2000,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "Rishita Sharma. Email id - rishita.sharma@axiomtechnologies.com",
      "timestamp": "2025-05-22T03:47:41.914Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot_button",
      "text": "Which area of support are you looking for?",
      "timestamp": "2025-05-22T03:47:41.914Z",
      "delay": 2000,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "No one of the above",
      "timestamp": "2025-05-22T03:48:05.831Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot",
      "text": "Thank you for your response. Could you please specify the type of support or service you are looking for? This will help me assist you better. If you have any particular requirements or questions, feel free to share them.",
      "timestamp": "2025-05-22T03:48:05.831Z",
      "delay": 100,
      "contact_owner": "Chatbot"
    },
    {
      "user_type": "visitor",
      "text": "I'm looking annual maintaince service in Singapore for Dell product",
      "timestamp": "2025-05-22T03:49:25.917Z",
      "contact_owner": "Maheshwar Arulraj"
    },
    {
      "user_type": "bot",
      "text": "Could you please specify the Dell product model and the desired duration for the annual maintenance service in Singapore? This will help us provide you with an accurate and tailored quote.",
      "timestamp": "2025-05-22T03:49:25.917Z",
      "delay": 100,
      "contact_owner": "Chatbot"
    }
]
}

# Header fields just for testing
nonce = "e795e65add1b10adf6f2315e81fa26b7"  # Should be unique per request
timestamp = str(int(1747120885)) # UNIX timestamp

# Serialize and concatenate payload
message = json.dumps(payload, separators=(',', ':'), sort_keys=True)
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
