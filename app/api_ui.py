import hashlib
import hmac
import base64
import json
import uuid
import time
import requests
import chainlit as cl

HMAC_SECRET = "5z1OCulNuL8MA4qTzr9g9xRvM3VwiJSPHdmqXOgAqWM0mcYotxsWJQQJ99BEACHYHv6XJ3w3AAAAACOGm8RX"
API_URL = "http://localhost:8000/chat"
AGENT_NAME = "Chatbot"


# HMAC Signature Generator
def generate_signature(payload: dict, secret: str):
    nonce = "c2a6028e63cdbc74fe5cc6f537d452a3"
    timestamp = str(1747923485124)
    serialized_payload = json.dumps(payload, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    concatenated = nonce + timestamp + serialized_payload
    signature = hmac.new(secret.encode(), concatenated.encode(), hashlib.sha256).digest()
    signature_b64 = base64.b64encode(signature).decode()

    return {
        "nonce": nonce,
        "timestamp": timestamp,
        "signature_b64": signature_b64
    }


@cl.on_chat_start
async def start():
    # Each session starts with an empty chat history
    cl.user_session.set("chat_history", [])
    welcome = "Hi! I'm Sophie. How can I assist you today?"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    cl.user_session.get("chat_history").append({
        "user_type": "bot",
        "text": welcome,
        "timestamp": timestamp,
        "delay": 500,
        "contact_owner": AGENT_NAME
    })
    await cl.Message(content=welcome).send()


@cl.on_message
async def handle_message(message: cl.Message):
    user_text = message.content
    user_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    chat_history = cl.user_session.get("chat_history")

    # Append user message to history
    chat_history.append({
        "user_type": "visitor",
        "text": user_text,
        "timestamp": user_timestamp,
        "contact_owner": "Visitor"
    })

    # Prepare payload
    payload = {
        "history": chat_history,
        "message": user_text
    }

    # Sign payload
    sig = generate_signature(payload, HMAC_SECRET)

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "X-Nonce": sig["nonce"],
        "X-Timestamp": sig["timestamp"],
        "X-Signature": sig["signature_b64"]
    }

    try:
        # Send request
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        bot_reply = response.json().get("reply", "Sorry, I didn’t get a response.")

    except Exception as e:
        bot_reply = f"⚠️ API error: {e}"

    # Append bot response to history
    bot_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    chat_history.append({
        "user_type": "bot",
        "text": bot_reply,
        "timestamp": bot_timestamp,
        "delay": 1000,
        "contact_owner": AGENT_NAME
    })

    cl.user_session.set("chat_history", chat_history)


    # Send message to UI
    await cl.Message(content=bot_reply).send()