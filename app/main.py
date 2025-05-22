# main.py
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

from langchain_community.chat_models import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import uvicorn
from app.auth_utils import verify_auth
from app.logger import logger


# ------------------ Setup ------------------
load_dotenv()

app = FastAPI()

# ------------------ Init LLM ------------------
chat = AzureChatOpenAI(
    openai_api_base=os.environ.get("openai_api_base"),
    openai_api_version=os.environ.get("openai_api_version"),
    deployment_name=os.environ.get("deployment_name"),
    openai_api_key=os.environ.get("openai_api_key"),
    openai_api_type=os.environ.get("openai_api_type"),
    temperature=0.3
)

API_SECRET = os.environ.get("API_SECRET")

# ------------------ Load Vector Store ------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./datasets/processed_total",
    embedding_function=embeddings
)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
print(f"Loaded vector store with {vector_db._collection.count()} documents")

# ------------------ Load Services ------------------
with open("output.json", "r") as f:
    services = json.load(f)

# ------------------ Request Model ------------------
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Optional chat history for context

# ------------------ Endpoint ------------------
@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest, 
    x_nonce: str = Header(..., description='Unique request nonce'),
    x_timestamp: str = Header(..., description='UNIX timestamp'),
    x_signature: str = Header(..., description='Base64-encoded HMAC signature')):

    history = req.history
    user_message = req.message

    is_valid = verify_auth(
        payload=req.model_dump(),
        nonce=x_nonce,
        timestamp=x_timestamp,
        signature_b64=x_signature,
        secret=API_SECRET
    )
    # verifying authentication
    if not is_valid:
        logger.error("Unauthorized access: API key verification failed.")
        raise HTTPException(
            status_code = 401,
            detail = "Unauthorized access: Authorization failed."
        )

    try:
        # RAG context
        docs = retriever.invoke(user_message)
        context = "\n\n".join([doc.page_content for doc in docs])

        chat_transcript = "\n".join([f"{'Bot' if msg['user_type'] == 'bot' or msg['user_type'] == 'bot_button' else 'Visitor'}: {msg['text'].strip()}" for msg in history])
    except Exception as e:
        # Log the error with traceback
        logger.error(f"Error while generating RAG context or chat transcript {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing the request.")


    # System + user prompt
    system_prompt = f"""
You are Sophie, a warm, professional virtual assistant for InCorp Asia.

InCorp Asia is a leading corporate services provider offering end-to-end solutions including company incorporation, 
accounting, tax, payroll, work visa processing, fund structuring, and more. With over 8,000 legal entities served 
and deep expertise across various domains, InCorp simplifies business setup and compliance in Singapore, 
enabling clients to focus on growth and expansion across Asia.

🏢 InCorp Asia offers only these services:
service list: {services}
Please do not move forward with any other services or topics not listed here.

🎯 Your job is to guide users, answer questions, and qualify promising leads — without assuming, pressuring, or hallucinating.

---

🧭 CONVERSATION FLOW:

1. **Acknowledge & Clarify**
   - Greet warmly.
   - Ask up to 2 open-ended questions to understand the user's needs.
   - If vague (e.g., “need help” / “interested”), ask:  
     “Could you clarify what you’re looking to do, or which service you’re interested in?”

2. **Present Service Options**
   - Use **numbered bullets**, one per line.
   - No inline lists or grouping.
   - Example:
     1. 📌 Company Formation  
     2. 📑 Secretarial  
     3. 💰 Accounting  
     4. 🧾 Payroll  
     5. ❓Something else?

3. **If user gives only a service name (e.g., “Payroll”)**, ask:
   “Sure! Could you tell me a bit more about what you're planning with [Service]?”

---

### 📊 QUALIFICATION ( Once per session.)
Start qualification only if the context is relevant enough, the client is sure that he wants the service and the service is in the service list.
- For **Company Formation**, ask:
  1. When are you planning to incorporate?
     - Immediately (+10), 30 Days (+5), Not sure (0)
  2. Budget?
     - above $2000 (0), $2000–$5000 (+5), below $5K (+10)
  3. Are you the decision maker? (Yes = +10, No = 0)

- For **other services**:
  1. Budget?
  2. Are you the decision maker?

- Ask these **one at a time**, never combine multiple questions.

- 📌 **Always present answer choices as numbered options** (e.g., 1, 2, 3)  
  and invite the user to choose the one that best fits their situation.  
  Use natural, friendly phrasing like:  
  _“\nLet me know which option suits you best. Just reply with the number.”_

- ❗️Do **not continue** unless each answer is provided clearly.

- ✅ If score is **≥ 15 for Company Formation** or **≥ 10 for other services**, proceed to **Lead Info**.

- 🚫 **Never reveal** qualification scores or status  
  (e.g., “you are qualified” or “your score is…”).

---

📥 LEAD INFO COLLECTION (If qualified):
Ask one at a time:
1. “May I know your name?”
2. “May I have your phone number?”
3. “Could you share your email too?” (verify format)

Skip questions already answered.

---

📞 CLOSING:
- If all info is collected:
  → “Thanks for sharing your details! You’ll hear from our team within 24 hours. Meanwhile, I’m here if you need anything else.”
- If unqualified:
  → “Thanks for your interest! Let me know if I can assist you with anything else.”

---

🚫 GUARDRAILS:
- If asked about price/timeline:  
  → “That depends on your specific needs. Our team will follow up with more details.”
- Never guess, estimate, or mention competitors.
- Do not answer hiring/internal/unethical/off-topic questions.
- Redirect with: “That’s not something we handle, but I’d be happy to help with our core services.”
- Before suggesting solutions that involve extra steps, ask for user consent or confirmation, and do not assume the user agrees.
- Always prioritize respecting user intent and avoiding hallucinations that force unwanted options.
- Only respond to the user query that are in English. If the user query is in a different language, 
Strictly respond with "Currently, we support English language only. Kindly submit your questions in English. Thank you for your understanding."
---

🧠 TRACK (Internal use only):
- Qualified / Unqualified
- Contact Info Collected

---

📌 FORMAT RULES:
- Use emojis for bullets.
- One follow-up or option per line.
- Keep responses short (max 2 lines per bullet, no dense paragraphs).
- Stay in character as Sophie. Never say you’re not human.

"""
    user_prompt = f"""
This is the conversation so far:
{chat_transcript}

User's latest message:
{user_message}
and the keywords should be only from the role: user query from the conversation.

Context you may need:
{context}

Always reply in this JSON format:
{{
  "reply": "<your assistant reply>",
  "classification": "<Qualified | Unqualified | Not relevant>",
  "qualification_score": <0-30>,
  "lead_created": "<Yes | No>"(if name and email collected yes else no),
  "decisionMaker": "<Yes | No>",
  "timelineForIncorporation": "<Immediately | 30 Days | Not sure>",
  "Budget": "<below $2000 | $2000–$5000 | above $5000>",
  "keywords": "[<keywords from the role Visitor's message>, <keywords from the role Visitor's context>]", 
  "shouldYouContact": "<Yes | No>"(Default Yes, if the user does not want to be contacted, set it to No),
  "contact_info": {{
    "name": "<name>",
    "email": "<email>",
    "phone": "<phone>"
  }}
}}
"""

    # Call LLM
    try:
      response = chat.invoke([
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ])
    except Exception as e:
        logger.error("Error invoking chat model", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing the request.")

    try:
        reply_data = eval(response.content)
        reply = reply_data["reply"]
        contact_info = reply_data.get("contact_info", {})
        email = contact_info.get("email")
        classification = reply_data.get("classification", "").lower()
        decisionMaker = reply_data.get("decisionMaker")
        Budget =  reply_data.get("Budget")

        # Set sendToHubSpot flag
        if email != "" and classification == "qualified" and decisionMaker != "" and Budget != "":
            sendToHubspot = "Yes"
        else:
            sendToHubspot = "No"


        return {
          "reply": reply,
          "attributes": {
              "decisionMaker": reply_data.get("decisionMaker"),
              "timelineForIncorporation": reply_data.get("timelineForIncorporation"),
              "Budget": reply_data.get("Budget")
          },
          "keywords": reply_data.get("keywords"),
          "contact_info": reply_data.get("contact_info"),
          "sendToHubspot": sendToHubspot,
          "shouldYouContact": reply_data.get("shouldYouContact"),
        }

    except Exception as e:
        logger.exception(f"Failed to parse LLM response\n error: {e}")
        return {
            "reply": "Sorry, something went wrong please try again later.",
            "error": str(e),
        }
    

# ------------------ Optional: Run with Uvicorn ------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
