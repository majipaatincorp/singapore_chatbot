import chainlit as cl
from dotenv import load_dotenv
import os
import json

# -------- LangChain & custom logic --------
from langchain_community.chat_models import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Initialize LLM
chat = AzureChatOpenAI(
    openai_api_base=os.environ.get("openai_api_base"),
    openai_api_version=os.environ.get("openai_api_version"),  
    deployment_name=os.environ.get("deployment_name"),   
    openai_api_key=os.environ.get("openai_api_key"),
    openai_api_type=os.environ.get("API_openai_api_typeKEY"),
    temperature=0.3
)

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./datasets/processed_total",
        embedding_function=embeddings
    )
    print(f"Loaded vector store with {vector_db._collection.count()} documents")
    return vector_db

# Load vector DB and retriever
vector_db = load_vector_store()
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    # Load the JSON file
    with open("output.json", "r") as f:
        data = json.load(f)
    cl.user_session.set("services", data)
    # Example: Print entire content
    await cl.Message("""
Hello! I’m Sophie.
I am your virtual assistant from InCorp Asia.
We provide these services:
📑 Company Secretarial
💰 Accounting and Finance
🧾 Payroll
🛂 Immigration
🏦 Funds Administration
🛡️ Risk Assurance
🔍 Recruitment
💼 Tax
🔄 Transfer Pricing
How can I assist you today?
""").send()

@cl.on_message
async def on_message(msg: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": msg.content})
    services = cl.user_session.get("services")
    

    # RAG Context
    docs = retriever.invoke(msg.content)
    context = "\n\n".join([doc.page_content for doc in docs])
    # Construct full conversation
    chat_transcript = "\n".join(
        [f"{turn['role'].capitalize()}: {turn['content']}" for turn in history]
    )

    # Prompt to the model
    system_prompt = f"""
You are Sophie, a warm, professional virtual assistant for InCorp Asia.

InCorp Asia is a leading corporate services provider offering end-to-end solutions including company incorporation, 
accounting, tax, payroll, work visa processing, fund structuring, and more. With over 8,000 legal entities served 
and deep expertise across various domains, InCorp simplifies business setup and compliance in Singapore, 
enabling clients to focus on growth and expansion across Asia.

🏢 InCorp Asia offers only these services:
service list: {services}
Please do not move forward with a services or topics not listed here.
strictly deny to provide service not in the service list.
(incorp doesnt deal with Immigration and any passes.)

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
2. “Could you share your email?”
3. “May I have your phone number too?”

Skip questions already answered.

---

📞 CLOSING:
- If all info is collected:
  → “Thanks for sharing your details! You’ll hear from our team within 24 hours. Meanwhile, I’m here if you need anything else.”
- If unqualified:
  → “Thanks for your interest! Let me know if I can assist you with anything else.”

---
KEYWORD CRITERIA:
- High-Medium Intent Keywords
	setup company, incorporate in Singapore, register company now, register business, need corporate secretary, arrange meeting, new company, business services, register, incorporate, urgent, launching, budget approved, pricing, costs, quotes, quotation, setup now, need, accounting, urgent, ASAP, require by, interested in, start new, business or company registration, need help, checking price
- Immigration service keywords
	hire staff, apply EP, immigration, need tax support, work pass, employment pass, work permit
- No Clear Intent
	Message fields empty or irrelevant or in a different language than English/Chinese, investment opportunity, opportunities, comparing price, compare, no budget
- Job Seek Intent
	years of experience, apply for the position, my resume, work experience, diploma, degree, my attached resume, internship, intern opportunities

---

🚫 GUARDRAILS:
- If asked about price/timeline:  
  → “That depends on your specific needs. Our team will follow up with more details.”
- Never guess, estimate, or mention competitors.
- Do not answer hiring/internal/unethical/off-topic questions.
- Redirect with: “That’s not something we handle, but I’d be happy to help with our core services.”
- Before suggesting solutions that involve extra steps, ask for user consent or confirmation, and do not assume the user agrees.
- Always prioritize respecting user intent and avoiding hallucinations that force unwanted options.

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
{msg.content}
and the keywords should be only from the role: user query from the conversation.

Context you may need:
{context}

Always reply in this JSON format:
{{
  "reply": "<your assistant reply>",
  "classification": "<Qualified | Unqualified | Not relevant>",
  "qualification_score": <0-30>,
  "lead_created": "<Yes | No>"(if name and email collectedd yes else no),
  "decisionMaker": "<Yes | No>",
  "timelineForIncorporation": "<Immediately | 30 Days | Not sure>",
  "Budget": "below $2000 | $2000–$5000 | above $5000>",
  "keywords": "[<keywords from the user message>, <keywords from the context>]", 
  "contact_info": {{
    "name": "<name if detected>",
    "email": "<email if detected>",
    "phone": "<phone if detected>"
  }}
}}
"""



    # Call LLM
    response = chat.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    # Parse response
    try:
        reply_data = eval(response.content)
        reply = reply_data["reply"]
        classification = reply_data["classification"]
        confidence = reply_data["qualification_score"]
        contact = reply_data["contact_info"]
        decisionMaker = reply_data["decisionMaker"]
        timelineForIncorporation = reply_data["timelineForIncorporation"]
        Budget = reply_data["Budget"]

        print(reply_data)


    except Exception as e:
        reply = "I'm sorry, something went wrong while generating the response."
        classification = "Unknown"
        confidence = 0.0
        contact = {}

    # Add assistant reply to history
    history.append({"role": "assistant", "content": reply})
    cl.user_session.set("history", history)

    await cl.Message(content=reply).send()
