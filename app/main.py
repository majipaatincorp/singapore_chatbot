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
    temperature=0.3,
    response_format={"type": "json_object"}
)


API_SECRET = os.environ.get("API_SECRET")

# ------------------ Load Vector Store ------------------
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to load HuggingFace embeddings: {e}")
    raise RuntimeError(f"Failed to initialize embeddings model: {e}")

try:
    vector_db = Chroma(
        persist_directory="./datasets/processed_total",
        embedding_function=embeddings
    )
except Exception as e:
    logger.error(f"Failed to load Chroma vector database: {e}")
    raise RuntimeError(f"Failed to initialize vector database: {e}")

try:
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    doc_count = vector_db._collection.count()
    if doc_count == 0:
        logger.warning("Vector database is empty - no documents loaded")
    print(f"Loaded vector store with {doc_count} documents")
except Exception as e:
    logger.error(f"Failed to create retriever or access vector database: {e}")
    raise RuntimeError(f"Failed to setup document retriever: {e}")

# ------------------ Load Services ------------------
try:
    with open("output.json", "r") as f:
        services = json.load(f)
    
    if not services:
        logger.warning("Services configuration is empty")
        services = {}
        
except Exception as e:
    logger.error(f"Error loading services configuration: {e}")
    raise RuntimeError(f"Failed to load services configuration: {e}")

# ------------------ Request Model ------------------
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Optional chat history for context

# ------------------ Endpoint ------------------
@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest, 
    x_nonce: str = Header(None, description='Unique request nonce'),
    x_timestamp: str = Header(None, description='UNIX timestamp'),
    x_signature: str = Header(None, description='Base64-encoded HMAC signature')):

    try:
        # Check if headers are missing
        if x_nonce is None:
            logger.error("Missing required header: x-nonce")
            raise HTTPException(
                status_code=400,
                detail="Missing required header: x-nonce"
            )
        
        if x_timestamp is None:
            logger.error("Missing required header: x-timestamp")
            raise HTTPException(
                status_code=400,
                detail="Missing required header: x-timestamp"
            )
        
        if x_signature is None:
            logger.error("Missing required header: x-signature")
            raise HTTPException(
                status_code=400,
                detail="Missing required header: x-signature"
            )

        # Check if message is missing or empty
        if not req.message or req.message.strip() == "":
            logger.error("Missing or empty message in request")
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        # Check if history is None (though it has default value, extra safety)
        if req.history is None:
            logger.error("History is None")
            raise HTTPException(
                status_code=400,
                detail="History cannot be None"
            )

        history = req.history
        user_message = req.message

        # Verify authentication
        is_valid = verify_auth(
            payload=req.model_dump(),
            nonce=x_nonce,
            timestamp=x_timestamp,
            signature_b64=x_signature,
            secret=API_SECRET
        )
        
        if not is_valid:
            logger.error("Unauthorized access: API key verification failed.")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized access: Authorization failed."
            )

        # Try to retrieve documents
        docs = retriever.invoke(user_message)
        if not docs:
            logger.error("Failed to retrieve documents from vector database")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve relevant documents"
            )

        context = "\n\n".join([doc.page_content for doc in docs])
        chat_transcript = "\n".join([f"{'Bot' if msg['user_type'] == 'bot' or msg['user_type'] == 'bot_button' else 'Visitor'}: {msg['text'].strip()}" for msg in history])

        # Try to read system prompt
        try:
            with open('app/system_prompt.txt', 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except (FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"Failed to read system prompt file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load system prompt configuration"
            )

        if not prompt_template or prompt_template.strip() == "":
            logger.error("System prompt is empty")
            raise HTTPException(
                status_code=500,
                detail="System prompt configuration is empty"
            )

        system_prompt = prompt_template.format(services=services)

        # Try to read user prompt
        try:
            with open('app/user_prompt.txt', 'r', encoding='utf-8') as f:
                user_prompt_template = f.read()
        except (FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"Failed to read user prompt file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load user prompt configuration"
            )

        if not user_prompt_template or user_prompt_template.strip() == "":
            logger.error("User prompt template is empty")
            raise HTTPException(
                status_code=500,
                detail="User prompt configuration is empty"
            )

        user_prompt = user_prompt_template.format(chat_transcript=chat_transcript, user_message=user_message, context=context)

        # Call LLM
        response = chat.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if not response or not response.content:
            logger.error("LLM returned empty response")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response from AI model"
            )

        # Parse LLM response (replacing dangerous eval())
        try:
            reply_data = json.loads(response.content)

        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from AI model"
            )

        reply = reply_data.get("reply", "")
        if not reply or reply.strip() == "":
            logger.error("Reply is empty in LLM response")
            raise HTTPException(
                status_code=500,
                detail="AI model returned empty reply"
            )

        contact_info = reply_data.get("contact_info", {})
        email = contact_info.get("email", "")
        classification = reply_data.get("classification", "").lower()
        decisionMaker = reply_data.get("decisionMaker", "")
        Budget = reply_data.get("Budget", "")

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
            "qualification_score": reply_data.get("qualification_score")
        }

    except HTTPException:
        # Re-raise HTTPExceptions as they are already properly formatted
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )

# ------------------ Optional: Run with Uvicorn ------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)