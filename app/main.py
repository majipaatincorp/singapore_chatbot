# main.py
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import logging
import sys

from langchain_community.chat_models import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import uvicorn
from app.auth_utils import verify_auth
from app.logger import logger

# ------------------ Setup Logging ------------------
# Configure root logger to print to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Print to terminal
    ]
)

# Create a logger for this module
app_logger = logging.getLogger(__name__)

# ------------------ Setup ------------------
load_dotenv()
app_logger.info("Environment variables loaded")

app = FastAPI()
app_logger.info("FastAPI application initialized")

# ------------------ Init LLM ------------------
app_logger.info("Initializing Azure OpenAI client...")
try:
    chat = AzureChatOpenAI(
        openai_api_base=os.environ.get("openai_api_base"),
        openai_api_version=os.environ.get("openai_api_version"),
        deployment_name=os.environ.get("deployment_name"),
        openai_api_key=os.environ.get("openai_api_key"),
        openai_api_type=os.environ.get("openai_api_type"),
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    app_logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    app_logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    raise

API_SECRET = os.environ.get("API_SECRET")
if API_SECRET:
    app_logger.info("API_SECRET loaded from environment")
else:
    app_logger.warning("API_SECRET not found in environment variables")

# ------------------ Load Vector Store ------------------
app_logger.info("Loading HuggingFace embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    app_logger.info("HuggingFace embeddings loaded successfully")
except Exception as e:
    app_logger.error(f"Failed to load HuggingFace embeddings: {e}")
    raise RuntimeError(f"Failed to initialize embeddings model: {e}")

app_logger.info("Loading Chroma vector database...")
try:
    vector_db = Chroma(
        persist_directory="./datasets/processed_total",
        embedding_function=embeddings
    )
    app_logger.info("Chroma vector database loaded successfully")
except Exception as e:
    app_logger.error(f"Failed to load Chroma vector database: {e}")
    raise RuntimeError(f"Failed to initialize vector database: {e}")

app_logger.info("Setting up document retriever...")
try:
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    doc_count = vector_db._collection.count()
    if doc_count == 0:
        app_logger.warning("Vector database is empty - no documents loaded")
    else:
        app_logger.info(f"Loaded vector store with {doc_count} documents")
    print(f"Loaded vector store with {doc_count} documents")
except Exception as e:
    app_logger.error(f"Failed to create retriever or access vector database: {e}")
    raise RuntimeError(f"Failed to setup document retriever: {e}")

# ------------------ Load Services ------------------
app_logger.info("Loading services configuration...")
try:
    with open("output.json", "r") as f:
        services = json.load(f)
    
    if not services:
        app_logger.warning("Services configuration is empty")
        services = {}
    else:
        app_logger.info(f"Services configuration loaded with {len(services)} services")
        
except Exception as e:
    app_logger.error(f"Error loading services configuration: {e}")
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

    app_logger.info("=== New chat request received ===")
    app_logger.info(f"Message length: {len(req.message) if req.message else 0}")
    app_logger.info(f"History length: {len(req.history) if req.history else 0}")
    
    try:
        # Check if headers are missing
        app_logger.info("Validating request headers...")
        if x_nonce is None:
            app_logger.error("Missing required header: x-nonce")
            raise HTTPException(
                status_code=400,
                detail="Missing required header: x-nonce"
            )
        
        if x_timestamp is None:
            app_logger.error("Missing required header: x-timestamp")
            raise HTTPException(
                status_code=400,
                detail="Missing required header: x-timestamp"
            )
        
        if x_signature is None:
            app_logger.error("Missing required header: x-signature")
            raise HTTPException(
                status_code=400,
                detail="Missing required header: x-signature"
            )
        
        app_logger.info("All required headers present")

        # Check if message is missing or empty
        app_logger.info("Validating request message...")
        if not req.message or req.message.strip() == "":
            app_logger.error("Missing or empty message in request")
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        # Check if history is None (though it has default value, extra safety)
        if req.history is None:
            app_logger.error("History is None")
            raise HTTPException(
                status_code=400,
                detail="History cannot be None"
            )

        history = req.history
        user_message = req.message
        app_logger.info(f"Processing message: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'")

        # Verify authentication
        app_logger.info("Verifying authentication...")
        is_valid = verify_auth(
            payload=req.model_dump(),
            nonce=x_nonce,
            timestamp=x_timestamp,
            signature_b64=x_signature,
            secret=API_SECRET
        )
        
        if not is_valid:
            app_logger.error("Unauthorized access: API key verification failed.")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized access: Authorization failed."
            )
        
        app_logger.info("Authentication successful")

        # Try to retrieve documents
        app_logger.info("Retrieving relevant documents from vector database...")
        docs = retriever.invoke(user_message)
        if not docs:
            app_logger.error("Failed to retrieve documents from vector database")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve relevant documents"
            )

        app_logger.info(f"Retrieved {len(docs)} relevant documents")
        context = "\n\n".join([doc.page_content for doc in docs])
        app_logger.info(f"Context length: {len(context)} characters")

        # Prepare chat transcript
        app_logger.info("Preparing chat transcript...")
        chat_transcript = "\n".join([f"{'chat' if msg['user_type'] == 'bot' or msg['user_type'] == 'bot_button' else 'Visitor'}: {msg['text'].strip()}" for msg in history])
        app_logger.info(f"Chat transcript length: {len(chat_transcript)} characters")

        # Try to read system prompt
        app_logger.info("Loading system prompt...")
        try:
            with open('app/system_prompt.txt', 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            app_logger.info("System prompt loaded successfully")
        except (FileNotFoundError, IOError, PermissionError) as e:
            app_logger.error(f"Failed to read system prompt file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load system prompt configuration"
            )

        if not prompt_template or prompt_template.strip() == "":
            app_logger.error("System prompt is empty")
            raise HTTPException(
                status_code=500,
                detail="System prompt configuration is empty"
            )

        system_prompt = prompt_template.format(services=services)
        app_logger.info(f"System prompt prepared, length: {len(system_prompt)} characters")

        # Try to read user prompt
        app_logger.info("Loading user prompt template...")
        try:
            with open('app/user_prompt.txt', 'r', encoding='utf-8') as f:
                user_prompt_template = f.read()
            app_logger.info("User prompt template loaded successfully")
        except (FileNotFoundError, IOError, PermissionError) as e:
            app_logger.error(f"Failed to read user prompt file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load user prompt configuration"
            )

        if not user_prompt_template or user_prompt_template.strip() == "":
            app_logger.error("User prompt template is empty")
            raise HTTPException(
                status_code=500,
                detail="User prompt configuration is empty"
            )

        user_prompt = user_prompt_template.format(chat_transcript=chat_transcript, user_message=user_message, context=context)
        app_logger.info(f"User prompt prepared, length: {len(user_prompt)} characters")

        # Call LLM
        app_logger.info("Invoking LLM...")
        try:
            response = chat.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            app_logger.info("LLM invocation successful")
        except Exception as e:
            app_logger.error(f"LLM invocation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response from AI model: {e}"
            )

        if not response or not response.content:
            app_logger.error("LLM returned empty response")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response from AI model"
            )

        app_logger.info(f"LLM response received, length: {len(response.content)} characters")

        # Parse LLM response
        app_logger.info("Parsing LLM response...")
        try:
            reply_data = json.loads(response.content)
            app_logger.info("LLM response parsed successfully")
        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to parse LLM response as JSON: {e}")
            app_logger.error(f"Raw response: {response.content}")
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from AI model"
            )

        reply = reply_data.get("reply", "")
        if not reply or reply.strip() == "":
            app_logger.error("Reply is empty in LLM response")
            raise HTTPException(
                status_code=500,
                detail="AI model returned empty reply"
            )

        app_logger.info(f"Reply extracted, length: {len(reply)} characters")

        # Extract response data
        app_logger.info("Extracting response data...")
        contact_info = reply_data.get("contact_info", {})
        email = contact_info.get("email", "")
        classification = reply_data.get("classification", "").lower()
        decisionMaker = reply_data.get("decisionMaker", "")
        Budget = reply_data.get("Budget", "")

        app_logger.info(f"Extracted data - Email: {'Yes' if email else 'No'}, Classification: {classification}, Decision Maker: {'Yes' if decisionMaker else 'No'}, Budget: {'Yes' if Budget else 'No'}")

        # Set sendToHubSpot flag
        if email != "" and classification == "qualified" and decisionMaker != "" and Budget != "":
            sendToHubspot = "Yes"
            app_logger.info("Setting sendToHubspot = Yes (all qualification criteria met)")
        else:
            sendToHubspot = "No"
            app_logger.info("Setting sendToHubspot = No (qualification criteria not met)")

        # Prepare final response
        final_response = {
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

        app_logger.info("=== Chat request completed successfully ===")
        return final_response

    except HTTPException as e:
        # Re-raise HTTP exceptions without modification
        app_logger.error(f"HTTP Exception: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        # Catch any other unexpected errors
        app_logger.exception(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred: {e}"
        )

# ------------------ Optional: Run with Uvicorn ------------------
if __name__ == "__main__":
    app_logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)