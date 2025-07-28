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
    temperature=0.3,
    response_format={"type": "json_object"}
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
Hello! I’m Sophie. I can help with:
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

        # Try to read system prompt
    try:
        with open('app/system_prompt.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except (FileNotFoundError, IOError, PermissionError) as e:
        print(f"Failed to read system prompt file: {e}")


    if not prompt_template or prompt_template.strip() == "":
        print("System prompt is empty")


    system_prompt = prompt_template.format(services=services)

    # Try to read user prompt
    try:
        with open('app/user_prompt.txt', 'r', encoding='utf-8') as f:
            user_prompt_template = f.read()
    except (FileNotFoundError, IOError, PermissionError) as e:
        print(f"Failed to read user prompt file: {e}")

    if not user_prompt_template or user_prompt_template.strip() == "":
        print("User prompt template is empty")


    user_prompt = user_prompt_template.format(chat_transcript=chat_transcript, user_message= msg.content, context=context)



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
        reply = f"I'm sorry, something went wrong while generating the response.{e}"
        classification = "Unknown"
        confidence = 0.0
        contact = {}

    # Add assistant reply to history
    history.append({"role": "assistant", "content": reply})
    cl.user_session.set("history", history)

    await cl.Message(content=reply).send()
