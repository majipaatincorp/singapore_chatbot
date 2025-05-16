from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter, LongContextReorder,
)
import time 
from langchain.chat_models import AzureChatOpenAI
import os

load_dotenv()

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./datasets/processed",
        embedding_function=embeddings
    )
    print(f"Loaded vector store with {vector_db._collection.count()} documents")
    return vector_db

def mmr_hybrid_search(vector_db, k: int = 5):
    """MMR Hybrid Search combining similarity and MMR."""
    retriever_sim = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    retriever_mmr = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}
    )

    merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])

    # Compression pipeline
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[_filter, reordering])

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=merger
    )

    return compression_retriever

# 2. Create RAG Chain
def create_rag_chain(vector_db):
    prompt_template = """
    **Role**: You’re the Fallback Specialist for Incorp Asia’s chatbot. 
    The main assistant couldn’t handle the user’s question, so it’s your turn to provide a helpful response.

    **Chat History(latest on bottom)**:
    {history}

    **Latest User Question**:
    "{question}"
    
    **Your Task**:
    Step in with a helpful and relevant reply using the info and tone guidelines below. Keep it natural and professional, like you're having a conversation with someone — no robotic scripts. Do **not** start with phrases like:
    - "I understand..."
    - "Let me help you with that..."
    - "You're asking about..."

    **Your goal**:
     Help users by answering their queries accurately and responsibly. Follow these rules:
        1. If the user asks something that is relevant to the context provided below:
           - Answer clearly and helpfully based on the available information.
        2. If the query is **not directly relevant**, try to **relate it to Incorp Asia** (services, mission, values, or industry insight).
        3. If the user asks about competitors, similar companies, or comparisons:
           - Do **not** mention or acknowledge other companies.
           - Confidently redirect to Incorp Asia’s strengths and relevance in that context.
           - Always keep the focus on what Incorp Asia offers, how it can help, and why it's valuable — without referencing external entities.
        4. If the user asks about **illegal, unethical, or harmful content**:
           - Respond firmly:
             > “This type of request goes against Incorp Asia’s policy and ethical standards. I cannot help with that.”
        5. If the query is **completely unrelated to the company** (e.g., “What’s the capital of Brazil?”), provide a general response **politely and briefly**, and:
           - Mention that you specialize in assisting with Incorp Asia queries.
           - After that, respond:
             > “I’m here to assist with Incorp Asia-related queries. Let me know how I can help you with that!”
         
    **Important Notes**:
    - No robotic or scripted language.
    - Do **not** offer to connect to a company specialist — we don’t provide that.
    - Always sound conversational, helpful, and engaged.

    **Incorp Asia Areas**:
    Immigration • Company Formation • Tax & Accounting • Business Compliance • Advisory

    **Context Information**:
    {context}

    **Answer**:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # # Initialize LLM (replace with your API key)
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0.6
    # )
    # Create an instance of AzureChatOpenAI
    chat = AzureChatOpenAI(
        # openai_api_base="https://sanka-mani4jq3-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",
        openai_api_base=os.environ.get("openai_api_base"),
        openai_api_version=os.environ.get("openai_api_version"),  # or the version you're using
        deployment_name=os.environ.get("deployment_name"),   # name you gave when deploying the model in Azure
        openai_api_key=os.environ.get("openai_api_key"),
        openai_api_type=os.environ.get("API_openai_api_typeKEY")
    )


    # Create retrieval chain
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    # retriever = mmr_hybrid_search(vector_db=vector_db, k=5)

    # rag_chain = (
    #         {"context": retriever, "question": RunnablePassthrough()}
    #         | prompt
    #         | #llm
    #           chat
    # )
    #
    rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: x["history"]
    }
    | prompt
    | chat
)

    return rag_chain

# def _run_rag():
#     vector_db = load_vector_store()
#     rag_chain = create_rag_chain(vector_db)
#
#     while True:
#         question = input("\n Question:")
#         response = rag_chain.invoke(question)
#         print(f" Response: {response}")

def _run_rag():
    vector_db = load_vector_store()
    rag_chain = create_rag_chain(vector_db)

    # Simulated message list from frontend
    client_messages = [
        {"role": "user", "content": "I want to register a company in singapore"},
        {"role": "user", "content": "yes ,  also i Interested in ACS Course with Sponsorship & IPA ?"},
        {"role": "user", "content": "need any Sponsorship"},
        # {"role": "user", "content": "My work permit is rejected"},
        # {"role": "user", "content": "And black list how can I appeal?"},

    ]

    latest_question = client_messages[-1]["content"]

    formatted_history = "\n".join([
        f'{msg["content"]}'
        for msg in client_messages[:-1]
    ])
    print(formatted_history)

    inputs = {
        "question": latest_question,
        "history": formatted_history
    }

    response = rag_chain.invoke(inputs)
    print(f"\nAssistant: {response}")


if __name__ == "__main__":
    _run_rag()