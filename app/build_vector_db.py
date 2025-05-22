from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
import glob
from tqdm import tqdm
from typing import List
from langchain.schema import Document
import shutil
from preprocessing import process_immigration_doc
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter, LongContextReorder,
)


# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight but effective
# PERSIST_DIR = "./datasets/processed"  # Where Chroma will store data
PERSIST_DIR = "./datasets/processed_total"  # Where Chroma will store data


def get_embedding_device():
    """Returns the device to use for embedding based on CUDA availability."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def embedding_models():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def create_vector_db(documents: List[Document]):
    """Creates and persists Chroma DB with embeddings."""
    embeddings = embedding_models()

    # Ensure persistence directory is clean
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    if not documents:
        raise ValueError("No valid documents to process.")

    # Create Vector Store
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    # Persist to disk
    vector_db.persist()
    print(f"Vector DB created at {os.path.abspath(PERSIST_DIR)}")
    return vector_db


def process_all_documents(md_files):
    """Processes all markdown files into chunks for vector DB."""
    all_chunks = []
    for file_path in tqdm(md_files, desc="Processing Documents"):
        all_chunks.extend(process_immigration_doc(file_path))

    print(f"Total documents processed: {len(all_chunks)}")
    return all_chunks


def db_created():
    """Creates the vector DB if it doesn't exist."""
    print("Creating new vector DB...")
    # md_files = glob.glob("./datasets/raw_data/*.md")
    md_files = glob.glob("./datasets/total/*.md")

    if not md_files:
        raise FileNotFoundError("No markdown files found in the specified directory.")

    print(f"Found {len(md_files)} markdown files")
    all_chunks = process_all_documents(md_files)
    vector_db = create_vector_db(all_chunks)

    return vector_db


def similarity_search(vector_db, query: str, k: int = 5):
    """Perform a similarity search on the vector DB."""
    results = vector_db.similarity_search(query, k=k)
    print(f"Similarity Search Results (Top {k}):")
    for doc in results:
        print(f"\nFrom {doc.metadata['source']}:\n{doc.page_content}")  # Truncated for clarity


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
    _filter = EmbeddingsRedundantFilter(embeddings=embedding_models())
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[_filter, reordering])

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=merger
    )

    # context = compression_retriever.get_relevant_documents(query)
    # print(f"MMR Search Results (Top {k}):")
    # for doc in context:
    #     print(f"\nFrom {doc.metadata['source']}:\n{doc.page_content}")  # Truncated for clarity
    # return context
    return compression_retriever


# if __name__ == "__main__":
#     db_created()

    # query = "What are EP salary requirements?"
    #
    # # Test similarity search
    # print("-------Similarity Search------")
    # similarity_search(vector_db, query)
    #
    # # Test MMR hybrid search
    # print("--------MMR Hybrid Search-------")
    # mmr_hybrid_search(vector_db, query)

db_created()
