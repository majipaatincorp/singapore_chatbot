import glob
from tqdm import tqdm
from preprocessing import process_immigration_doc
import pandas as pd
from sentence_transformers import SentenceTransformer
from azure.search.documents import SearchClient
from azure.core.exceptions import ResourceExistsError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearchProfile,
    VectorSearch,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SemanticPrioritizedFields,
    HnswParameters,
    HnswAlgorithmConfiguration,
    SearchSuggester
)

from dotenv import load_dotenv
import os

load_dotenv()


# connection creds
endpoint = os.environ.get("endpoint")
key = os.environ.get("key")
index_name = "chatbot_knowledge"
credential = AzureKeyCredential(key)




# Initialize the sentence transformer model for generating embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
model = SentenceTransformer(EMBEDDING_MODEL)

# Process each markdown file with a progress bar
def process_all_documents(md_files):
    """
    Processes all markdown files into chunks suitable for vector database storage.
    
    Args:
        md_files (list): List of file paths to markdown files
        
    Returns:
        list: List of processed document chunks with metadata
    """
    all_chunks = []
    for file_path in tqdm(md_files, desc="Processing Documents"):
        all_chunks.extend(process_immigration_doc(file_path))

    print(f"Total documents processed: {len(all_chunks)}")
    return all_chunks


# Main processing pipeline starts here
print("Creating new vector DB...")

md_files = glob.glob("./datasets/total/*.md")

if not md_files:
    raise FileNotFoundError("No markdown files found in the specified directory.")

print(f"Found {len(md_files)} markdown files")

all_chunks = process_all_documents(md_files)

texts = [doc.page_content for doc in all_chunks]
sources = [doc.metadata["source"] for doc in all_chunks]
vectors = model.encode(texts).tolist()

# Create a DataFrame to organize the data before uploading to Azure Search
df = pd.DataFrame(
    {
        "content":texts,
        "sources":sources,
        "vectors":vectors
    }
)
df["content_id"] = df.index.astype(str)

hnsw_config = HnswAlgorithmConfiguration(
    name="hnsw-config",
    parameters = {
        "m":10,
        "efConstruction":200,
        "efSearch":100,
        "metric":"cosine"
        }
    )

vector_search_profile = VectorSearchProfile(
    name="my-vector-profile",
    algorithm_configuration_name="hnsw-config"
    )

vector_search = VectorSearch(
    algorithms=[hnsw_config],
    profiles=[vector_search_profile]
    )


fields = [
    SearchableField(name="content_id", type=SearchFieldDataType.String,key=True,filterable=True),
    SearchableField(name="content",type=SearchFieldDataType.String,searchable=True,filterable=False,sortable=False,analyzer_name="en.lucene"),
    SearchableField(name="sources",type=SearchFieldDataType.String,searchable=True,filterable=True,sortable=True),
    SearchField(
        name="vectors",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        filterable=False,
        sortable=False,
        facetable=False,
        vector_search_dimensions=384,
        vector_search_profile_name="my-vector-profile"
    )
]



prioritized_fields = SemanticPrioritizedFields(
    title_field = SemanticField(field_name="content"),
    content_fields=[SemanticField(field_name="content")]
    )

semantic_config = SemanticConfiguration(
    name="my-semantic-config", 
    prioritized_fields=prioritized_fields
    )




index = SearchIndex(name = index_name,
                    fields=fields,
                    vector_search= vector_search,
                    semantic_search= SemanticSearch(configurations=[semantic_config]),
                     )

search_client = SearchIndexClient(endpoint=endpoint, credential=credential, index=index_name)



try:
    search_client.create_index(index)
    print(f"Index created {index_name}")
except ResourceExistsError:
    print(f"Index {index_name} already exists.")
except Exception as e:
    print(f"Unexpected error: {e}")

index = search_client.get_index(index_name)
for field in index.fields:
    print(field.name)



upload_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# Convert DataFrame to documents format
documents = df.to_dict('records')


# Upload in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    try:
        result = upload_client.upload_documents(documents=batch)
        print(f"Uploaded batch {i//batch_size + 1}: {len(batch)} documents")
    except Exception as e:
        print(f"Error uploading batch: {e}")

print(f"Total documents uploaded: {len(documents)}")
