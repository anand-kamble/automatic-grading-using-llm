"""
Automated Query and Evaluation Pipeline with Phoenix and LlamaIndex
-------------------------------------------------------------------
This script leverages Phoenix, LlamaIndex, and Ollama for creating, querying, and evaluating a corpus of documents.
The process includes setting up embedding models, defining schemas for document retrieval, launching a query engine,
and interacting with Phoenix for evaluation and result tracking. Designed for use in automated systems where
retrieving, querying, and evaluating text documents are critical, such as knowledge-based applications and
automated grading systems.

Author: Anand Kamble
Usage: Suitable for academic or professional use in projects requiring asynchronous document retrieval,
       large-scale LLM queries, and real-time evaluation tracking.
"""

# %% Imports
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, set_global_handler
import phoenix as px
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import asyncio

# %% Embedding Model Setup
# Initialize embedding model with the Ollama Embedding for local model
embedding_model = OllamaEmbedding(
    model_name="phi3:latest", base_url="http://localhost:11434")

# %% Language Model Configuration
# Set the local language model with specific parameters
llm = Ollama(model="phi3:latest", temperature=0.7,
             base_url="http://localhost:11434", request_timeout=3600.0)
set_global_handler("arize_phoenix")
Settings.embed_model = embedding_model
Settings.llm = llm

# %% Schema for Corpus
# Define the schema for document embedding in Phoenix for consistent document retrieval
corpus_schema = px.Schema(
    id_column_name="id",
    document_column_names=px.EmbeddingColumnNames(
        vector_column_name="embedding",
        raw_data_column_name="text",
    ),
)

async def evaluate():
    """
    Asynchronous evaluation function that launches a Phoenix application for managing and tracking
    document queries and evaluations. It performs the following actions:
    - Launches the Phoenix app for the current session.
    - Loads documents from the specified directory and indexes them.
    - Sets up a query engine and runs example queries to test retrieval.
    - Connects to Phoenix's API client to retrieve evaluations and spans data.
    - Prints results of document queries and evaluations.

    The function is asynchronous to enable real-time interactions and evaluation.
    """
    # Launch the Phoenix application interface
    px.launch_app()

    # Load documents from the 'files' directory and initialize an index
    documents = SimpleDirectoryReader("files").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Create a query engine based on the loaded index
    qe = index.as_query_engine(llm=llm)
    
    # Perform sample queries to verify setup
    response1 = qe.query("what is k-means")
    response2 = qe.query("what is fuzzy clustering?")
    print(str(response1) + "\n" + str(response2))

    # Reinitialize the LLM for future queries
    model = Ollama(model="phi3:latest")  # Optional OpenAI model configuration

    # Initialize Phoenix Client for evaluation tracking
    Client = px.Client(endpoint="http://localhost:6006")

    # Retrieve and display evaluations and spans data from the Phoenix session
    Client.get_evaluations()
    Client.get_spans_dataframe()
    retrieved_documents_df = get_retrieved_documents(Client)

    # Query spans and retrieve question-answer pairs with reference
    queries_df = get_qa_with_reference(Client)
    print(queries_df.head())
    print("Press Ctrl+C to stop the server.")


if __name__ == "__main__":
    """
    Main entry point of the script. Sets up an asyncio event loop to run the evaluate() function.
    The loop will run until interrupted by a KeyboardInterrupt, upon which it will exit gracefully.
    This setup allows the script to maintain an active session for continuous querying and evaluation.
    """
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(evaluate())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
        quit()
