"""
LlamaIndex and RAGAS Evaluation Pipeline for LLM Query Systems
--------------------------------------------------------------

This script facilitates the evaluation of a query engine built with LlamaIndex, leveraging RAGAS evaluation metrics
to measure system performance across multiple criteria, including faithfulness, answer relevancy, and harmfulness.
The evaluation pipeline consists of several key steps: embedding setup, document loading, vector index construction,
query engine building, and evaluation against a standardized dataset.

Key Components:
    - Embedding setup using Ollama models
    - Document parsing and vector index creation from PDF and TXT files
    - Query engine construction and testing with LlamaIndex
    - Evaluation using RAGAS metrics on a prepared dataset, with time-tracking for each step

Metrics are recorded in a dictionary `time_dict` and saved as a text file, with evaluation results stored in CSV format.
This pipeline is particularly valuable for validating the response quality of LLM-driven query engines, supporting tasks
such as benchmarking and optimization in high-performance environments.

Author: Anand Kamble
Date: September 29, 2024
Usage: Ideal for LLM-based search and query system evaluations, suitable for isolated, local testing to benchmark models.
"""

# %%
import time
from ragas.integrations.llama_index import evaluate
from ragas.metrics.critique import harmfulness
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv
import json
import os
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from datasets import Dataset

# %% Configuration Constants
QUERY_MODEL = "llama3.1"  # Model for generating embeddings and querying
EVALUATION_MODEL = "llama3.1"  # Model for evaluation
DATASET = "PatronusAIFinanceBenchDataset"  # Dataset used for evaluation

# Dictionary to store execution times of each pipeline step
time_dict = {}
start_time = time.time()

# %% Embedding Setup
# Initializes embeddings using Ollama's LLMs. 
# Embeddings are stored in `Settings` for easy access across steps.
embeddings = OllamaEmbedding(model_name=QUERY_MODEL, base_url="http://class02:11434")
Settings.embed_model = embeddings
end_time = time.time()
time_dict['embedding_setup'] = end_time - start_time
print("Time taken for embedding setup: ", time_dict['embedding_setup'])

# %% Document Loading
# Load documents from the dataset directory, recursively searching 
# for files with .pdf and .txt extensions.
start_time = time.time()
documents = SimpleDirectoryReader(
    f"./data/{DATASET}", required_exts=[".pdf", ".txt"], recursive=True
).load_data()
end_time = time.time()
time_dict['document_loading'] = end_time - start_time
print("Time taken for document loading: ", time_dict['document_loading'])

# %% Vector Index Construction
# Builds a vector index from loaded documents to be used by the query engine.
print("Building the vector index...")
start_time = time.time()
vector_index = VectorStoreIndex.from_documents(documents[:2])  # Limiting to first 2 documents for efficiency
end_time = time.time()
time_dict['vector_index_building'] = end_time - start_time
print("Time taken for vector index building: ", time_dict['vector_index_building'])

# %% Query Engine Construction
# Creates a query engine using the vector index and the specified LLM model.
print("Building the query engine...")
start_time = time.time()
generator_llm = Ollama(
    model=QUERY_MODEL,
    request_timeout=600.0,
    base_url="http://class02:11434",
    additional_kwargs={"max_length": 512}
)
query_engine = vector_index.as_query_engine(llm=generator_llm)
end_time = time.time()
time_dict['query_engine_building'] = end_time - start_time
print("Time taken for query engine building: ", time_dict['query_engine_building'])

# %% Metric Configuration
# Defines the set of RAGAS evaluation metrics to apply to the model outputs.
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    harmfulness,
]

# %% Evaluation Model Configuration
# Sets up the LLM used as a "critic" for evaluating the query engine's outputs.
critic_llm = Ollama(model=EVALUATION_MODEL, base_url="http://class01:11434", request_timeout=600.0)
evaluator_llm = critic_llm  # Using Ollama model locally to avoid external calls

# %% Dataset Preparation
# Converts loaded JSON data to a Dataset object required by the evaluate function.
start_time = time.time()

llama_rag_dataset = None
with open(f"data/{DATASET}/rag_dataset.json", "r") as f:
    llama_rag_dataset = json.load(f)

# Populate testset dictionary with questions and reference answers
testset = {
    "question": [item["query"] for item in llama_rag_dataset["examples"]],
    "ground_truth": [item["reference_answer"] for item in llama_rag_dataset["examples"]],
}

# Convert dictionary to Hugging Face Dataset format
dataset = Dataset.from_dict(testset)
end_time = time.time()
time_dict['testset_loading'] = end_time - start_time
print("Time taken for testset loading: ", time_dict['testset_loading'])

# %% Evaluation Execution
# Runs the evaluation pipeline and captures the result in a DataFrame.
print("Evaluating the query engine...")
start_time = time.time()
result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=dataset,
    llm=evaluator_llm,
    embeddings=OllamaEmbedding(model_name=EVALUATION_MODEL, base_url="http://class03:11434"),
    raise_exceptions=False
)
end_time = time.time()
time_dict['evaluation'] = end_time - start_time
print("Time taken for evaluation: ", time_dict['evaluation'])

# %% Save Results
# Saves evaluation results and time metrics to respective files.
result.to_pandas().to_csv(f"results/{DATASET}_query_{QUERY_MODEL}_eval_{EVALUATION_MODEL}.csv")
with open(f"results/{DATASET}_query_{QUERY_MODEL}_eval_{EVALUATION_MODEL}.txt", "w") as f:
    for key, value in time_dict.items():
        f.write(f"{key}: {value} seconds\n")
