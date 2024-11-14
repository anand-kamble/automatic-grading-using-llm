"""
Query and Feedback Evaluation Pipeline with LlamaIndex, TruLlama, and LiteLLM
-----------------------------------------------------------------------------
This script sets up a document query engine and a feedback evaluation pipeline using LlamaIndex, TruLlama, and LiteLLM.
The workflow includes steps to:
- Initialize embedding and LLM models.
- Create a query engine using a vector store index.
- Define feedback functions for groundedness and relevance.
- Query the engine with test data and record feedback using TruLlama.

This setup is intended for real-time evaluation of LLM-based applications, where feedback functions assess the relevance
and groundedness of answers. TruLlama is configured to track query sessions, collect feedback, and run a dashboard for
insights.

Author: Anand Kamble
Usage: Ideal for projects requiring structured feedback on LLM responses in knowledge-based or document-intensive applications.
"""

# Uncomment the following lines to install the necessary packages
# !pip install trulens_eval llama_index openai
# !pip install "litellm>=1.25.2"

# Import necessary libraries
import time
import numpy as np
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.app import App
from trulens_eval.feedback.provider import LiteLLM

# %% Initialize Embedding Model
# Initialize Ollama embeddings for document indexing with specified context length
embed_batch_size = 16
num_ctx = 2048 * 4
embeddings = OllamaEmbeddings(model="phi3:latest", num_ctx=num_ctx, show_progress=True)
Settings.embed_model = embeddings

# Load documents and create a vector store index for the query engine
documents = SimpleDirectoryReader("../data").load_data()
start = time.time()
index = VectorStoreIndex.from_documents(documents)
end = time.time()

print(f"Indexing took {end - start} seconds with num_ctx={num_ctx}")

# %% Initialize the LLM and Query Engine
# Set up the generator LLM model and query engine for document-based queries
generator_llm = Ollama(model="phi3:latest")
query_engine = index.as_query_engine(llm=generator_llm)

# %% Configure LiteLLM Provider
# Set up LiteLLM provider for feedback functions with verbose mode enabled
provider = LiteLLM(
    model_engine="ollama/phi3:latest",
    endpoint="http://localhost:11434",
    kwargs={"set_verbose": True},
)

# Select the application context based on the initialized query engine
context = App.select_context(query_engine)

# %% Define Feedback Functions
# Groundedness feedback function to assess response grounding with context
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on(context.collect())
    .on_output()
)

# Relevance feedback functions for assessing answer and context relevance
f_answer_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

# %% Initialize TruLlama for Query Recording and Feedback
# Set up TruLlama query engine recorder with feedback functions for groundedness and relevance
tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id="LlamaIndex_App1",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# %% Load Test Dataset
# Load the test dataset containing questions and ground truths
testset = pd.read_csv("../testset.csv")
testset_dict = {
    "question": list(testset["question"]),
    "ground_truth": list(testset["ground_truth"]),
}

# %% Query the Engine and Record Feedback
# Query the engine with each question in the test set and record responses and feedback
with tru_query_engine_recorder as recording:
    for question in testset_dict["question"]:
        print(f"Querying the engine with question: {question}")
        query_engine.query(question)

    # Retrieve the record of the query session
    rec = recording.get()

    # Initialize Tru for accessing records and feedback results
    tru = Tru()
    # Uncomment the following line to run the Tru dashboard
    # tru.run_dashboard()

    # Retrieve feedback results and print each feedback score
    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        print(feedback.name, feedback_result.result)

    # Retrieve records and feedback for the specified app_id and display records
    records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])
    print(records.head())

    # Run Tru dashboard (uncomment to run)
    tru.run_dashboard()

# %%
