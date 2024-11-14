"""
Automated Evaluation Pipeline for Document Querying System
----------------------------------------------------------
This script sets up a query evaluation pipeline using LlamaIndex and Ragas for document retrieval,
query generation, and performance assessment. It includes steps to load documents, initialize embedding models,
generate a test set, create a query engine, and evaluate the engine's responses based on multiple metrics.

The evaluation metrics include:
- Faithfulness: Measures the alignment of answers with the provided context.
- Answer Relevancy: Assesses the relevance of the answer to the query.
- Context Precision and Recall: Evaluate the accuracy and completeness of retrieved context.
- Harmfulness: Ensures that responses do not contain harmful content.

Author: Anand Kamble
Usage: Suitable for academic and professional applications that require rigorous evaluation of document-based LLM systems.

"""

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ragas.integrations.llama_index import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics.critique import harmfulness
from ragas.testset.generator import TestsetGenerator


def main():
    """
    Main function to execute the evaluation pipeline. This function performs the following steps:
    1. Loads documents from a specified directory.
    2. Initializes the generator and critic models for LLM processing.
    3. Sets up the embeddings for use in the query engine.
    4. Generates a test set from the loaded documents based on specified distribution.
    5. Creates a vector-based query engine and queries it with sample test set data.
    6. Evaluates the query engine’s performance using various metrics.
    
    Output:
    -------
    - Saves a generated test set as 'testset.csv'.
    - Saves evaluation results as 'evaluation_results.csv'.
    """
    print("Loading documents...")
    # Load documents from the specified directory
    documents = SimpleDirectoryReader("./data").load_data()
    
    print("Initializing generator and critic models...")
    # Initialize generator and critic models using Ollama LLM
    generator_llm = Ollama(model="phi3:latest")
    critic_llm = Ollama(model="phi3:latest")  # Optionally OpenAI(model="gpt-4")
    
    print("Initializing embeddings...")
    # Initialize embeddings using HuggingFace embedding model
    embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Initializing testset generator...")
    # Create a test set generator using initialized models and embeddings
    generator = TestsetGenerator.from_llama_index(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
    )

    print("Generating testset...")
    # Generate the test set from loaded documents with specified distribution
    testset = generator.generate_with_llamaindex_docs(
        documents,
        test_size=10,
        distributions={"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25},
    )

    print("Writing testset to CSV...")
    testset_df = testset.to_pandas()
    testset_df.to_csv("testset.csv", index=False)
    print("Testset generation completed successfully.")

    # Set embedding model for VectorStoreIndex
    Settings.embed_model = embeddings

    print("Building the vector index...")
    # Build a vector-based index from documents
    vector_index = VectorStoreIndex.from_documents(documents)

    print("Building the query engine...")
    # Create a query engine using the generator model
    query_engine = vector_index.as_query_engine(llm=generator_llm)

    print("Querying the engine...")
    # Query the engine with a sample question from the test set
    response_vector = query_engine.query(testset_df["question"][0])
    print(f"{response_vector=}")

    ### ============= Evaluating the Query Engine ============= ###

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        harmfulness,
    ]

    # Using the critic LLM for evaluation to keep everything local for now.
    testset_dict = (testset.to_dataset()).to_dict()
    print("Evaluating the query engine...")

    # Evaluate the query engine based on the test set and selected metrics
    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=testset_dict,
        llm=critic_llm,
        embeddings=OllamaEmbedding(model_name="phi3:latest"),
        raise_exceptions=False
    )

    print("========= RESULTS =========")
    print(result)
    result.to_pandas().to_csv("evaluation_results.csv", index=False)


if __name__ == "__main__":
    main()
