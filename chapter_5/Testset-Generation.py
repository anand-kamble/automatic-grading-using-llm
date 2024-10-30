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
    print("Loading documents...")
    # Load documents from the specified directory
    documents = SimpleDirectoryReader("./data").load_data()
    print("Initializing generator and critic models...")
    
    # Initialize the generator and critic models using the Ollama LLM
    generator_llm = Ollama(model="phi3:latest")
    critic_llm = Ollama(model="phi3:latest")  # Alternatively, OpenAI(model="gpt-4")
    print("Initializing embeddings...")
    
    # Initialize embeddings using the HuggingFace embedding model
    embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Initializing testset generator...")
    # Create a testset generator using the initialized models and embeddings
    generator = TestsetGenerator.from_llama_index(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
    )
    print("Generating testset...")
    # Generate the test set from the loaded documents
    # Note: This process might take some time; be patient if it seems to be stuck at certain percentages
    testset = generator.generate_with_llamaindex_docs(
        documents,
        test_size=10,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    )

    print("Writing testset to CSV...")

    testset_df = testset.to_pandas()

    # Save the generated test set as a CSV file
    testset_df.to_csv("testset.csv", index=False)
    print("Testset generation completed successfully.")

    ### ============= BUILDING THE QUERY ENGINE ================== ###

    # Updating the embed model, this is required for the VectorStoreIndex
    # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#modules
    Settings.embed_model = embeddings

    print("Building the vector index...")
    vector_index = VectorStoreIndex.from_documents(documents)

    print("Building the query engine...")
    query_engine = vector_index.as_query_engine(llm=generator_llm)

    print("Querying the engine...")
    response_vector = query_engine.query(testset_df["question"][0])
    print(f"{response_vector=}")

    ### ============= EVALUATING THE QUERY ENGINE ============= ###

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        harmfulness,
    ]

    # using GPT 3.5, use GPT 4 / 4-turbo for better accuracy
    evaluator_llm = critic_llm  # OpenAI(model="gpt-3.5-turbo")
    # USING CRITIC LLM TO KEEP EVERYTHING LOCAL FOR NOW.
    testset_dict = (testset.to_dataset()).to_dict()
    print("Evaluating the query engine...")
    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=testset_dict,
        llm=evaluator_llm,
        embeddings=OllamaEmbedding(model_name="phi3:latest"),
	raise_exceptions=False
    )

    print("========= RESULTS =========")
    print(result)

    result.to_pandas().to_csv("evaluation_results.csv", index=False)


if __name__ == "__main__":
    main()