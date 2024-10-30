# %%
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, set_global_handler
import phoenix as px
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import set_global_handler
import asyncio

# %%
# Without this command, OpenAI model is used
# embedding_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"  # small model
# )

embedding_model = OllamaEmbedding(
    model_name="phi3:latest", base_url="http://localhost:11434")

# %%
llm = Ollama(model="phi3:latest", temperature=0.7,
             base_url="http://localhost:11434", request_timeout=3600.0)
set_global_handler("arize_phoenix")
Settings.embed_model = embedding_model
Settings.llm = llm
# %%

corpus_schema = px.Schema(
    id_column_name="id",
    document_column_names=px.EmbeddingColumnNames(
        vector_column_name="embedding",
        raw_data_column_name="text",
    ),
)

async def evaluate():
    """
    This asynchronous function is responsible for launching a Phoenix application.
    The Phoenix application is launched in the current event loop.
    """
    # %%
    px.launch_app()

    # %%
    documents = SimpleDirectoryReader("files").load_data()
    #%%
    index = VectorStoreIndex.from_documents(documents)
    #%%
    qe = index.as_query_engine(llm=llm)
    #%%
    response1 = qe.query("what is k-means")
    # %%
    response2 = qe.query("what is fuzzy clustering?")
    # %%
    print(str(response1) + "\n" + str(response2))

    # %%
    model = Ollama(model="phi3:latest") #OpenAIModel(model="gpt-3.5-turbo-instruct")
    # %%
    Client = px.Client(endpoint="http://localhost:6006")

    # %%
    Client.get_evaluations()
    # %%
    Client.get_spans_dataframe()
    # %%
    retrieved_documents_df = get_retrieved_documents(Client)

    # %%
    Client.query_spans
    # %%
    get_qa_with_reference(Client)

    # %%
    queries_df = get_qa_with_reference(Client)
    # %%
    # print("queries_df= ", queries_df)
    print(queries_df.head())
    print("Press Ctrl+C to stop the server.")


if __name__ == "__main__":
    """
    This is the main entry point of the program. It sets up an asyncio event loop,
    ensures that the evaluate() coroutine is scheduled to run, and starts the event loop.
    If a KeyboardInterrupt is raised (which usually happens when the user hits Ctrl+C),
    it simply passes and proceeds to the finally block.
    In the finally block, it prints a message, closes the event loop, and quits the program.
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