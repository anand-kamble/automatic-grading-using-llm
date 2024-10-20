# %%############## IMPORTS ################
from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from utils import Models, list_files

# %%############## LLM ################
llm = llamacpp.LlamaCpp(model_path=Models.MISTRAL.value, n_ctx=2048, f16_kv=True)

# %%############## DATA FROM PDF ################
DATA_ROOT = "../data/"
pdf_filenames = list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)

# %%############## SPLIT THE DATA ################
splitData = []
for pdf in pdf_filenames:
    pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
    data = pdfLoader.load_and_split()
    for d in data:
        splitData.append(d)

# %%############## EMBEDDING ################
llamaEmed = LlamaCppEmbeddings(seed=100, model_path=Models.MISTRAL.value)


# %%############## PROMPT ################
template = """
You are a professor of graduate level course data mining.
The question asked is: `{question}`
rate the following answer on a scale of 1 to 10, where 1 is the worst and 10 is the best:
`{answer}`
Give answer in format: Rating = x/10
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["answer", "question"],
)

# %%############## MAKING THE CONTEXT STRING ################
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
docs = text_splitter.split_documents(splitData)
vec_store = Chroma.from_documents(splitData, llamaEmed)
base_retriever = vec_store.as_retriever()

# %%
# llamaEmed.embed_documents(texts=[context_str])
# https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
from langchain.chains import LLMChain, RetrievalQA

chain = LLMChain(llm=llm, prompt=prompt)
# https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="map_reduce", retriever=base_retriever
)


question = "Is K-means a clustering method?"
answer = "K-means is not a clustering method"
qa.run(answer=answer, question=question)
