"""
Automated Grading System using LangChain and LlamaCpp
-----------------------------------------------------
This script utilizes the LangChain library and LlamaCpp for automated grading and evaluation of student answers.
The process involves loading data from PDF files, embedding data, and creating a prompt template to rate
student answers based on relevance to a given question. The model leverages `MISTRAL` for embeddings and 
contextual document retrieval. This setup is intended for large-scale grading of open-ended answers, enabling 
professors to assess student responses on a standardized scale.

Author: Anand Kamble
Usage: Ideal for automated evaluation and grading tasks in educational and data-intensive applications.
"""

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
# Initialize the LlamaCpp model with the MISTRAL model path and configuration settings.
llm = llamacpp.LlamaCpp(model_path=Models.MISTRAL.value, n_ctx=2048, f16_kv=True)

# %%############## DATA FROM PDF ################
# Load PDF files from the data directory, listing available PDFs for processing.
DATA_ROOT = "../data/"
pdf_filenames = list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)

# %%############## SPLIT THE DATA ################
# Load and split the data from each PDF document using the PyPDFLoader and RecursiveCharacterTextSplitter.
splitData = []
for pdf in pdf_filenames:
    pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
    data = pdfLoader.load_and_split()
    for d in data:
        splitData.append(d)

# %%############## EMBEDDING ################
# Initialize LlamaCpp embeddings for creating embeddings from the loaded documents.
llamaEmed = LlamaCppEmbeddings(seed=100, model_path=Models.MISTRAL.value)

# %%############## PROMPT ################
# Define a prompt template for grading student answers based on question relevance and answer accuracy.
template = """
You are a professor of a graduate-level course in data mining.
The question asked is: `{question}`
Rate the following answer on a scale of 1 to 10, where 1 is the worst and 10 is the best:
`{answer}`
Give answer in format: Rating = x/10
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["answer", "question"],
)

# %%############## MAKING THE CONTEXT STRING ################
# Create context strings by splitting documents into manageable chunks and creating a retriever.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
docs = text_splitter.split_documents(splitData)
vec_store = Chroma.from_documents(splitData, llamaEmed)
base_retriever = vec_store.as_retriever()

# %%############## QA CHAIN ################
# Set up the LLMChain and RetrievalQA chain to evaluate answers in context and grade accordingly.
from langchain.chains import LLMChain, RetrievalQA

chain = LLMChain(llm=llm, prompt=prompt)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="map_reduce", retriever=base_retriever
)

# Define a sample question and answer for testing.
question = "Is K-means a clustering method?"
answer = "K-means is not a clustering method"
qa.run(answer=answer, question=question)
