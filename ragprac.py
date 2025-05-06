from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
loader = PyPDFLoader('ResumeTayyabaArooj.pdf')
pages= loader.load()
print(len(pages))
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 0)
texts = text_splitter.split_documents(pages)
len(texts)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


client= MongoClient("mongodb+srv://rag_implement:123abc@cluster0.sfbmsvf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
dbName= "newdatabase"
collectionName= "resume"
db= client[dbName]
collection= db[collectionName]

docsearch= MongoDBAtlasVectorSearch.from_documents (texts,embeddings, collection=collection)
docsearch.embeddings
custom_prompt_template= """Use the database to answer questions:
{context}

Answer the following question:
{question}

Provide a answer below:
"""
prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


pipe = pipeline("text-generation", model="gpt2", max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)
qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={'k':5}),
    chain_type_kwargs={"prompt":prompt},
)

query = "see the newdatabse collection name resume , and share the funding amount for melanodtecetai in hkd?"

def search_resume(query):
    result = qa.invoke({"query": query})['result']
    print("Answer:", result)
    return result

search_resume(query)
