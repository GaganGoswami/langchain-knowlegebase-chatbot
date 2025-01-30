import os
import glob
from dotenv import load_dotenv
import gradio as gr

# imports for langchain

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# price is a factor for our company, so we're going to use a low cost model

# MODEL = "gpt-4o-mini"
MODEL = "llama3.2"
db_name = "vector_db"

# Load environment variables in a file called .env

# load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
# openai = OpenAI()

openai = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase

folders = glob.glob("knowledge-base/*")

# With thanks to CG and Jon R, students on the course, for this fix needed for some users 
text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

len(chunks)

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
# Chroma is a popular open source Vector Database based on SQLLite

#embeddings = OllamaEmbeddings(model="llama3.2")  # Use an available model

#embeddings = OpenAIEmbeddings()

# If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
# Then replace embeddings = OpenAIEmbeddings()
# with:
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Delete if already exists

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create vectorstore

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# OPEN AI VERSION

# create a new Chat with OpenAI
#llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
#memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
#retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
#conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# LOCAL OLLAMA VERSION
# Define the model (replace with any available Ollama model like "mistral", "llama3", etc.)
MODEL = "mistral:7b"  # Example: Use "llama3" or any locally installed model

# Create a new Chat with Ollama
llm = ChatOllama(model=MODEL, temperature=0.7)

# Set up the conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# The retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever(search_kargs={"k":25})  # Ensure `vectorstore` is properly defined

# Putting it together: set up the conversation chain with Ollama LLM, the vector store, and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "Can you describe Insurellm in a few sentences"
# OPEN AI
result = conversation_chain.invoke({"question":query})
# OLLAMA
# result = conversation_chain.invoke({"input":query})

print(result["answer"])


# set up a new conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    print(result)
    return result["answer"]


# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)