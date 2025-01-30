import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from localollama import get_openai_client, get_ollama_llm
from locallangchain import load_documents, split_documents, create_vectorstore
from pprint import pprint

# Load environment variables in a file called .env
# load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Initialize OpenAI client
openai = get_openai_client()

# Read in documents using LangChain's loaders
folders = glob.glob("knowledge-base/*")
documents = load_documents(folders)
chunks = split_documents(documents)

# Create vectorstore
db_name = "vector_db"
vectorstore = create_vectorstore(chunks, db_name)
pprint(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Create a new Chat with Ollama
MODEL = "mistral:7b"  # Example: Use "llama3" or any locally installed model
llm = get_ollama_llm(model=MODEL)

# Set up the conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# The retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever(search_kargs={"k": 25})  # Ensure `vectorstore` is properly defined

# Putting it together: set up the conversation chain with Ollama LLM, the vector store, and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    pprint(result)
    return result["answer"]

# And in Gradio:
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)