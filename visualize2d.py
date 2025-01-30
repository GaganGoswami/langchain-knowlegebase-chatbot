import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from localollama import get_openai_client, get_ollama_llm
from locallangchain import load_documents, split_documents, create_vectorstore
from pprint import pprint

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


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


# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Prework

result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
colors = [['blue', 'green', 'red', 'orange','yellow'][['products', 'employee', 'clients', 'company', 'kyc'].index(t)] for t in doc_types]

# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()


# Set up the conversation memory
#memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# The retriever is an abstraction over the VectorStore that will be used during RAG
#retriever = vectorstore.as_retriever(search_kargs={"k": 25})  # Ensure `vectorstore` is properly defined

# Putting it together: set up the conversation chain with Ollama LLM, the vector store, and memory
#conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# def chat(message, history):
#     result = conversation_chain.invoke({"question": message})
#     pprint(result)
#     return result["answer"]

# And in Gradio:
#view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)