from openai import OpenAI
from langchain_ollama import ChatOllama

def get_openai_client():
    return OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # required, but unused
    )

def get_ollama_llm(model="mistral:7b", temperature=0.7):
    return ChatOllama(model=model, temperature=temperature)