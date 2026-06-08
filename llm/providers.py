import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from llm.base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

    def get_chat_model(self , model , temperature):
        return ChatOpenAI(
            api_key=self.api_key,
            model=model,
            temperature=temperature
        )

class GroqLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

    def get_chat_model(self , model , temperature):
        return ChatGroq(
            api_key=self.api_key,
            model=model,
            temperature=temperature
        )
