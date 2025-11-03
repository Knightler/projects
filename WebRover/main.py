from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import os
import requests
from fastapi import FastAPI

load_dotenv()
llm = ChatOpenAI(
        model="gpt-4o-mini"
        )

@tool
def fetch_website(url: str) -> str:
  response = requests.get(url)
  return response.text
