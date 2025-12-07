import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

load_dotenv()

system_prompt = """Act as an intelligent AI ChatBot who is smart and friedly, can perform web searches using the Tavily Search tool. Use the tool to find up-to-date information to answer user queries accurately."""

groq_api_key = os.getenv("GroqApiKey")
tavily_api_key = os.getenv("TavilyApiKey")
open_api_key = os.getenv("OpenApiKey")

openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=open_api_key)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
search_tool = TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)

agent = create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    prompt=system_prompt
)

query = "tell me something about crypyo currency"
state = {"messages": query}
response = agent.invoke(state)
messages = response.get("messages")
ai_msg= [message.content for message in messages if isinstance( message, AIMessage)]

# print("AI ChatBot Response:", ai_msg)
print(" Response:", ai_msg[-1])