import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)
load_dotenv()

system_prompt = """Act as an intelligent AI ChatBot who is smart and friedly, can perform web searches using the Tavily Search tool. Use the tool to find up-to-date information to answer user queries accurately."""

groq_api_key = os.getenv("GroqApiKey")
tavily_api_key = os.getenv("TavilyApiKey")
open_api_key = os.getenv("OpenApiKey")

# openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=open_api_key)
# groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
# search_tool = TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)

def get_response_from_agent(model_name: str, model_provider: str, system_prompt: str, query, allow_search: bool):
    
    try:
        if model_provider == "Groq":
            llm = ChatGroq(model=model_name, api_key=groq_api_key)
        elif model_provider == "OpenAI":
            llm = ChatOpenAI(model=model_name, api_key=open_api_key)
        else:
            logging.error(f"Invalid Model Provider: {model_provider}")
            raise Exception("Invalid Model Provider. Please choose either 'Groq' or 'OpenAI'.")
        
        search_tool = [TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)] if allow_search else []
        
        agent = create_react_agent(
            model=llm,
            tools=search_tool,
            prompt=system_prompt
        )

        # query = "tell me something about crypyo currency"
        state = {"messages": query}
        response = agent.invoke(state)
        messages = response.get("messages")
        ai_msg= [message.content for message in messages if isinstance( message, AIMessage)]
        response = ai_msg[-1]
        response = response.replace("</s>", "")
        return response
    except Exception as e:
        logging.error(f"Error in get_response_from_agent: {str(e)}")
        raise Exception(f"Error in get_response_from_agent: {e}")