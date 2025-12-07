import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from typing import TypedDict,  List

class AgentState(TypedDict):
    messages: List[str]

load_dotenv()

system_prompt = """Act as an intelligent AI ChatBot who is smart and friedly, can perform web searches using the Tavily Search tool. Use the tool to find up-to-date information to answer user queries accurately."""

groq_api_key = os.getenv("GroqApiKey")
tavily_api_key = os.getenv("TavilyApiKey")
open_api_key = os.getenv("OpenApiKey")

openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=open_api_key)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
search_tool = TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)

# agent = create_react_agent(
#     model=groq_llm,
#     tools=[search_tool],
#     prompt=system_prompt
# )

query = "tell me something about crypyo currency"

def llmnode(state:AgentState):
    response = groq_llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def searchnode(state: AgentState):
    query = state["messages"][-1]
    result = search_tool.run(query)
    return {"messages": state["messages"] + [result]}

def router_node(state: AgentState):
    last_msg = state["messages"][-1]
    
    if "search" in last_msg:
        return "search"
    return "llm"

graph = StateGraph(AgentState)

graph.add_node("llm", llmnode)
graph.add_node("search", searchnode)
graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    router_node,
    {
        "search": "search",
        "llm": END
    }
)

graph.add_edge("search", "llm")
agent = graph.compile()


response = agent.invoke({"messages": [system_prompt, query]} )

print("AI ChatBot Response:", response["messages"][-1].content)