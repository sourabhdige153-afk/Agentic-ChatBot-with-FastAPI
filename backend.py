from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List
from utils.constant import ALLOWED_MODELS
from fastapi.responses import JSONResponse 
from ai_agent import get_response_from_agent

app = FastAPI(
    title="Agentic ChatBot with FastAPI",
    description="An intelligent AI ChatBot that can perform web searches using the Tavily Search tool.",
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/docs",
    redoc_url="/api/redoc",
    root_path="/myapp"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

class RequestBody(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool
    

@app.post("/chat", summary="Get response from AI ChatBot")
async def chat_endpoint(request_body: RequestBody):
    """
    - Chat endpoint for the Agentic AI ChatBot.
    Args:
    - request_body (RequestBody):
        - model_name: Name of the LLM model to use.
        - model_provider: Provider of the model (e.g., OpenAI, Groq).
        - system_prompt: System instructions for the chatbot.
        - messages: List of user messages (conversation history).
        - allow_search: Whether to enable web search via Tavily.
    
    Returns:
        - dict: JSON response containing the chatbot's reply.
    """
    
    if request_body.model_name not in ALLOWED_MODELS:
        return {"error": "Invalide Model Name. Please choose a valid model."}
    
    try:
        # query = " ".join(request_body.messages)
        response = get_response_from_agent(
            model_name=request_body.model_name,
            model_provider=request_body.model_provider,
            system_prompt=request_body.system_prompt,
            query=request_body.messages,
            allow_search=request_body.allow_search
        )
        
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": f"{str(e)}"}, status_code=500)
    
    

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000)