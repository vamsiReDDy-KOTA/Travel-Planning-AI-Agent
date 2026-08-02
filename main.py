import os
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from agent.graph import travel_agent_app
from memory.long_term import qdrant_memory
from memory.short_term import session_memory

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent 2-Day Travel Planning AI Agent using LangGraph, LangChain, Google Gemini, Qdrant Vector DB, and Guardrails."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class ChatRequest(BaseModel):
    query: str = Field(..., example="Plan a 2-day trip to Tokyo with a budget under $600")
    user_id: Optional[str] = Field(default="user_default", example="user_123")
    session_id: Optional[str] = Field(default=None)
    budget: Optional[float] = Field(default=None, example=600.0)

class PreferenceRequest(BaseModel):
    user_id: str = Field(..., example="user_123")
    preference: str = Field(..., example="Prefers pet-friendly hotels and vegetarian food")

class ChatResponse(BaseModel):
    status: str
    session_id: str
    destination: str
    itinerary: str
    execution_logs: List[str]
    guardrail_passed: bool

@app.get("/health")
def health_check():
    """System health & readiness check endpoint."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "qdrant_status": "connected" if qdrant_memory.client else "disconnected",
        "gemini_configured": bool(settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY"))
    }

@app.post("/api/chat")
def plan_trip_chat(request: ChatRequest):
    """Main Agentic Chat endpoint. Streams LangGraph execution logs and final itinerary."""
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    initial_state = {
        "query": request.query,
        "user_id": request.user_id,
        "session_id": session_id,
        "city": "Tokyo",
        "days": 2,
        "budget": request.budget,
        "preferences": [],
        "guardrail_status": {"passed": True},
        "execution_logs": [],
        "weather_data": "",
        "attraction_data": "",
        "rag_data": "",
        "final_itinerary": ""
    }
    config = {"configurable": {"thread_id": session_id}}
    
    def generate_events():
        try:
            yield f"data: {json.dumps({'type': 'log', 'content': '🚀 Initializing LangGraph state machine...'})}\n\n"
            
            final_state = initial_state.copy()
            guardrail_passed = True
            
            # Phase 1: Run LangGraph up to RAG Retrieval
            for event in travel_agent_app.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    if "execution_logs" in state_update and state_update["execution_logs"]:
                        latest_log = state_update["execution_logs"][-1]
                        yield f"data: {json.dumps({'type': 'log', 'content': latest_log})}\n\n"
                        
                    if "guardrail_status" in state_update:
                        guardrail_passed = state_update["guardrail_status"].get("passed", True)
                        if not guardrail_passed:
                            final_state["final_itinerary"] = state_update.get("final_itinerary", "Request blocked by Guardrails.")
                            
                    final_state.update(state_update)
            
            # If input guardrail blocked the request early
            if not guardrail_passed:
                yield f"data: {json.dumps({'type': 'complete', 'itinerary': final_state.get('final_itinerary', ''), 'status': 'blocked', 'session_id': session_id})}\n\n"
                return

            # Phase 2: Token-Level Streaming for LLM Synthesis
            from agent.planner import TravelPlannerEngine
            from agent.graph import get_gemini_llm
            from agent.guardrails import OutputGuardrail
            
            yield f"data: {json.dumps({'type': 'log', 'content': '⚡ [Synthesis Node]: Synthesizing real-time itinerary stream...'})}\n\n"
            
            llm = get_gemini_llm()
            chunk_generator = TravelPlannerEngine.synthesize_itinerary_stream(
                city=final_state.get("city", "Tokyo"),
                days=final_state.get("days", 2),
                weather_info=final_state.get("weather_data", ""),
                attraction_info=final_state.get("attraction_data", ""),
                rag_context=final_state.get("rag_data", ""),
                user_preferences=final_state.get("preferences", []),
                budget=final_state.get("budget"),
                llm=llm
            )
            
            final_itinerary = ""
            for chunk in chunk_generator:
                final_itinerary += chunk
                # Yield raw token to frontend for immediate display
                yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
                
            yield f"data: {json.dumps({'type': 'log', 'content': '🛡️ [Output Guardrail]: Validating output constraints...'})}\n\n"
            valid, verified_text = OutputGuardrail.validate_itinerary(final_itinerary, user_budget=final_state.get("budget"))
            
            yield f"data: {json.dumps({'type': 'log', 'content': '✨ [Workflow Complete]: Itinerary successfully compiled.'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'itinerary': verified_text, 'status': 'success', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Agent Execution Failure: {str(e)}'})}\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")

@app.post("/api/memory/preferences")
def save_user_preference(req: PreferenceRequest):
    """Save user preference into Qdrant long-term vector database."""
    success = qdrant_memory.save_preference(user_id=req.user_id, preference_text=req.preference)
    if success:
        return {"status": "success", "message": f"Saved preference '{req.preference}' to Qdrant Vector DB for user '{req.user_id}'"}
    raise HTTPException(status_code=500, detail="Failed to save preference to Qdrant Vector DB.")

@app.get("/api/memory/preferences/{user_id}")
def get_user_preferences(user_id: str):
    """Fetch stored user preferences from Qdrant vector database."""
    prefs = qdrant_memory.get_all_user_preferences(user_id=user_id)
    return {"user_id": user_id, "preferences": prefs}

@app.delete("/api/memory/preferences/{user_id}/{preference}")
def delete_user_preference(user_id: str, preference: str):
    """Delete a stored user preference from Qdrant/FAISS vector database."""
    qdrant_memory.delete_preference(user_id=user_id, preference_text=preference)
    return {"status": "success", "message": f"Deleted preference '{preference}'"}


@app.get("/api/guardrails/status")
def get_guardrails_info():
    """Information on active Input & Output Guardrails rules."""
    return {
        "input_guardrails": [
            "Prompt Injection Detection",
            "Jailbreak Attempt Interception",
            "Off-topic Non-Travel Query Filtering"
        ],
        "output_guardrails": [
            "2-Day Structure Validation (Day 1 & Day 2)",
            "Budget Sanity & Threshold Guardrail Check",
            "Constraint Compliance Verification"
        ]
    }

# Mount static web interface files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
