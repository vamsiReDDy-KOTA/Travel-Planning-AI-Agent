import os
import re
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


from config import settings
from agent.guardrails import InputGuardrail, OutputGuardrail
from agent.planner import TravelPlannerEngine
from agent.rag import rag_engine
from tools.weather_tool import get_weather_forecast
from tools.attraction_tool import lookup_flights_hotels_attractions
from tools.search_tool import search_web_travel
from memory.long_term import qdrant_memory
from memory.short_term import short_term_checkpointer, session_memory


class TravelAgentState(TypedDict):
    query: str
    user_id: str
    session_id: str
    city: str
    days: int
    budget: Optional[float]
    preferences: List[str]

    guardrail_status: Dict[str, Any]
    execution_logs: List[str]
    weather_data: str
    attraction_data: str
    rag_data: str
    final_itinerary: str


# Initialize Gemini LLM if API Key exists
def get_gemini_llm():
    api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
    import sys
    print(f"DEBUG: get_gemini_llm called. API KEY length: {len(api_key) if api_key else 0}", flush=True)
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=api_key,
                temperature=0.0,
                max_retries=2,
                timeout=120
            )
            print("DEBUG: Invoking test on Gemini...", flush=True)
            llm.invoke("test") # Test the API
            print("DEBUG: Gemini invoke success!", flush=True)
            return llm
        except Exception as e:
            print(f"[Gemini Init Warning] {e}. Falling back to deterministic planner.", flush=True)
            return None
        
    return None


# LangGraph Node Functions
def input_guardrail_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 1: Input Guardrail Validation."""
    query = state.get("query", "")
    is_valid, msg = InputGuardrail.validate(query)
    
    logs = state.get("execution_logs", [])
    if not is_valid:
        logs.append(f"🛑 [Guardrail Node]: Input blocked. Reason: {msg}")
        return {
            "guardrail_status": {"passed": False, "reason": msg},
            "execution_logs": logs,
            "final_itinerary": msg
        }
        
    logs.append("✅ [Guardrail Node]: Input query validated as safe and travel-focused.")
    return {
        "guardrail_status": {"passed": True, "reason": "Query approved"},
        "execution_logs": logs
    }


def memory_retrieval_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 2: Long-Term Qdrant Vector DB & Short-Term Memory Retrieval."""
    user_id = state.get("user_id", "default_user")
    session_id = state.get("session_id", "default_session")
    query = state.get("query", "").lower()
    logs = state.get("execution_logs", [])
    
    # Qdrant long-term preferences
    stored_prefs = list(qdrant_memory.get_all_user_preferences(user_id=user_id))
    
    # Merge preferences present directly in prompt
    if "pet" in query and not any("pet" in p.lower() for p in stored_prefs):
        stored_prefs.append("Pet-Friendly Hotels")
    if "veg" in query and not any("veg" in p.lower() for p in stored_prefs):
        stored_prefs.append("Vegetarian Options")
    if stored_prefs:
        logs.append(f"🧠 [Memory Node]: Retrieved from Vector DB & Query context.")

    
    # Save conversation context into short-term session memory
    session_memory.add_message(session_id, "user", state.get("query", ""))
    
    return {
        "preferences": stored_prefs,
        "execution_logs": logs
    }



def city_extraction_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 3: Dynamically Detect Any Destination City, Requested Days, and Budget from Query."""
    query = state.get("query", "").strip()
    logs = state.get("execution_logs", [])
    
    # Extract requested number of days (e.g. '3-day', '3 days', '4 day')
    days_match = re.search(r"(\d+)\s*-?\s*day", query, re.IGNORECASE)
    requested_days = 2
    if days_match:
        try:
            num = int(days_match.group(1))
            if 1 <= num <= 7:
                requested_days = num
        except Exception:
            pass

    # 1. Regex pattern search for 'to <city>', 'in <city>', 'for <city>', 'visit <city>'
    pattern = r"(?:to|in|for|visit|visiting|explore|exploring)\s+([A-Za-z\s\-]+?)(?:\s+under|\s+with|\s+budget|\s+on|\s+\$|\s+\d+|$)"
    match = re.search(pattern, query, re.IGNORECASE)
    
    detected_city = None
    if match:
        extracted = match.group(1).strip()
        # Clean up any trailing filler words
        extracted = re.sub(r"\b(a|the|trip|itinerary|1-day|2-day|3-day|4-day|5-day|day|days)\b", "", extracted, flags=re.IGNORECASE).strip()
        if extracted and len(extracted) > 1:
            detected_city = extracted.title()

    # 2. Fallback: Parse query tokens removing common stop words
    if not detected_city:
        stop_words = {
            "plan", "a", "the", "1-day", "2-day", "3-day", "4-day", "1", "2", "3", "4", "5", "day", "days", "trip", "itinerary",
            "under", "with", "budget", "for", "in", "to", "hotel", "hotels", "flight",
            "flights", "pet", "friendly", "options", "please", "help", "me", "show"
        }
        tokens = [w.strip("?,!.$") for w in query.split() if w.lower().strip("?,!.$") not in stop_words and not w.isdigit() and not w.startswith("$")]
        if tokens:
            detected_city = " ".join(tokens).title()

    if not detected_city:
        detected_city = "Tokyo"  # Intelligent default if completely unspecified

    logs.append(f"🎯 [Planner Node]: Dynamically extracted target destination '{detected_city}' for {requested_days}-Day trip.")
    return {
        "city": detected_city,
        "days": requested_days,
        "execution_logs": logs
    }




def tool_execution_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 4: Dynamic Tool Execution (Weather, Attraction DB, Web Search)."""
    city = state.get("city", "Tokyo")
    budget = state.get("budget")
    prefs = state.get("preferences", [])
    logs = state.get("execution_logs", [])
    
    # Tool 1: Weather API
    logs.append(f"🛠️ [Tool Node]: Invoking Open-Meteo Weather Tool for '{city}'...")
    weather_res = get_weather_forecast.invoke({"city": city})
    
    # Tool 2: Attraction & Flight DB
    filter_parts = []
    if budget:
        filter_parts.append(f"budget_{int(budget)}")
    if any("pet" in p.lower() for p in prefs):
        filter_parts.append("pet_friendly")
    filter_str = "_".join(filter_parts) if filter_parts else "all"

    logs.append(f"🛠️ [Tool Node]: Invoking Attraction & Booking Database Tool for '{city}' (Filter: {filter_str})...")
    attraction_res = lookup_flights_hotels_attractions.invoke({"city": city, "preference_filter": filter_str})
    
    # Tool 3: Web Search Tool fallback/supplement
    logs.append(f"🛠️ [Tool Node]: Invoking DuckDuckGo Travel Search Tool...")
    search_res = search_web_travel.invoke({"query": f"{city} tourism places to visit historic landmarks travel guide"})
    
    combined_attraction = f"{attraction_res}\n\n🔍 LIVE SEARCH TIPS:\n{search_res}"
    
    return {
        "weather_data": weather_res,
        "attraction_data": combined_attraction,
        "execution_logs": logs
    }



def rag_retrieval_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 5: Multi-Hop RAG Knowledge Base Retrieval."""
    city = state.get("city", "Tokyo")
    days = state.get("days", 2)
    query = state.get("query", "")
    logs = state.get("execution_logs", [])
    
    logs.append(f"📖 [RAG Node]: Performing Multi-Hop RAG retrieval over travel guide index for '{city}'...")
    rag_context = rag_engine.multi_hop_retrieve(city=city, days=days, query=query)
    
    return {
        "rag_data": rag_context,
        "execution_logs": logs
    }


def itinerary_synthesis_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 6: Synthesize N-Day Trip Itinerary using Gemini LLM / Plan Engine."""
    city = state.get("city", "Tokyo")
    days = state.get("days", 2)
    weather = state.get("weather_data", "")
    attractions = state.get("attraction_data", "")
    rag_ctx = state.get("rag_data", "")
    prefs = state.get("preferences", [])
    budget = state.get("budget")
    logs = state.get("execution_logs", [])
    
    logs.append(f"⚡ [Synthesis Node]: Generating customized {days}-Day travel plan for '{city}'...")
    llm = get_gemini_llm()
    
    raw_itinerary = TravelPlannerEngine.synthesize_itinerary(
        city=city,
        days=days,
        weather_info=weather,
        attraction_info=attractions,
        rag_context=rag_ctx,
        user_preferences=prefs,
        budget=budget,
        llm=llm
    )
    
    return {
        "final_itinerary": raw_itinerary,
        "execution_logs": logs
    }



def output_guardrail_node(state: TravelAgentState) -> Dict[str, Any]:
    """Node 7: Output Guardrail Validation on Final Itinerary."""
    raw_text = state.get("final_itinerary", "")
    budget = state.get("budget")
    logs = state.get("execution_logs", [])
    
    logs.append("🛡️ [Output Guardrail Node]: Validating output structure, day count, and budget bounds...")
    valid, verified_text = OutputGuardrail.validate_itinerary(raw_text, user_budget=budget)
    
    logs.append("✨ [Workflow Complete]: Itinerary successfully compiled and returned to user.")
    
    return {
        "final_itinerary": verified_text,
        "execution_logs": logs
    }


def guardrail_router(state: TravelAgentState) -> str:
    """Conditional Edge: Route to end if input guardrail fails."""
    status = state.get("guardrail_status", {})
    if not status.get("passed", True):
        return "blocked"
    return "allowed"


# Build LangGraph State Graph
workflow = StateGraph(TravelAgentState)

workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("memory_retrieval", memory_retrieval_node)
workflow.add_node("city_extraction", city_extraction_node)
workflow.add_node("tool_execution", tool_execution_node)
workflow.add_node("rag_retrieval", rag_retrieval_node)

# Set Entry Point
workflow.set_entry_point("input_guardrail")

# Add Conditional Edge for Guardrails
workflow.add_conditional_edges(
    "input_guardrail",
    guardrail_router,
    {
        "blocked": END,
        "allowed": "memory_retrieval"
    }
)

workflow.add_edge("memory_retrieval", "city_extraction")
workflow.add_edge("city_extraction", "tool_execution")
workflow.add_edge("tool_execution", "rag_retrieval")
workflow.add_edge("rag_retrieval", END)

# Compile LangGraph app
travel_agent_app = workflow.compile(checkpointer=short_term_checkpointer)
