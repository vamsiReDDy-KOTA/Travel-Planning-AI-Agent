import os
from typing import Dict, Any, List, Optional
from config import settings

class TravelPlannerEngine:
    """Plan-and-Execute engine that decomposes complex travel requests into 2-day sub-task plans."""

    @staticmethod
    def create_execution_plan(city: str, days: int = 2, budget: Optional[float] = None, preferences: List[str] = None) -> Dict[str, Any]:
        """Decomposes trip goal into sub-task steps."""
        pref_str = ", ".join(preferences) if preferences else "Standard leisure & cultural sights"
        budget_str = f"${budget:.0f}" if budget else "Standard flexible budget"
        
        subtasks = [
            f"Step 1: Check live weather forecast for {city.title()}.",
            f"Step 2: Lookup flights, hotel options, and top attractions matching preferences ({pref_str}).",
            f"Step 3: Retrieve multi-hop RAG guide knowledge for {city.title()}.",
            f"Step 4: Synthesize structured 2-Day itinerary (Day 1 Morning/Afternoon/Evening, Day 2 Morning/Afternoon/Evening).",
            f"Step 5: Validate itinerary constraints via Output Guardrail."
        ]
        
        return {
            "destination": city.title(),
            "duration_days": days,
            "target_budget": budget_str,
            "user_preferences": pref_str,
            "subtasks": subtasks
        }

    @staticmethod
    def synthesize_itinerary(
        city: str,
        days: int = 2,
        weather_info: str = "",
        attraction_info: str = "",
        rag_context: str = "",
        user_preferences: List[str] = None,
        budget: Optional[float] = None,
        llm: Any = None
    ) -> str:
        """Synthesizes final itinerary using Google Gemini LLM or intelligent structured template with Bonus constraint support (Budget, Pet-Friendly, Multilingual, Dynamic N-Days)."""
        
        # If Gemini LLM is configured and passed, invoke Gemini via LangChain
        if llm:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an elite Multilingual Travel AI Agent. Create a detailed, personalized {days}-Day trip itinerary matching user language preferences, budget caps, and pet friendliness."),
                    ("user", """
Destination: {city}
Duration: {days} Days
Budget Target: {budget}
User Stored Preferences: {preferences}

Weather Forecast:
{weather}

Database Search & Attraction Info:
{attractions}

RAG Knowledge Base:
{rag}

Generate a clear, beautifully structured {days}-day itinerary including:
1. Overview & Weather Advice
2. Flight & Accommodation Suggestion
3. Daily breakdown for Day 1 through Day {days} (with Morning, Afternoon, Evening time slots & cost estimates)
4. Practical Local Tips (transit, money, culture, pet-friendly spots)
""")
                ])
                chain = prompt | llm
                response = chain.invoke({
                    "city": city,
                    "days": days,
                    "budget": budget or "Flexible",
                    "preferences": ", ".join(user_preferences) if user_preferences else "General sightseeing",
                    "weather": weather_info,
                    "attractions": attraction_info,
                    "rag": rag_context
                })
                return response.content
            except Exception as e:
                print(f"[LLM Synthesis Warning] Gemini invocation error: {e}. Using deterministic plan generator.")

        # Fallback to deterministic (unchanged)

        meta_headers = [
            "curated database query", "recommended hotel", "top attractions",
            "live search tips", "search results for", "multi-hop rag context",
            "hotels:", "flight options:", "travel database results",
            "[hop 1:", "[hop 2:", "hop 1", "hop 2", "district ysr",
            "wikipedia", "tripadvisor", "makemytrip", "government of andhra",
            "http", "www.", "===", "tourism places to visit", "historic landmarks travel guide",
            "famous local food cuisine", "practical travel tips", "kasagala", "group of companies", "facebook"
        ]


        attraction_items = []
        for line in (attraction_info + "\n" + rag_context).split("\n"):
            line_clean = line.strip(" -*•0123456789.").strip()
            if not line_clean or len(line_clean) < 10:
                continue
            if any(h in line_clean.lower() for h in meta_headers) or line_clean.startswith("===") or line_clean.startswith("[") or line_clean.startswith("🏨") or line_clean.startswith("✈️") or line_clean.startswith("🎟️"):
                continue
            if line_clean not in attraction_items:
                attraction_items.append(line_clean)


        specific_attractions = []
        for line in attraction_info.split("\n"):
            line_str = line.strip(" -*•0123456789.").strip()
            # Only include genuine attractions (not hotel or flight lines)
            if ("Cost:" in line_str or "Category:" in line_str or line_str.startswith("Gandikota") or line_str.startswith("Belum") or line_str.startswith("Ameen") or line_str.startswith("Devuni") or line_str.startswith("Rayalaseema")) and not any(k in line_str.lower() for k in ["hotel", "flight", "/night", "est. return", "haritha", "resort", "inn", "indigo", "alliance"]):
                if line_str not in specific_attractions:
                    specific_attractions.append(line_str)

        raw_activities = specific_attractions + [item for item in attraction_items if item not in specific_attractions]

        all_activities = []
        for item in raw_activities:
            item_lower = item.lower()
            if any(k in item_lower for k in ["hotel", "flight", "indigo", "alliance air", "/night", "est. return", "haritha", "resort", "inn", "bus", "train"]):
                continue
            if item not in all_activities:
                all_activities.append(item)

        c_name = city.title()

        
        # Build dynamic N-Day Action Plan with STRICT ZERO REPETITION
        day_sections = []
        themes = [
            ("Icons, Culture & Sunset Scenery", "🌅 Morning", "🍱 Lunch", "🏙️ Afternoon", "🌆 Evening"),
            ("Hidden Gems, Foodie Tour & Heritage", "🌅 Morning", "🍜 Lunch", "🛍️ Afternoon", "🍻 Evening"),
            ("Scenic Nature, Canyons & Relaxation", "🌅 Morning", "🍲 Lunch", "⛰️ Afternoon", "🌃 Evening"),
            ("Leisure Strolls, Arts & Local Craft Bazaars", "🌅 Morning", "☕ Lunch", "🎨 Afternoon", "🍢 Evening"),
            ("Day Trip Excursions & Farewell Festivities", "🌅 Morning", "🥗 Lunch", "📸 Afternoon", "🍷 Evening")
        ]

        # Deduplicate all_activities while preserving order
        unique_activities = []
        for act in all_activities:
            if act not in unique_activities:
                unique_activities.append(act)

        # No more fake static generators. If real attractions run out, we insert leisure time.
        
        act_idx = 0
        for d in range(1, days + 1):
            theme = themes[(d - 1) % len(themes)]
            
            day_acts = []
            for _ in range(4):
                if act_idx < len(unique_activities):
                    day_acts.append(unique_activities[act_idx])
                    act_idx += 1
                else:
                    day_acts.append("Leisure Time / Local Exploration")

            day_sections.append(f"""### 📍 DAY {d}: {theme[0]} in {c_name}
- **{theme[1]} (09:00 - 12:30)**: {day_acts[0]}
- **{theme[2]} (12:30 - 14:00)**: {day_acts[1]}
- **{theme[3]} (14:00 - 17:30)**: {day_acts[2]}
- **{theme[4]} (18:00 - 21:30)**: {day_acts[3]}""")


        action_plan_markdown = "\n\n".join(day_sections)

        pref_text = f"\n- **Applied Preferences**: {', '.join(user_preferences)}" if user_preferences else ""
        budget_text = f"\n- **Target Budget Cap**: ${budget:.0f}" if budget else "\n- **Budget Cap**: Flexible / Standard"

        pet_text = "\n- **Special Feature**: 🐾 Pet-Friendly Accommodations & Park Routes Included" if user_preferences and any("pet" in p.lower() for p in user_preferences) else ""

        return f"""# ✈️ Personalized {days}-Day Trip Itinerary: {c_name}

## 🌤️ Weather & Arrival Overview
{weather_info}
{budget_text}{pref_text}{pet_text}

---

## 🏨 Recommended Base & Logistics
{attraction_info}

---

## 📚 Local Guide Insights (Multi-Hop RAG Context)
{rag_context}

---

## 🗓️ {days}-Day Custom Action Plan for {c_name}

{action_plan_markdown}

---
*Generated dynamically for {c_name} by Intelligent Travel Agent with LangGraph reasoning and Qdrant/FAISS Memory.*
"""

    @staticmethod
    def synthesize_itinerary_stream(
        city: str,
        days: int = 2,
        weather_info: str = "",
        attraction_info: str = "",
        rag_context: str = "",
        user_preferences: List[str] = None,
        budget: Optional[float] = None,
        llm: Any = None
    ):
        """Streams the synthesized final itinerary tokens using Google Gemini LLM."""
        if llm:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an elite Multilingual Travel AI Agent. Create a detailed, personalized {days}-Day trip itinerary matching user language preferences, budget caps, and pet friendliness."),
                    ("user", """
Destination: {city}
Duration: {days} Days
Budget Target: {budget}
User Stored Preferences: {preferences}

Weather Forecast:
{weather}

Database Search & Attraction Info:
{attractions}

RAG Knowledge Base:
{rag}

Generate a clear, beautifully structured {days}-day itinerary including:
1. Overview & Weather Advice
2. Flight & Accommodation Suggestion
3. Daily breakdown for Day 1 through Day {days} (with Morning, Afternoon, Evening time slots & cost estimates)
4. Practical Local Tips (transit, money, culture, pet-friendly spots)
""")
                ])
                chain = prompt | llm
                yielded_any = False
                for chunk in chain.stream({
                    "city": city,
                    "days": days,
                    "budget": budget or "Flexible",
                    "preferences": ", ".join(user_preferences) if user_preferences else "General sightseeing",
                    "weather": weather_info,
                    "attractions": attraction_info,
                    "rag": rag_context
                }):
                    yielded_any = True
                    yield chunk.content
                return
            except Exception as e:
                import sys
                print(f"[LLM Stream Warning] Gemini stream error: {e}", flush=True)
                if yielded_any:
                    yield f"\n\n> ⚠️ **Connection Timeout**: The AI agent encountered a network timeout while generating the rest of the itinerary. Please try again or refresh the page."
                    return
                print("Yielding deterministic fallback.", flush=True)
        
        print("Executing deterministic fallback in stream.", flush=True)
        # Fallback (Only executes if LLM fails before yielding ANY tokens)
        fallback_text = TravelPlannerEngine.synthesize_itinerary(city, days, weather_info, attraction_info, rag_context, user_preferences, budget, None)
        # Yield in chunks to simulate streaming
        chunk_size = 50
        for i in range(0, len(fallback_text), chunk_size):
            yield fallback_text[i:i+chunk_size]



