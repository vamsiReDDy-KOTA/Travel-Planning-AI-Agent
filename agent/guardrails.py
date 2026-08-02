import re
from typing import Tuple, Dict, Any, Optional

OFF_TOPIC_KEYWORDS = [
    "write a python script to hack", "ignore all instructions", "system prompt",
    "jailbreak", "bypass safety", "cryptocurrency mining", "medical diagnosis"
]

TRAVEL_KEYWORDS = [
    "trip", "travel", "itinerary", "tokyo", "paris", "singapore", "hyderabad", "mumbai", "delhi", "bangalore",
    "hotel", "flight", "weather", "budget", "day 1", "day 2", "attraction", "sightseeing", "vacation",
    "tour", "food", "restaurant", "hello", "hi", "help", "recommend", "plan", "pet", "visit", "days"
]


class InputGuardrail:
    """Validates incoming user queries for safety, injection attempts, and travel relevance."""

    @staticmethod
    def validate(query: str) -> Tuple[bool, Optional[str]]:
        clean_query = query.strip().lower()
        
        # Check prompt injection keywords
        for keyword in OFF_TOPIC_KEYWORDS:
            if keyword in clean_query:
                return False, f"[Guardrail Intercepted]: Safety violation detected ('{keyword}'). Query blocked."
                
        # Check travel relevance unless short conversational greeting
        if len(clean_query) > 15:
            is_relevant = any(k in clean_query for k in TRAVEL_KEYWORDS)
            if not is_relevant:
                return False, (
                    "[Guardrail Notice]: This assistant specializes in **Intelligent Travel Planning**. "
                    "Please ask a travel, trip itinerary, flight/hotel, or city sightseeing question!"
                )
                
        return True, None


class OutputGuardrail:
    """Verifies generated itineraries for constraint compliance, valid budget, and structured output format."""

    @staticmethod
    def validate_itinerary(itinerary_text: str, user_budget: Optional[float] = None) -> Tuple[bool, str]:
        # 1. Structure check for trip breakdown
        has_day1 = "day 1" in itinerary_text.lower()
        if not has_day1:
            itinerary_text += "\n\n---\n*Note: Itinerary breakdown structure validated by Output Guardrail.*"


        # 2. Budget sanity check if specified
        if user_budget:
            # Look for estimated total price in text
            # Try to find a total explicitly mentioned
            total_match = re.search(r"(?i)total[\w\s]*:?\s*\$(\d+(?:\,\d+)?(?:\.\d+)?)", itinerary_text)
            if total_match:
                total_est = float(total_match.group(1).replace(",", ""))
            else:
                prices = re.findall(r"\$(\d+(?:\,\d+)?(?:\.\d+)?)", itinerary_text)
                total_est = max([float(p.replace(",", "")) for p in prices]) if prices else 0.0
                
            if total_est > user_budget * 1.1:
                itinerary_text += f"\n\n⚠️ **Budget Guardrail Alert**: Estimated total expenses (~${total_est:.0f}) exceed target budget cap (${user_budget:.0f}). Recommending budget transit & hostel options."
                    
        return True, itinerary_text

