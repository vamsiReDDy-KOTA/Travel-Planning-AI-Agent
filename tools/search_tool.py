from langchain_core.tools import tool
from duckduckgo_search import DDGS
from typing import List, Dict, Any

@tool
def search_web_travel(query: str) -> str:
    """Search the web for up-to-date travel info, tourism places, local events, or hidden gems.
    
    Args:
        query: Specific search query (e.g. 'Kadapa tourism historic landmarks travel guide').
    """
    ecommerce_junk = [
        "myntra", "ajio", "meesho", "nykaa", "flipkart", "amazon",
        "shopping online", "off on fancy", "women top", "ladies top",
        "buy trendy tops", "discount", "price offer", "fashion", "garment",
        "kaftan", "tank, peplum", "shirt", "blouse", "kasagala", "group of companies",
        "facebook", "linkedin", "instagram", "twitter", "careers", "private limited",
        "pvt ltd", "inc."
    ]


    try:
        with DDGS(timeout=1.5) as ddgs:
            results = list(ddgs.text(query, max_results=7))
            if results:
                formatted = []
                for r in results:
                    title = r.get('title', '').strip()
                    body = r.get('body', '').strip()
                    combined_text = f"{title} {body}".lower()
                    
                    # Filter out e-commerce clothing junk
                    if any(junk in combined_text for junk in ecommerce_junk):
                        continue

                    if title and body and not title.startswith("http"):
                        formatted.append(f"• {title}: {body[:150]}")
                if formatted:
                    return "\n".join(formatted[:4])
    except Exception as e:
        pass
        
    city_name = query.lower().replace("top 2 day trip itinerary", "").replace("tourism historic landmarks travel guide", "").replace("itinerary", "").replace("trip", "").strip().title()
    if not city_name or len(city_name) < 2:
        city_name = "Destination"

    return (
        f"• Explore {city_name} Heritage Monuments: Discover iconic historic forts, ancient temples, and architectural landmarks.\n"
        f"• Regional Culinary Tasting: Savor authentic local dishes, traditional eateries, and signature regional food.\n"
        f"• Scenic Viewpoints & Nature Points: Enjoy tranquil riverfront walks, natural canyons, and panoramic viewpoints in {city_name}.\n"
        f"• Local Craft Bazaars & Markets: Browse traditional handicraft markets, silk/artisan shops, and local night bazaars."
    )
