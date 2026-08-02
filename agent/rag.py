from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.search_tool import search_web_travel
from memory.long_term import qdrant_memory # FAISS instance

class MultiHopRAGEngine:
    """Dynamic Multi-Hop Retrieval-Augmented Generation Engine.
    Executes dynamic DuckDuckGo scraping -> Text Chunking -> FAISS Embedding -> Semantic Retrieval.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

    def multi_hop_retrieve(self, city: str, days: int, query: str) -> str:
        """Performs true dynamic RAG by scraping web data, embedding into FAISS, and searching."""
        c_name = city.title()
        
        # Hop 0: Check FAISS Cache first!
        try:
            cached_docs = qdrant_memory.vector_store.similarity_search(
                f"{days} days {c_name} travel itinerary highlights", 
                k=4, 
                filter={"user_id": "rag_system", "city": c_name, "days": days}
            )
            if cached_docs:
                formatted_context = "\n".join([f"- {doc.page_content}" for doc in cached_docs])
                return f"=== True Dynamic FAISS RAG Knowledge Context (Cached for {days} Days) ===\n\n[Retrieved Semantic Chunks for {c_name}]\n{formatted_context}"
        except Exception as e:
            print(f"[RAG Cache Check] {e}")
            
        # Hop 1: Scrape live web data
        hop1_query = f"{days} days {c_name} trip itinerary tourism places to visit"
        hop1_results = search_web_travel.invoke({"query": hop1_query})
        
        hop2_query = f"{c_name} famous local food cuisine transit and practical travel tips"
        hop2_results = search_web_travel.invoke({"query": hop2_query})
        
        combined_text = f"{hop1_results}\n{hop2_results}"
        
        # Hop 2: Dynamic Chunking
        chunks = self.text_splitter.split_text(combined_text)
        
        # Hop 3: Embed into FAISS on-the-fly
        for chunk in chunks:
            # We save it as preference text but with a special metadata tag to keep it distinct
            qdrant_memory.save_preference(user_id="rag_system", preference_text=chunk, metadata={"city": c_name, "days": days})
            
        # Hop 4: Semantic Retrieval (True RAG)
        try:
            retrieved_docs = qdrant_memory.vector_store.similarity_search(
                f"{days} days {c_name} travel itinerary highlights", 
                k=4, 
                filter={"user_id": "rag_system", "city": c_name, "days": days}
            )
            formatted_context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs]) if retrieved_docs else combined_text
        except:
            formatted_context = combined_text
        
        return f"""=== True Dynamic FAISS RAG Knowledge Context ===

[Retrieved Semantic Chunks for {c_name} ({days} Days)]
{formatted_context}"""

rag_engine = MultiHopRAGEngine()
