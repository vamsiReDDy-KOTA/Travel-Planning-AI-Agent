import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings

class DualVectorMemoryStore:
    """Manages long-term user preferences using FAISS with Dual-Embedding fallback."""

    def __init__(self):
        self.vector_store = None
        self.active_backend = "FAISS (In-Memory / Local)"
        self.faiss_path = os.path.join(settings.DATA_DIR, ".faiss_index")
        
        # 1. Initialize Dual Embeddings
        self.embeddings = None
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                temp_emb = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004", 
                    google_api_key=api_key
                )
                temp_emb.embed_query("test") # Test the API
                self.embeddings = temp_emb
                print("[Vector DB] Initialized Google Gemini Embeddings.")
            except Exception as e:
                print(f"[Vector DB Warning] Gemini embeddings failed ({e}).")
        
        if not self.embeddings:
            print("[Vector DB] Falling back to local HuggingFace embeddings (all-MiniLM-L6-v2)...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("[Vector DB] Successfully loaded local HuggingFace embeddings.")

        self._initialize_backend()
        
    def _initialize_backend(self):
        """Loads FAISS index from disk or creates a new one."""
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        if os.path.exists(self.faiss_path):
            try:
                self.vector_store = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"[Vector DB] Loaded existing FAISS index from {self.faiss_path}")
            except Exception as e:
                print(f"[Vector DB] Failed to load existing index: {e}. Creating new FAISS index.")
                self.vector_store = FAISS.from_texts(["[System Initialization]"], self.embeddings)
        else:
            self.vector_store = FAISS.from_texts(["[System Initialization]"], self.embeddings)
            
    def _save_local(self):
        if self.vector_store:
            self.vector_store.save_local(self.faiss_path)

    def save_preference(self, user_id: str, preference_text: str, metadata: Dict[str, Any] = None) -> bool:
        """Saves user preference into FAISS Vector Store."""
        try:
            meta = {"user_id": user_id, "preference": preference_text, **(metadata or {})}
            self.vector_store.add_texts(texts=[preference_text], metadatas=[meta])
            self._save_local()
            return True
        except Exception as e:
            print(f"[Vector DB Save Error]: {e}")
            return False

    def search_preferences(self, user_id: str, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves stored user preferences via semantic similarity."""
        try:
            search_query = query if query else "Travel preferences, constraints, likes, and dislikes"
            
            # Simple metadata filtering in FAISS
            docs = self.vector_store.similarity_search(search_query, k=limit * 2, filter={"user_id": user_id})
            
            user_prefs = []
            for doc in docs:
                if doc.metadata.get("user_id") == user_id:
                    # Ignore the dummy initialization document
                    if "System Initialization" not in doc.page_content:
                        user_prefs.append(doc.page_content)
                 
            # Deduplicate
            return list(set(user_prefs))[:limit]
        except Exception as e:
            print(f"[Vector DB Search Error]: {e}")
            return []

    def get_all_user_preferences(self, user_id: str) -> List[str]:
        return self.search_preferences(user_id=user_id, query="What are all the specific requirements and rules for the user's travel?", limit=20)

qdrant_memory = DualVectorMemoryStore()

