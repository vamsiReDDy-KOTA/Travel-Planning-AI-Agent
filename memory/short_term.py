from typing import List, Dict, Any, Optional
from langgraph.checkpoint.memory import MemorySaver

# Global checkpointer instance for LangGraph state persistence
short_term_checkpointer = MemorySaver()

class SessionMemoryManager:
    """Manages short-term conversation context for individual user session IDs."""
    
    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        # Keep last 20 messages per session
        if len(self._sessions[session_id]) > 20:
            self._sessions[session_id] = self._sessions[session_id][-20:]
            
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._sessions.get(session_id, [])
        
    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

session_memory = SessionMemoryManager()
