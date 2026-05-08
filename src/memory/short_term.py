"""
Short-term Memory: Konuşma geçmişi (messages) zaten AgentState içinde 
langgraph.graph.message.add_messages reducer ile yönetilir.
Bu modül ek kısa süreli bellek fonksiyonları için ayrılmıştır.
"""

from src.state import AgentState


def get_conversation_summary(state: AgentState, last_n: int = 6) -> str:
    """Son N mesajın özetini döndürür (short-term memory helper)."""
    messages = state.get("messages", [])
    recent = messages[-last_n:] if len(messages) > last_n else messages
    return "\n".join([f"{m.type}: {m.content[:200]}" for m in recent])
