from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Paylaşılan durum (Shared State) tüm agent'lar tarafından kullanılır.
    
    Fields:
        messages: Konuşma geçmişi (short-term memory). LangGraph add_messages reducer ile yönetilir.
        tool_calls: Agent'lar tarafından üretilen tool çağrıları.
        current_agent: Şu anda aktif olan agent'ın adı (supervisor | research | code | tool).
        iteration_count: Döngü sayacı (infinite loop koruması).
        final_answer: Supervisor tarafından birleştirilen nihai yanıt.
        pending_tool: Onay bekleyen kritik tool çağrısı (interrupt için).
    """
    messages: Annotated[list, add_messages]
    tool_calls: List[Dict[str, Any]]
    current_agent: str
    iteration_count: int
    final_answer: Optional[str]
    pending_tool: Optional[Dict[str, Any]]
