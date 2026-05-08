from langchain_core.messages import SystemMessage
from src.config import get_llm
from src.state import AgentState
from src.tools import web_search, summarize


RESEARCH_PROMPT = """Sen bir Research Agent'sın. Web araması yaparak bilgi toplar ve özetlersin.

Kullanılabilir araçlar:
- web_search: Web'de arama yapar, sonuçları JSON döndürür
- summarize: Uzun metinleri belirtilen kelime sınırına kadar özetler

Görevi tamamlamak için gerekli tool'ları kullan.
Eğer yeterli bilgi topladıysan kısa ve öz bir özet mesajı yaz.
Kaynakları belirtmeye çalış."""


def research_node(state: AgentState):
    """Research node: Web arama ve özetleme işlemleri yapar."""
    llm = get_llm(temperature=0.2).bind_tools([web_search, summarize])
    messages = [SystemMessage(content=RESEARCH_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
        "current_agent": "research"
    }
