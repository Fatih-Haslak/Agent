from langchain_core.messages import SystemMessage
from src.config import get_llm
from src.state import AgentState
from src.tools import calculator, http_request


TOOL_PROMPT = """Sen bir Tool Agent'sın. Matematiksel hesaplama ve HTTP istekleri yaparsın.

Kullanılabilir araçlar:
- calculator: Matematiksel ifadeleri güvenli bir şekilde hesaplar
- http_request: HTTP GET/POST/PUT/DELETE istekleri gönderir (kritik, onay gerektirir)

Kurallar:
1. Hesaplama sonuçlarını net ve anlaşılır şekilde sun.
2. HTTP isteklerinde URL ve parametreleri doğrula.
3. API yanıtlarını özetle."""


def tool_agent_node(state: AgentState):
    """Tool node: Hesaplama ve HTTP istekleri yapar."""
    llm = get_llm(temperature=0).bind_tools([calculator, http_request])
    messages = [SystemMessage(content=TOOL_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
        "current_agent": "tool"
    }
