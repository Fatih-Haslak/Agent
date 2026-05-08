from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


TOOL_PROMPT = """Sen bir Tool Agent'sın. Matematiksel hesaplama ve HTTP istekleri yaparsın.

Kullanılabilir araçlar:
- calculator: Matematiksel ifadeleri hesaplar. Argüman: {"expression": "2+2"}
- http_request: HTTP isteği gönderir (kritik, onay gerektirir). Argüman: {"method": "GET", "url": "https://..."}

Eğer araç kullanman gerekiyorsa, JSON formatında tool çağrısı yap:
{"tool": "calculator", "args": {"expression": "15 * 23"}}

Eğer araç kullanmana gerek yoksa, doğrudan yanıt ver.
Yanıtın başında JSON kullanma, doğrudan cevap ver."""


def tool_agent_node(state: AgentState):
    """Tool node: Hesaplama ve HTTP istekleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla."
    raw = llm.chat(TOOL_PROMPT, user_prompt, max_new_tokens=200, temperature=0.1)

    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=raw)],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_tool"}],
            "current_agent": "tool"
        }

    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "tool"
    }
