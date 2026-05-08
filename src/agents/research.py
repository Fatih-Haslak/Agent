from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


RESEARCH_PROMPT = """Sen bir Research Agent'sın. Web araması yaparak bilgi toplar ve özetlersin.

Kullanılabilir araçlar:
- web_search: Web'de arama yapar. Argüman: {"query": "arama sorgusu", "max_results": 3}
- summarize: Metni özetler. Argüman: {"text": "özetlenecek metin", "max_words": 100}

Eğer araç kullanman gerekiyorsa, aşağıdaki JSON formatında tool çağrısı yap:
{"tool": "web_search", "args": {"query": "..."}}

Eğer araç kullanmana gerek yoksa (doğrudan yanıt verebiliyorsan), sadece yanıtını yaz.
Yanıtın başında JSON kullanma, doğrudan cevap ver."""


def research_node(state: AgentState):
    """Research node: Web arama ve özetleme işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla."
    raw = llm.chat(RESEARCH_PROMPT, user_prompt, max_new_tokens=300, temperature=0.3)

    # Tool call var mı kontrol et
    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=raw)],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_research"}],
            "current_agent": "research"
        }

    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "research"
    }
