from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


TOOL_PROMPT = """Sen bir Tool Agent'sın. Matematiksel hesaplama ve HTTP istekleri yaparsın.

Kullanılabilir araçlar:
- calculator(expression: str)  → Matematiksel ifadeyi hesaplar
- http_request(method: str, url: str, headers: Optional[str] = None, body: Optional[str] = None) → HTTP isteği gönderir

KURAL: Eğer araç kullanman gerekiyorsa, KESİNLİKLE aşağıdaki JSON formatında tool çağrısı yap.
Eğer araç kullanmana gerek yoksa, doğrudan yanıt ver.

DOĞRU ÖRNEK (araç kullanımı):
{"tool": "calculator", "args": {"expression": "15 * 23 + 7"}}

DOĞRU ÖRNEK (doğrudan yanıt):
15 çarpı 23, 345 eder.

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Hesaplayalım: {"tool": ...}  ← JSON öncesi metin yazma!
- calculator(15*23)  ← fonksiyon çağrısı yazma!

ÖNEMLİ: Tool çağrısı yapacaksan SADECE tek satır JSON yaz. Doğrudan yanıt vereceksen SADECE yanıtını yaz."""


def tool_agent_node(state: AgentState):
    """Tool node: Hesaplama ve HTTP istekleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Tool kullanacaksan SADECE JSON yaz."
    raw = llm.chat(TOOL_PROMPT, user_prompt, max_new_tokens=200, temperature=0.0)

    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Tool] Tool çağrısı: {tool_call['tool']}")],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_tool"}],
            "current_agent": "tool"
        }

    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "tool"
    }
