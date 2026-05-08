from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


TOOL_PROMPT = """Sen bir Tool Agent'sın. Matematiksel hesaplama ve HTTP istekleri yaparsın.

Kullanılabilir araçlar:
- calculator(expression: str)  → Matematiksel ifadeyi hesaplar
- http_request(method: str, url: str, headers: Optional[str] = None, body: Optional[str] = None) → HTTP isteği gönderir

KURAL:
1. EĞER konuşma geçmişinde zaten bir calculator veya http_request SONUCU varsa (ToolMessage ile sonuç):
   - Bu sonucu kullanarak doğrudan kullanıcıya cevap ver.
   - TEKRAR calculator veya http_request çağrısı yapma!
   - Sonucu açıkla, Türkçe olarak cevapla.

2. EĞER henüz hesaplama yapılmadıysa:
   - Araç kullanman gerekiyorsa, KESİNLİKLE JSON formatında tool çağrısı yap.

DOĞRU ÖRNEK (araç kullanımı):
{"tool": "calculator", "args": {"expression": "15 * 23"}}

DOĞRU ÖRNEK (sonucu değerlendirme):
15 ile 23'ü çarptığımızda 345 eder. Bu sonuç...

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Tekrar araç çağrısı yapmak (eğer zaten sonuç varsa)
- calculator(15*23)  ← fonksiyon çağrısı yazma!

ÖNEMLİ: 
- Zaten hesaplama sonucu varsa SADECE cevap ver.
- Hesaplama yoksa SADECE JSON yaz."""


def tool_agent_node(state: AgentState):
    """Tool node: Hesaplama ve HTTP istekleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-10:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content[:500]}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Zaten hesaplama sonucu varsa onu kullan. Yoksa SADECE JSON yaz."
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
