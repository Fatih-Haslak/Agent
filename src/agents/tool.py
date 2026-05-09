from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


TOOL_PROMPT = """Sen bir Tool Agent'sın. Matematiksel hesaplama ve HTTP istekleri yaparsın.

## Kullanılabilir Araçlar
- calculator(expression): Matematiksel hesaplama
- http_request(method, url): HTTP isteği gönderir

## ReAct Pattern (Düşün → Hareket Et → Gözlemle)
1. **Düşün**: Kullanıcı ne istiyor? Matematik mi? API mi?
2. **Hareket Et**: İlgili tool'u çağır
3. **Gözlemle**: Sonucu analiz et

## Kurallar
- Matematik → calculator
- API/HTTP → http_request
- Eğer zaten sonuç varsa → TEKRAR çağırma, sonucu değerlendir

## ÖNEMLİ: SADECE JSON

### DOĞRU ÖRNEK:
{"tool": "calculator", "args": {"expression": "15 * 23"}}

### YANLIŞ ÖRNEKLER:
- Hesaplayalım: {"tool": ...}  ← metin + JSON karıştırma
- calculator(15*23)  ← fonksiyon çağrısı

SADECE düz JSON."""


def tool_agent_node(state: AgentState):
    """Tool node: Hesaplama ve HTTP istekleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-10:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content[:500]}\n"

    user_prompt = f"""Konuşma geçmişi:
{context}

ReAct Pattern:
ADIM 1 - Düşün: Matematiksel işlem mi, yoksa HTTP mi?
ADIM 2 - Karar: Hangi tool'u kullanmalıyım?
ADIM 3 - Eğer tool kullanacaksam SADECE JSON yaz. Yoksa doğrudan cevap ver."""

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
