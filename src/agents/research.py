from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


RESEARCH_PROMPT = """Sen bir Research Agent'sın. Web araması yaparak bilgi toplar ve özetlersin.

Kullanılabilir araçlar:
- web_search(query: str, max_results: int = 3)
- summarize(text: str, max_words: int = 100)

KURAL: Eğer araç kullanman gerekiyorsa, KESİNLİKLE aşağıdaki JSON formatında tool çağrısı yap.
Eğer araç kullanmana gerek yoksa, doğrudan kısa yanıt ver.

DOĞRU ÖRNEK (araç kullanımı):
{"tool": "web_search", "args": {"query": "Atatürk kimdir", "max_results": 3}}

DOĞRU ÖRNEK (doğrudan yanıt):
Atatürk, Türkiye Cumhuriyeti'nin kurucusudur.

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Arama yapacağım: {"tool": ...}  ← JSON öncesi metin yazma!
- {"tool": "web_search", "args": {"query": "..."}} İşte sonuçlar:  ← JSON sonrası metin yazma!
- web_search(query="...")  ← fonksiyon çağrısı yazma, JSON formatı kullan!

ÖNEMLİ: Tool çağrısı yapacaksan SADECE JSON yaz. Doğrudan yanıt vereceksen SADECE yanıtını yaz."""


def research_node(state: AgentState):
    """Research node: Web arama ve özetleme işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Tool kullanacaksan SADECE JSON yaz."
    raw = llm.chat(RESEARCH_PROMPT, user_prompt, max_new_tokens=300, temperature=0.1)

    # Önce JSON ara
    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Research] Tool çağrısı: {tool_call['tool']}")],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_research"}],
            "current_agent": "research"
        }

    # JSON bulunamadıysa, doğrudan yanıt olarak kabul et
    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "research"
    }
