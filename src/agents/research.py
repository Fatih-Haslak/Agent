from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


RESEARCH_PROMPT = """Sen bir Research Agent'sın. Web araması yaparak bilgi toplar ve özetlersin.

Kullanılabilir araçlar:
- web_search(query: str, max_results: int = 3)
- summarize(text: str, max_words: int = 100)

KURAL:
1. EĞER konuşma geçmişinde zaten bir web_search SONUCU varsa (ToolMessage veya arama sonuçları):
   - Bu sonuçları kullanarak doğrudan kullanıcıya cevap ver.
   - TEKRAR web_search veya summarize çağrısı yapma!
   - Sonuçları analiz et, önemli bilgileri çıkar, Türkçe olarak cevapla.

2. EĞER henüz araştırma yapılmadıysa:
   - Araç kullanman gerekiyorsa, KESİNLİKLE JSON formatında tool çağrısı yap.

DOĞRU ÖRNEK (araç kullanımı):
{"tool": "web_search", "args": {"query": "Atatürk kimdir", "max_results": 3}}

DOĞRU ÖRNEK (sonucu değerlendirme):
Web arama sonuçlarına göre Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusudur...

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Tekrar araç çağrısı yapmak (eğer zaten sonuç varsa)
- {"tool": "web_search", ...} Sonuç: ...  ← JSON ve metin karıştırma

ÖNEMLİ: 
- Zaten araştırma sonucu varsa SADECE cevap ver.
- Araştırma yoksa SADECE JSON yaz."""


def research_node(state: AgentState):
    """Research node: Web arama ve özetleme işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-10:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content[:500]}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Zaten araştırma sonucu varsa onu kullan. Yoksa SADECE JSON yaz."
    raw = llm.chat(RESEARCH_PROMPT, user_prompt, max_new_tokens=512, temperature=0.3)

    # Önce JSON ara (yeni araştırma)
    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Research] Tool çağrısı: {tool_call['tool']}")],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_research"}],
            "current_agent": "research"
        }

    # JSON bulunamadıysa, doğrudan yanıt olarak kabul et (sonuç değerlendirme)
    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "research"
    }
