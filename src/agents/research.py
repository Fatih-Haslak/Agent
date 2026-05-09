from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


RESEARCH_PROMPT = """Sen bir Research Agent'sın. Web veya Wikipedia araması yaparak bilgi toplar ve özetlersin.

Kullanılabilir araçlar:
- wiki_search(query: str, max_chars: int = 1500)  → Türkçe Wikipedia'da arama yapar, GÜVENİLİR ve DETAYLI sonuçlar döndürür
- web_search(query: str, max_results: int = 3)    → Genel web araması yapar
- summarize(text: str, max_words: int = 100)       → Metni özetler

KURAL:
1. EĞER konuşma geçmişinde zaten bir wiki_search SONUCU varsa (ToolMessage veya Wikipedia sonucu):
   - Bu sonuçları kullanarak doğrudan kullanıcıya cevap ver.
   - TEKRAR wiki_search veya web_search çağrısı yapma!
   - Sonuçları analiz et, önemli bilgileri çıkar, Türkçe olarak cevapla.

2. EĞER henüz araştırma yapılmadıysa:
   - Bilgi sorusu için ÖNCELİKLE wiki_search kullan (Wikipedia daha güvenilir)
   - Güncel haber/yazılım gibi konularda web_search kullan
   - Araç kullanman gerekiyorsa, KESİNLİKLE JSON formatında tool çağrısı yap.

DOĞRU ÖRNEK (araç kullanımı):
{"name": "wiki_search", "args": {"query": "Sergen Yalçın", "max_chars": 1500}}

DOĞRU ÖRNEK (sonucu değerlendirme):
Sergen Yalçın, 5 Kasım 1972'de İstanbul'da doğan Türk futbol antrenörü ve eski futbolcudur...

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Tekrar araç çağrısı yapmak (eğer zaten sonuç varsa)
- {"tool": "wiki_search", ...} Sonuç: ...  ← JSON ve metin karıştırma

ÖNEMLİ: 
- Zaten araştırma sonucu varsa SADECE cevap ver.
- Araştırma yoksa SADECE JSON yaz."""


def research_node(state: AgentState):
    """Research node: Wikipedia ve web arama işlemleri yapar."""
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
    if tool_call and "name" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Research] Tool çağrısı: {tool_call['name']}")],
            "tool_calls": [{"name": tool_call["name"], "args": tool_call.get("args", {}), "id": "tc_research"}],
            "current_agent": "research"
        }

    # JSON bulunamadıysa, doğrudan yanıt olarak kabul et (sonuç değerlendirme)
    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "research"
    }
