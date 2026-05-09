from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


RESEARCH_PROMPT = """Sen bir Research Agent'sın. Web veya Wikipedia araması yaparak bilgi toplarsın.

## Kullanılabilir Araçlar
- wiki_search(query, max_chars): Türkçe Wikipedia'da detaylı arama
- web_search(query, max_results): Genel web araması
- summarize(text, max_words): Metin özetleme

## ReAct Pattern (Düşün → Hareket Et → Gözlemle)
1. **Düşün**: Kullanıcı ne hakkında bilgi istiyor?
2. **Hareket Et**: Hangi tool'u kullanmalıyım?
3. **Gözlemle**: Tool sonucunu analiz et

## Kurallar
- Bilgi sorusu → wiki_search (ÖNCELİKLİ)
- Güncel haber → web_search
- Eğer zaten araştırma sonucu varsa → TEKRAR tool çağırma, sonucu değerlendir

## ÖNEMLİ: Tool kullanacaksan SADECE JSON yaz

### DOĞRU ÖRNEK (araç çağrısı):
{"tool": "wiki_search", "args": {"query": "Atatürk", "max_chars": 1500}}

### DOĞRU ÖRNEK (sonuç değerlendirme):
Atatürk, Türkiye Cumhuriyeti'nin kurucusudur...

### YANLIŞ ÖRNEKLER:
- JSON ve metin karıştırma
- Tekrar araç çağrısı (eğer sonuç varsa)

SADECE düz JSON veya düz metin yaz."""


def research_node(state: AgentState):
    """Research node: Wikipedia ve web arama işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-10:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content[:500]}\n"

    user_prompt = f"""Konuşma geçmişi:
{context}

ReAct Pattern ile görevi tamamla:
ADIM 1 - Düşün: Ne hakkında bilgi toplamalıyım?
ADIM 2 - Karar: Tool kullanmalı mıyım?
ADIM 3 - Eğer tool kullanacaksam SADECE JSON yaz, yoksa doğrudan cevap ver."""

    raw = llm.chat(RESEARCH_PROMPT, user_prompt, max_new_tokens=512, temperature=0.3)

    # JSON ara
    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Research] Tool çağrısı: {tool_call['tool']}")],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_research"}],
            "current_agent": "research"
        }

    # Doğrudan yanıt
    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "research"
    }
