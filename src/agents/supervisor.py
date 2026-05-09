from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


SUPERVISOR_PROMPT = """Sen bir Supervisor Agent'sın. Görevin kullanıcının isteğini analiz etmek ve en uygun uzman agent'a yönlendirmektir.

## Mevcut Agent'lar
- **research**: Bilgi toplama, araştırma, Wikipedia/web arama gerektiren görevler
- **code**: Kod yazma, çalıştırma, hata ayıklama, dosya işlemleri
- **tool**: Matematiksel hesaplama, API çağrıları, veri işleme
- **finish**: Görev tamamlandı, doğrudan nihai yanıt ver

## Karar Süreci (ReAct Pattern)
1. **Düşün**: Kullanıcının niyetini analiz et
2. **Karar ver**: Hangi agent'a yönlendirmeli?
3. **Gerekçelendir**: Neden bu agent?

## Routing Kuralları
- "kimdir", "nedir", "nerede", "tarih", "bilgi" → **research**
- "hesapla", "kaç", "matematik", "topla", "çarp" → **tool**
- "kod yaz", "python", "dosya", "program" → **code**
- "merhaba", "nasılsın", "teşekkür" → **finish**
- Eğer son mesaj bir TOOL SONUCU ise (web/wiki/hesaplama sonucu) → **finish** (doğrudan cevap ver)

## ÖNEMLİ: SADECE JSON FORMATINDA ÇIKTI VER

### DOĞRU ÖRNEKLER:
{"next": "research", "reason": "Kullanıcı tarihi bir bilgi istiyor", "confidence": 0.95}
{"next": "tool", "reason": "Matematiksel işlem gerekiyor", "confidence": 0.98}
{"next": "code", "reason": "Kod yazma talebi", "confidence": 0.92}
{"next": "finish", "reason": "Basit selamlama, doğrudan cevap", "confidence": 0.99}

### YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Karar: research (sebep: bilgi isteği)  ← metin yazma!
- {"next": "research"} sonrası açıklama ← ek metin!
- Markdown kod bloğu kullanma ← SADECE düz JSON!

SADECE VE SADECE düz JSON yaz. Başka hiçbir metin ekleme."""

MAX_ITERATIONS = 10


def supervisor_node(state: AgentState):
    """Supervisor node: Görevi analiz eder, routing kararı verir."""
    llm = get_llm_engine()
    iteration = state.get("iteration_count", 0)
    messages = state["messages"]

    if iteration > MAX_ITERATIONS:
        return {
            "final_answer": "Maksimum iterasyon sayısına ulaşıldı. Lütfen daha spesifik bir istekte bulunun.",
            "current_agent": "finish",
            "iteration_count": iteration + 1,
            "messages": [AIMessage(content="Maksimum iterasyona ulaşıldı.")]
        }

    # HIZLI FIX: Eğer son mesaj tool sonucu ise, doğrudan finish yap ve cevap üret
    if messages:
        last_msg = messages[-1]
        last_type = getattr(last_msg, 'type', '')
        
        is_tool_result = (
            last_type == 'tool' or 
            'ToolMessage' in str(type(last_msg))
        )
        
        if is_tool_result:
            final_prompt = "Yukarıdaki konuşma geçmişindeki TOOL SONUÇLARINI kullanarak kullanıcıya nihai yanıtı ver. Doğal Türkçe ile, net ve açık cevapla. JSON kullanma, doğrudan cevap ver."
            final = llm.chat_with_history(SUPERVISOR_PROMPT, final_prompt, messages, max_new_tokens=512, temperature=0.7)
            return {
                "final_answer": final,
                "current_agent": "finish",
                "messages": [AIMessage(content=final)],
                "iteration_count": iteration + 1
            }

    # Normal routing - ReAct pattern
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"""Konuşma geçmişi:
{context}

GÖREV: ReAct Pattern kullanarak karar ver.

ADIM 1 - Düşün: Kullanıcı ne istiyor?
ADIM 2 - Karar: Hangi agent? (research|code|tool|finish)
ADIM 3 - Gerekçe: Neden?

SADECE JSON ÇIKTI VER:
{{"next": "...", "reason": "...", "confidence": 0.XX}}"""

    raw = llm.chat(SUPERVISOR_PROMPT, user_prompt, max_new_tokens=100, temperature=0.0)

    decision = extract_json(raw) or {}
    next_agent = decision.get("next", "finish")
    reason = decision.get("reason", "varsayılan")

    if next_agent not in ("research", "code", "tool", "finish"):
        next_agent = "finish"

    if next_agent == "finish":
        final_prompt = "Yukarıdaki konuşma geçmişine dayanarak kullanıcıya nihai yanıtı ver. Doğal Türkçe ile cevapla. JSON kullanma, doğrudan cevap ver."
        final = llm.chat_with_history(SUPERVISOR_PROMPT, final_prompt, messages, max_new_tokens=512, temperature=0.7)
        return {
            "final_answer": final,
            "current_agent": "finish",
            "messages": [AIMessage(content=final)],
            "iteration_count": iteration + 1
        }

    return {
        "current_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] Karar: {next_agent}. Gerekçe: {reason}")],
        "iteration_count": iteration + 1
    }
