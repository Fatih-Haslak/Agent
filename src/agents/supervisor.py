from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


SUPERVISOR_PROMPT = """Sen bir Supervisor Agent'sın. Kullanıcının isteğini analiz eder ve en uygun uzman agent'a yönlendirirsin.

Mevcut agent'lar:
- research: Web araması, bilgi toplama, özetleme gerektiren görevler
- code: Kod yazma, çalıştırma, hata ayıklama, dosya işlemleri
- tool: Matematiksel hesaplama, API çağrıları, veri işleme
- finish: Görev tamamlandı, nihai yanıt hazır

KARAR KURALLARI:
- Bilgi sorusu, araştırma, kimdir/nedir → research
- Kod yazma, dosya işlemi, hesaplama → code
- Matematiksel işlem, API çağrısı → tool
- Basit selamlama, teşekkür, kısa sohbet → finish
- Eğer yeterli bilgi toplandıysa ve cevap hazırsa → finish

ÖNEMLİ: SADECE ve SADECE aşağıdaki JSON formatında çıktı ver. Başka hiçbir metin yazma.

DOĞRU ÖRNEK:
{"next": "research", "reason": "Kullanıcı tarihi bir bilgi istiyor"}

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Karar: research (sebep: bilgi isteği)  ← metin yazma!
- research  ← tek kelime yazma!
- {"next": "research", "reason": "..."} ve ardından açıklama ← ek metin yazma!

SADECE JSON ÇIKTI VER."""

MAX_ITERATIONS = 10


def supervisor_node(state: AgentState):
    """Supervisor node: Görevi analiz eder, routing kararı verir."""
    llm = get_llm_engine()
    iteration = state.get("iteration_count", 0)

    if iteration > MAX_ITERATIONS:
        return {
            "final_answer": "Maksimum iterasyon sayısına ulaşıldı. Lütfen daha spesifik bir istekte bulunun.",
            "current_agent": "finish",
            "iteration_count": iteration + 1,
            "messages": [AIMessage(content="Maksimum iterasyona ulaşıldı.")]
        }

    messages = state["messages"]
    context = ""
    for m in messages[-6:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nRouting kararını ver. SADECE JSON."
    raw = llm.chat(SUPERVISOR_PROMPT, user_prompt, max_new_tokens=80, temperature=0.0)

    decision = extract_json(raw) or {}
    next_agent = decision.get("next", "finish")
    reason = decision.get("reason", "varsayılan")

    if next_agent not in ("research", "code", "tool", "finish"):
        next_agent = "finish"

    if next_agent == "finish":
        final_prompt = "Yukarıdaki konuşma geçmişine dayanarak kullanıcıya nihai yanıtı ver. Doğal Türkçe ile cevapla."
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
