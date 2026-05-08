from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage
from src.config import get_llm
from src.state import AgentState


class RoutingDecision(BaseModel):
    """Supervisor'ın routing kararı."""
    next: Literal["research", "code", "tool", "finish"] = Field(
        description="Hangi agent'ın çalışması gerektiği: research, code, tool, veya finish"
    )
    reason: str = Field(description="Kararın gerekçesi")


SUPERVISOR_PROMPT = """Sen bir Supervisor Agent'sın. Kullanıcının isteğini analiz eder ve en uygun uzman agent'a yönlendirirsin.

Mevcut agent'lar:
- research: Web araması, bilgi toplama, özetleme gerektiren görevler
- code: Kod yazma, çalıştırma, hata ayıklama, dosya işlemleri
- tool: Matematiksel hesaplama, API çağrıları, veri işleme
- finish: Görev tamamlandı, nihai yanıt hazır

Karar verirken şunları dikkate al:
1. Görevin doğası (bilgi mi, kod mu, hesaplama mı?)
2. Önceki adımlar ve mevcut durum
3. Eğer yeterli bilgi toplandıysa ve nihai yanıt verilebiliyorsa 'finish' seç

Önemli: Sadece 'next' ve 'reason' alanlarını doldur."""

MAX_ITERATIONS = 10


def supervisor_node(state: AgentState):
    """Supervisor node: Görevi analiz eder, routing kararı verir, yanıt birleştirir."""
    llm = get_llm(temperature=0)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    
    iteration = state.get("iteration_count", 0)
    if iteration > MAX_ITERATIONS:
        return {
            "final_answer": "Maksimum iterasyon sayısına ulaşıldı. Lütfen daha spesifik bir istekte bulunun.",
            "current_agent": "finish",
            "iteration_count": iteration + 1
        }
    
    try:
        decision = llm.with_structured_output(RoutingDecision).invoke(messages)
    except Exception as e:
        # Fallback: düz metin yanıt
        return {
            "current_agent": "finish",
            "final_answer": f"Routing hatası: {str(e)}",
            "iteration_count": iteration + 1
        }
    
    if decision.next == "finish":
        final = llm.invoke(messages)
        return {
            "final_answer": final.content,
            "current_agent": "finish",
            "messages": [final],
            "iteration_count": iteration + 1
        }
    
    return {
        "current_agent": decision.next,
        "messages": [AIMessage(content=f"[Supervisor] Karar: {decision.next}. Gerekçe: {decision.reason}")],
        "iteration_count": iteration + 1
    }
