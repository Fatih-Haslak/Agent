from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


CODE_PROMPT = """Sen bir Code Agent'sın. Kod yazma, çalıştırma ve düzeltme işlemleri yaparsın.

Kullanılabilir araçlar:
- code_exec(language: str, code: str, timeout: int = 10)
- file_io(action: str, file_path: str, content: Optional[str] = None)

KURAL:
1. EĞER konuşma geçmişinde zaten bir code_exec veya file_io SONUCU varsa (ToolMessage ile çalıştırma sonucu):
   - Bu sonucu kullanarak doğrudan kullanıcıya cevap ver.
   - TEKRAR code_exec veya file_io çağrısı yapma!
   - Sonucu analiz et, açıkla, Türkçe olarak cevapla.

2. EĞER henüz kod çalıştırılmadıysa:
   - Araç kullanman gerekiyorsa, KESİNLİKLE JSON formatında tool çağrısı yap.

DOĞRU ÖRNEK (araç kullanımı):
{"tool": "code_exec", "args": {"language": "python", "code": "print(1+1)"}}

DOĞRU ÖRNEK (sonucu değerlendirme):
Kod çalıştırıldı ve sonuç 2 olarak döndü. Bu işlem...

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Tekrar araç çağrısı yapmak (eğer zaten sonuç varsa)
- ```json\n{...}\n```  ← kod bloğu kullanma!

ÖNEMLİ: 
- Zaten çalıştırma sonucu varsa SADECE cevap ver.
- Çalıştırma yoksa SADECE JSON yaz."""


def code_node(state: AgentState):
    """Code node: Kod yazma, çalıştırma ve dosya işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-10:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content[:500]}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Zaten çalıştırma sonucu varsa onu kullan. Yoksa SADECE JSON yaz."
    raw = llm.chat(CODE_PROMPT, user_prompt, max_new_tokens=400, temperature=0.1)

    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Code] Tool çağrısı: {tool_call['tool']}")],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_code"}],
            "current_agent": "code"
        }

    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "code"
    }
