from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


CODE_PROMPT = """Sen bir Code Agent'sın. Kod yazma, çalıştırma ve düzeltme işlemleri yaparsın.

Kullanılabilir araçlar:
- code_exec(language: str, code: str, timeout: int = 10)
- file_io(action: str, file_path: str, content: Optional[str] = None)

KURAL: Eğer araç kullanman gerekiyorsa, KESİNLİKLE aşağıdaki JSON formatında tool çağrısı yap.
Eğer araç kullanmana gerek yoksa, doğrudan kod yaz veya yanıt ver.

DOĞRU ÖRNEK (araç kullanımı):
{"tool": "code_exec", "args": {"language": "python", "code": "print(sum(range(10)))"}}

DOĞRU ÖRNEK (doğrudan yanıt):
Python'da liste toplamı `sum()` fonksiyonu ile yapılır.

YANLIŞ ÖRNEKLER (ASLA YAPMA):
- Kodu çalıştırıyorum: {"tool": ...}  ← JSON öncesi metin yazma!
- ```json\n{...}\n```  ← kod bloğu kullanma!
- code_exec(language="python", code="...")  ← fonksiyon çağrısı yazma!

ÖNEMLİ: Tool çağrısı yapacaksan SADECE tek satır JSON yaz. Doğrudan yanıt vereceksen SADECE yanıtını yaz."""


def code_node(state: AgentState):
    """Code node: Kod yazma, çalıştırma ve dosya işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Tool kullanacaksan SADECE JSON yaz."
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
