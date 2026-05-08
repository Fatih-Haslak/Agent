from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


CODE_PROMPT = """Sen bir Code Agent'sın. Kod yazma, çalıştırma ve düzeltme işlemleri yaparsın.

Kullanılabilir araçlar:
- code_exec: Python, bash veya JavaScript kodu çalıştırır (kritik, onay gerektirir). Argüman: {"language": "python", "code": "..."}
- file_io: Dosya okuma/yazma/silme. Argüman: {"action": "read|write|delete|append", "file_path": "...", "content": "..."}

Eğer araç kullanman gerekiyorsa, JSON formatında tool çağrısı yap:
{"tool": "code_exec", "args": {"language": "python", "code": "print(1+1)"}}

Eğer araç kullanmana gerek yoksa, doğrudan kod yaz veya yanıt ver.
Yanıtın başında JSON kullanma, doğrudan cevap ver."""


def code_node(state: AgentState):
    """Code node: Kod yazma, çalıştırma ve dosya işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla."
    raw = llm.chat(CODE_PROMPT, user_prompt, max_new_tokens=400, temperature=0.2)

    tool_call = extract_json(raw)
    if tool_call and "tool" in tool_call:
        return {
            "messages": [AIMessage(content=raw)],
            "tool_calls": [{"name": tool_call["tool"], "args": tool_call.get("args", {}), "id": "tc_code"}],
            "current_agent": "code"
        }

    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "code"
    }
