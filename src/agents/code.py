from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


CODE_PROMPT = """Sen bir Code Agent'sın. Kullanıcının istediği kodu yazarsın ve çalıştırırsın.

Mevcut araçların:
- code_exec(language, code): Kodu çalıştırır
- file_io(action, file_path, content): Dosya işlemleri

KURAL:
1. Kullanıcı kod istediğinde, ÖNCE code_exec ile kodu ÇALIŞTIR.
2. Kodu doğrudan yazıp cevap verme! code_exec toolunu KULLAN.
3. SADECE aşağıdaki JSON formatında tool çağrısı yap:

DOĞRU ÖRNEK (kod çalıştırma):
{"name": "code_exec", "args": {"language": "python", "code": "for i in range(5): print(i)"}}

DOĞRU ÖRNEK (dosya yazma):
{"name": "file_io", "args": {"action": "write", "file_path": "merhaba.py", "content": "print('Merhaba')"}}

YANLIŞ ÖRNEK (ASLA YAPMA):
- Doğrudan kod yazıp cevap verme!
- Markdown kod bloğu kullanma!
- Fonksiyon çağrısı yazma, SADECE JSON!

ÖNEMLİ: 
- Kod istendiğinde SADECE JSON yaz.
- Araç kullanmana gerek yoksa (doğrudan yanıt) SADECE yanıtını yaz."""


def code_node(state: AgentState):
    """Code node: Kod yazma, çalıştırma ve dosya işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-8:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content}\n"

    user_prompt = f"Konuşma geçmişi:\n{context}\n\nGörevi tamamla. Kod isteniyorsa SADECE code_exec JSON yaz. Yoksa doğrudan cevap ver."
    raw = llm.chat(CODE_PROMPT, user_prompt, max_new_tokens=400, temperature=0.1)

    # Markdown kod bloklarını da parse et
    tool_call = extract_json(raw)
    if tool_call and "name" in tool_call:
        return {
            "messages": [AIMessage(content=f"[Code] Tool çağrısı: {tool_call['name']}")],
            "tool_calls": [{"name": tool_call["name"], "args": tool_call.get("args", {}), "id": "tc_code"}],
            "current_agent": "code"
        }

    return {
        "messages": [AIMessage(content=raw)],
        "tool_calls": [],
        "current_agent": "code"
    }
