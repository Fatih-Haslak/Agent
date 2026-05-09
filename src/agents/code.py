from langchain_core.messages import AIMessage
from src.config import get_llm_engine, extract_json
from src.state import AgentState


CODE_PROMPT = """Sen bir Code Agent'sın. Kullanıcının istediği kodu yazarsın ve çalıştırırsın.

## Kullanılabilir Araçlar
- code_exec(language, code): Python/Bash/JS kodu çalıştırır
- file_io(action, file_path, content): Dosya okuma/yazma/silme

## ReAct Pattern (Düşün → Hareket Et → Gözlemle)
1. **Düşün**: Kullanıcı ne tür kod istiyor?
2. **Hareket Et**: code_exec ile kodu çalıştır
3. **Gözlemle**: Çıktıyı analiz et

## KURAL: Kod istendiğinde KESİNLİKLE code_exec kullan
Asla sadece kod yazıp cevap verme! Tool çağırması yap.

## ÖNEMLİ: SADECE JSON FORMATINDA

### DOĞRU ÖRNEK:
{"tool": "code_exec", "args": {"language": "python", "code": "for i in range(5): print(i)"}}

### YANLIŞ ÖRNEKLER:
- Doğrudan kod yazma
- Markdown kod bloğu kullanma
- Fonksiyon çağrısı yazma

SADECE düz JSON."""


def code_node(state: AgentState):
    """Code node: Kod yazma, çalıştırma ve dosya işlemleri yapar."""
    llm = get_llm_engine()
    messages = state["messages"]
    context = ""
    for m in messages[-10:]:
        role = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', str(m))
        context += f"{role}: {content[:500]}\n"

    user_prompt = f"""Konuşma geçmişi:
{context}

ReAct Pattern:
ADIM 1 - Düşün: Kullanıcı ne tür kod istiyor?
ADIM 2 - Karar: code_exec tool'u çağırmalı mıyım?
ADIM 3 - Eğer kod çalıştıracaksam SADECE JSON yaz. Yoksa doğrudan cevap ver."""

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
