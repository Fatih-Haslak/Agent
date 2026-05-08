from langchain_core.messages import SystemMessage
from src.config import get_llm
from src.state import AgentState
from src.tools import code_exec, file_io


CODE_PROMPT = """Sen bir Code Agent'sın. Kod yazma, çalıştırma ve düzeltme işlemleri yaparsın.

Kullanılabilir araçlar:
- code_exec: Python, bash veya JavaScript kodu çalıştırır (kritik, onay gerektirir)
- file_io: Dosya okuma/yazma/silme işlemleri (yazma/silme kritiktir, onay gerektirir)

Kurallar:
1. Kod yazarken güvenliği ön planda tut.
2. Hata alırsan düzelt ve tekrar dene.
3. Dosya işlemlerinde sadece proje dizini içinde çalış.
4. Kodun çıktısını açıkça belirt."""


def code_node(state: AgentState):
    """Code node: Kod yazma, çalıştırma ve dosya işlemleri yapar."""
    llm = get_llm(temperature=0.1).bind_tools([code_exec, file_io])
    messages = [SystemMessage(content=CODE_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
        "current_agent": "code"
    }
