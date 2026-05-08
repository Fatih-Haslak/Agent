from langchain_core.messages import ToolMessage
from src.state import AgentState
from src.tools.executor import execute_tool, is_critical


def tools_execution_node(state: AgentState):
    """Tool node: Agent'lar tarafından üretilen tool çağrılarını çalıştırır.
    
    Eğer kritik bir tool varsa, çalıştırmadan önce interrupt_node'a 
    yönlendirmek için pending_tool state'ine yazar.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"current_agent": "supervisor", "tool_calls": []}
    
    # İlk tool call'ı işle (sırayla)
    # Çoklu tool call desteği için loop kullanılabilir, 
    # ancak şimdilik ilk kritik olanı yakalama stratejisi uyguluyoruz.
    pending = None
    safe_results = []
    safe_ids = []
    
    for tc in tool_calls:
        if is_critical(tc):
            pending = tc
            break
        else:
            try:
                result = execute_tool(tc)
            except Exception as e:
                result = f"Tool hatası ({tc.get('name')}): {str(e)}"
            safe_results.append(result)
            safe_ids.append(tc.get("id", "unknown"))
    
    # Güvenli tool sonuçlarını messages'a ekle
    messages = []
    for res, tid in zip(safe_results, safe_ids):
        messages.append(ToolMessage(content=str(res), tool_call_id=tid))
    
    if pending:
        return {
            "messages": messages,
            "tool_calls": [],
            "pending_tool": pending,
            "current_agent": "tools"
        }
    
    return {
        "messages": messages,
        "tool_calls": [],
        "pending_tool": None,
        "current_agent": "supervisor"
    }
