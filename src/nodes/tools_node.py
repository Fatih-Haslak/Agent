from langchain_core.messages import ToolMessage, AIMessage
from src.state import AgentState
from src.tools.executor import execute_tool, is_critical


def tools_execution_node(state: AgentState):
    """Tool node: Agent'lar tarafından üretilen tool çağrılarını çalıştırır.
    
    Eğer kritik bir tool varsa, çalıştırmadan önce interrupt_node'a 
    yönlendirmek için pending_tool state'ine yazar.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"current_agent": "supervisor", "tool_calls": [], "messages": []}
    
    pending = None
    safe_results = []
    safe_messages = []
    
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
            safe_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc.get("id", "unknown")
            ))
    
    if pending:
        return {
            "messages": safe_messages,
            "tool_calls": [],
            "pending_tool": pending,
            "current_agent": "tools"
        }
    
    return {
        "messages": safe_messages,
        "tool_calls": [],
        "pending_tool": None,
        "current_agent": "supervisor"
    }
