from langchain_core.messages import ToolMessage, AIMessage
from langgraph.types import interrupt
from src.state import AgentState
from src.tools.executor import execute_tool


def interrupt_node(state: AgentState):
    """Interrupt Node: Kritik tool çağrıları öncesi human-in-the-loop onay ister."""
    pending = state.get("pending_tool")
    if not pending:
        return {}
    
    approval = interrupt({
        "tool_name": pending.get("name"),
        "args": pending.get("arguments") or pending.get("args", {}),
        "message": (
            f"⚠️ Kritik tool çağrısı onayı gerekiyor:\n"
            f"   Tool: {pending.get('name')}\n"
            f"   Args: {pending.get('arguments') or pending.get('args', {})}\n"
            f"Devam etmek için 'evet', iptal için 'hayır' yazın."
        )
    })
    
    is_approved = str(approval).lower() in ("evet", "yes", "e", "y", "true", "1")
    
    if is_approved:
        try:
            result = execute_tool(pending)
        except Exception as e:
            result = f"Tool çalıştırma hatası: {str(e)}"
        return {
            "messages": [ToolMessage(
                content=str(result),
                tool_call_id=pending.get("id", "unknown")
            )],
            "pending_tool": None
        }
    else:
        return {
            "messages": [AIMessage(
                content=f"❌ Kullanıcı '{pending.get('name')}' tool çağrısını iptal etti."
            )],
            "pending_tool": None
        }
