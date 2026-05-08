from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.agents import supervisor_node, research_node, code_node, tool_agent_node
from src.nodes import tools_execution_node, interrupt_node
from src.memory import get_checkpointer


# ── StateGraph Builder ──────────────────────────────────────────────
builder = StateGraph(AgentState)

# Node'ları ekle
builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_node)
builder.add_node("code", code_node)
builder.add_node("tool", tool_agent_node)
builder.add_node("tools", tools_execution_node)
builder.add_node("interrupt", interrupt_node)

# Başlangıç edge'i
builder.add_edge(START, "supervisor")


# ── Supervisor Routing ──────────────────────────────────────────────
def supervisor_router(state: AgentState):
    """Supervisor'ın hangi node'a yönlendireceğini belirler."""
    return state.get("current_agent", "finish")


builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "research": "research",
        "code": "code",
        "tool": "tool",
        "finish": END
    }
)


# ── Agent Routing (tool call var mı?) ──────────────────────────────
def agent_router(state: AgentState):
    """Agent çalıştıktan sonra tool call varsa tools node'a, yoksa supervisor'a döner."""
    if state.get("tool_calls"):
        return "tools"
    return "supervisor"


for node_name in ("research", "code", "tool"):
    builder.add_conditional_edges(
        node_name,
        agent_router,
        {"tools": "tools", "supervisor": "supervisor"}
    )


# ── Tools Routing (kritik tool var mı?) ────────────────────────────
def tools_router(state: AgentState):
    """Tools node'dan sonra kritik pending tool varsa interrupt'a, yoksa supervisor'a."""
    if state.get("pending_tool"):
        return "interrupt"
    return "supervisor"


builder.add_conditional_edges(
    "tools",
    tools_router,
    {"interrupt": "interrupt", "supervisor": "supervisor"}
)

# Interrupt'tan her durumda supervisor'a dön
builder.add_edge("interrupt", "supervisor")


# ── Compile with Long-term Memory ──────────────────────────────────
checkpointer = get_checkpointer()
graph = builder.compile(checkpointer=checkpointer)
