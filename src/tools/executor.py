from typing import Dict, Any
from src.tools import web_search, summarize, wiki_search, code_exec, file_io, calculator, http_request

TOOL_MAP = {
    "web_search": web_search,
    "summarize": summarize,
    "wiki_search": wiki_search,
    "code_exec": code_exec,
    "file_io": file_io,
    "calculator": calculator,
    "http_request": http_request,
}

CRITICAL_TOOLS = {"code_exec", "http_request", "file_io"}


def is_critical(tool_call: Dict[str, Any]) -> bool:
    """Bir tool çağrısının kritik (human-in-the-loop onay gerektiren) olup olmadığını kontrol eder."""
    name = tool_call.get("name", "")
    if name not in CRITICAL_TOOLS:
        return False
    if name == "file_io":
        args = tool_call.get("arguments") or tool_call.get("args", {})
        if isinstance(args, dict):
            action = args.get("action", "")
            if action in ("read",):
                return False
    return True


def execute_tool(tool_call: Dict[str, Any]) -> Any:
    """Tool çağrısını çalıştırır ve sonucu döndürür."""
    name = tool_call.get("name")
    args = tool_call.get("arguments") or tool_call.get("args", {})
    
    if name not in TOOL_MAP:
        raise ValueError(f"Bilinmeyen tool: {name}")
    
    tool = TOOL_MAP[name]
    if isinstance(args, dict):
        result = tool.invoke(args)
    else:
        result = tool.invoke({"input": args})
    return result
