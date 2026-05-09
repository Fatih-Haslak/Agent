import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from src.graph import graph

config = {'configurable': {'thread_id': 'code-debug-new'}}
state = {
    'messages': [HumanMessage(content='python ile merhaba yaz')],
    'tool_calls': [],
    'current_agent': 'supervisor',
    'iteration_count': 0,
    'final_answer': None,
    'pending_tool': None
}

try:
    for event in graph.stream(state, config):
        print(f"EVENT: {list(event.keys())}")
        for node_name, node_state in event.items():
            if node_name.startswith('__'): 
                print(f"  INTERNAL: {node_name}")
                continue
            print(f'  NODE: {node_name}')
            print(f'    current_agent: {node_state.get("current_agent", "N/A")}')
            print(f'    pending_tool: {str(node_state.get("pending_tool", "N/A"))[:80]}')
            print(f'    final_answer: {str(node_state.get("final_answer", "N/A"))[:80]}')
            msgs = node_state.get('messages', [])
            if msgs:
                for m in msgs[-1:]:
                    print(f'    msg: {getattr(m, "type", type(m))} = {str(getattr(m, "content", str(m)))[:80]}')
        
        # Interrupt kontrolü
        if '__interrupt__' in event:
            print("\n  >>> INTERRUPT DETECTED <<<")
            interrupt_info = event['__interrupt__'][0]
            value = interrupt_info.value
            print(f"    tool: {value.get('tool_name')}")
            print(f"    args: {value.get('args')}")
            print("    Auto-approving...")
            
            # Resume
            for event2 in graph.stream(Command(resume='evet'), config):
                print(f"  RESUME EVENT: {list(event2.keys())}")
                for node_name, node_state in event2.items():
                    if node_name.startswith('__'): continue
                    print(f'    NODE: {node_name}')
                    print(f'      final_answer: {str(node_state.get("final_answer", "N/A"))[:80]}')
            break
        
        print()
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
