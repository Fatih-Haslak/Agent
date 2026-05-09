import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from src.graph import graph

config = {'configurable': {'thread_id': 'code-debug'}}
state = {
    'messages': [HumanMessage(content='python ile merhaba yaz')],
    'tool_calls': [],
    'current_agent': 'supervisor',
    'iteration_count': 0,
    'final_answer': None,
    'pending_tool': None
}

for event in graph.stream(state, config):
    for node_name, node_state in event.items():
        if node_name.startswith('__'): continue
        print(f'NODE: {node_name}')
        print(f'  current_agent: {node_state.get("current_agent", "N/A")}')
        print(f'  final_answer: {str(node_state.get("final_answer", "N/A"))[:100]}')
        print(f'  tool_calls: {node_state.get("tool_calls", [])}')
        msgs = node_state.get('messages', [])
        if msgs:
            for m in msgs[-2:]:
                print(f'  msg: {getattr(m, "type", type(m))} = {str(getattr(m, "content", str(m)))[:80]}')
        print()
