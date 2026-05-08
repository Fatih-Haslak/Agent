import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from src.graph import graph

config = {'configurable': {'thread_id': 'debug-test'}}
state = {
    'messages': [HumanMessage(content='selam')],
    'tool_calls': [],
    'current_agent': 'supervisor',
    'iteration_count': 0,
    'final_answer': None,
    'pending_tool': None
}

print("=== DEBUG: Graph Event'leri ===\n")
for event in graph.stream(state, config):
    print(f"EVENT TYPE: {type(event)}")
    print(f"EVENT KEYS: {list(event.keys())}")
    for node_name, node_state in event.items():
        if node_name.startswith('__'): 
            print(f"  INTERNAL: {node_name} = {node_state}")
            continue
        print(f"\n  NODE: {node_name}")
        print(f"  STATE TYPE: {type(node_state)}")
        print(f"  STATE KEYS: {list(node_state.keys()) if hasattr(node_state, 'keys') else 'N/A'}")
        if hasattr(node_state, 'get'):
            print(f"  final_answer: {node_state.get('final_answer', 'YOK')}")
            print(f"  current_agent: {node_state.get('current_agent', 'YOK')}")
            msgs = node_state.get('messages', [])
            if msgs:
                for i, m in enumerate(msgs[-2:]):
                    print(f"  message[{i}]: {getattr(m, 'type', type(m))} = {getattr(m, 'content', str(m))[:100]}")
    print("\n" + "="*50 + "\n")

print("\n=== STATE AFTER GRAPH ===")
print(f"final_answer: {state.get('final_answer', 'YOK')}")
print(f"messages count: {len(state.get('messages', []))}")
for i, m in enumerate(state.get('messages', [])):
    print(f"  msg[{i}]: {getattr(m, 'type', type(m))} = {getattr(m, 'content', str(m))[:100]}")
