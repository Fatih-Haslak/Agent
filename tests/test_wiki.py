import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from src.graph import graph

config = {'configurable': {'thread_id': 'wiki-test'}}
state = {
    'messages': [HumanMessage(content='Sergen Yalçın kimdir')],
    'tool_calls': [],
    'current_agent': 'supervisor',
    'iteration_count': 0,
    'final_answer': None,
    'pending_tool': None
}

final_answer = None
for event in graph.stream(state, config):
    for node_name, node_state in event.items():
        if node_name.startswith('__'): continue
        print(f'→ {node_name.upper()}')
        if node_state.get('final_answer'):
            final_answer = node_state['final_answer']

if final_answer:
    print(f'\n🤖 YANIT: {final_answer[:500]}...')
else:
    print('\n⚠️ Yanıt üretilemedi.')
