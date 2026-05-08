import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from src.graph import graph

config = {'configurable': {'thread_id': 'code-test'}}
state = {
    'messages': [HumanMessage(content='Python ile 1den 10a kadar sayıları yazdır')],
    'tool_calls': [],
    'current_agent': 'supervisor',
    'iteration_count': 0,
    'final_answer': None,
    'pending_tool': None
}

final_answer = None
pending_interrupt = None

stream = graph.stream(state, config)
for event in stream:
    for node_name, node_state in event.items():
        if node_name.startswith('__'): continue
        print(f'→ {node_name.upper()}')
        if node_state.get('tool_calls'):
            for tc in node_state['tool_calls']:
                print(f'   → TOOL: {tc.get("name")} args={tc.get("args")}')
        if node_state.get('final_answer'):
            final_answer = node_state['final_answer']
    
    # Interrupt kontrolü
    if '__interrupt__' in event:
        interrupt_info = event['__interrupt__'][0]
        value = interrupt_info.value
        print(f'\n⛔ INTERRUPT: {value.get("tool_name")}')
        print(f'   Args: {value.get("args")}')
        print(f'   → Auto-approving for test...\n')
        
        # Test için otomatik onay
        stream = graph.stream(Command(resume='evet'), config)
        for event2 in stream:
            for node_name, node_state in event2.items():
                if node_name.startswith('__'): continue
                print(f'→ {node_name.upper()}')
                if node_state.get('final_answer'):
                    final_answer = node_state['final_answer']

if final_answer:
    print(f'\n🤖 YANIT: {final_answer[:500]}')
else:
    print('\n⚠️ Yanıt üretilemedi.')
