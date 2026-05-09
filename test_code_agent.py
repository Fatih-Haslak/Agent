#!/usr/bin/env python3
"""Test script for code agent fix"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.agents.code import code_node
from src.state import AgentState
from langchain_core.messages import HumanMessage

def test_code_agent():
    # Test state
    state = AgentState(
        messages=[HumanMessage(content='Python ile merhaba dünya yazdır')],
        current_agent='code',
        iteration_count=0
    )

    print("Testing code agent...")
    result = code_node(state)
    print('Result keys:', list(result.keys()))
    print('Tool calls:', result.get('tool_calls', []))
    print('Current agent:', result.get('current_agent'))

    if result.get('tool_calls'):
        tool_call = result['tool_calls'][0]
        print(f"Tool name: {tool_call.get('name')}")
        print(f"Tool args: {tool_call.get('args')}")

if __name__ == "__main__":
    test_code_agent()