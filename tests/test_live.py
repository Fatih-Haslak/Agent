"""
Canlı test scripti - Multi-Agent System
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from src.graph import graph
from src.config import get_llm_engine


def test_conversation():
    print("=" * 60)
    print("🧪 CANLI TEST BAŞLIYOR")
    print("=" * 60)
    
    # Modeli önceden yükle
    print("\n📥 Model yükleniyor...")
    get_llm_engine()
    print("✅ Model hazır!\n")
    
    tests = [
        "Merhaba, nasılsın?",
        "15 çarpı 23 kaç eder?",
        "Atatürk kimdir?",
    ]
    
    for i, query in enumerate(tests, 1):
        print(f"\n{'─' * 60}")
        print(f"🧑 TEST {i}/{len(tests)}: {query}")
        print("─" * 60)
        
        # Her test için YENİ thread_id (önceki konuşmaları karıştırmamak için)
        config = {"configurable": {"thread_id": f"test-session-{i}"}}
        
        state = {
            "messages": [HumanMessage(content=query)],
            "tool_calls": [],
            "current_agent": "supervisor",
            "iteration_count": 0,
            "final_answer": None,
            "pending_tool": None
        }
        
        final_answer = None
        trace = []
        for event in graph.stream(state, config):
            for node_name, node_state in event.items():
                if node_name.startswith("__"):
                    continue
                trace.append(node_name.upper())
                print(f"   → {node_name.upper()}: çalıştı")
                if node_state.get("final_answer"):
                    final_answer = node_state["final_answer"]
        
        print(f"\n   İzleme: {' → '.join(trace)}")
        
        if final_answer:
            print(f"\n🤖 YANIT: {final_answer[:300]}...")
        else:
            print("\n⚠️ Yanıt üretilemedi.")
    
    print(f"\n{'=' * 60}")
    print("✅ TEST TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    test_conversation()
