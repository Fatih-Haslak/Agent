"""
Multi-Agent System CLI
======================
Komut satiri arayuzu. Web UI icin:

    streamlit run src/streamlit_app.py

"""

import os
import sys

# Proje kökünü Python path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.graph import graph
from src.config import get_llm_engine


def handle_stream(stream, config):
    """Graph stream'ini işler, interrupt'ları yakalar."""
    final_answer = None
    
    for event in stream:
        # Interrupt yakalama
        if "__interrupt__" in event:
            interrupt_info = event["__interrupt__"][0]
            value = interrupt_info.value
            print(f"\n⛔ ONAY GEREKLİ")
            print(f"   {value.get('message', '')}")
            
            approval = input("\nOnaylıyor musunuz? (evet/hayır): ").strip()
            
            # Resume with user decision
            resume_stream = graph.stream(Command(resume=approval), config)
            inner_answer = handle_stream(resume_stream, config)
            if inner_answer:
                final_answer = inner_answer
            break
        
        # Her node çalıştığında state'i kontrol et
        for node_name, node_state in event.items():
            if node_name.startswith("__"):
                continue
            
            if node_state.get("final_answer"):
                final_answer = node_state["final_answer"]
    
    return final_answer


def main():
    print("=" * 60)
    print("🤖 Multi-Agent System (LangGraph + Turkish-Gemma-9b)")
    print("   Supervisor → Research | Code | Tool")
    print("   Model: ytu-ce-cosmos/Turkish-Gemma-9b-v0.1")
    print("   Memory: Short-term (state) + Long-term (SQLite)")
    print("=" * 60)
    
    # Modeli önceden yükle (lazy init)
    print("\n📥 LLM hazırlanıyor...")
    get_llm_engine()
    
    print("\nÇıkmak için 'exit' yazın.\n")
    
    thread_id = "default-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input("Siz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşmek üzere!")
            break
        
        if user_input.lower() in ("exit", "quit", "q", "çık"):
            print("Görüşmek üzere!")
            break
        
        if not user_input:
            continue
        
        # State oluştur
        state = {
            "messages": [HumanMessage(content=user_input)],
            "tool_calls": [],
            "current_agent": "supervisor",
            "iteration_count": 0,
            "final_answer": None,
            "pending_tool": None
        }
        
        # Graph'ı çalıştır
        stream = graph.stream(state, config)
        answer = handle_stream(stream, config)
        
        if answer:
            print(f"\n🤖 Assistant: {answer}\n")
        else:
            print("\n🤖 Assistant: (Yanıt üretilemedi)\n")


if __name__ == "__main__":
    main()
