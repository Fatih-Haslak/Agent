import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# Proje kökünü Python path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph import graph


load_dotenv()


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
            print(f"   Tool : {value.get('tool_name')}")
            print(f"   Args : {value.get('args', {})}")
            
            approval = input("\nOnaylıyor musunuz? (evet/hayır): ").strip()
            
            # Resume with user decision
            resume_stream = graph.stream(Command(resume=approval), config)
            # Recursive handle
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
    print("🤖 Multi-Agent System (LangGraph)")
    print("   Supervisor → Research | Code | Tool")
    print("   Memory: Short-term (state) + Long-term (SQLite)")
    print("   Interrupt: Human-in-the-loop onay")
    print("=" * 60)
    print("Çıkmak için 'exit' yazın.\n")
    
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
