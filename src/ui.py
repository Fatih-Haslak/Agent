"""
Gradio Web UI for Multi-Agent System
=====================================
Chat interface with agent reasoning display and interrupt handling.
Compatible with Gradio 6.x - Stable Version
"""

import os
import sys
import traceback
import gradio as gr
from typing import List, Tuple, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.graph import graph
from src.config import get_llm_engine


class ChatSession:
    """Gradio session state wrapper."""
    
    def __init__(self):
        self.thread_id = f"gradio-session-{id(self)}"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        # Gradio 6.x format: [{"role": "user", "content": "..."}, ...]
        self.history: List[Dict[str, str]] = []
        self.pending_interrupt: Dict[str, Any] = None
        self.current_trace: List[str] = []


def format_trace(trace: List[str]) -> str:
    if not trace:
        return "Henüz çalışma izi yok."
    lines = []
    for step in trace:
        emoji = {
            "supervisor": "🗺️", "research": "📚", "code": "💻",
            "tool": "🔧", "tools": "⚙️", "interrupt": "⛔", "finish": "✅",
        }.get(step.lower(), "•")
        lines.append(f"{emoji} {step}")
    return "\n".join(lines)


def process_message(message: str, history: List[Dict[str, str]], session_id: str) -> Tuple[List[Dict[str, str]], str, str]:
    """Kullanıcı mesajını işler.
    
    Returns:
        (chat_history, trace_display, status)
    """
    try:
        if not message or not message.strip():
            return history, format_trace([]), "Boş mesaj."
        
        # Session yönetimi
        if not hasattr(process_message, "sessions"):
            process_message.sessions = {}
        
        session = process_message.sessions.get(session_id)
        if not session:
            session = ChatSession()
            process_message.sessions[session_id] = session
        
        # Interrupt onayı mı?
        if session.pending_interrupt:
            approval = message.strip().lower()
            is_approved = approval in ("evet", "yes", "e", "y", "true", "1", "onayla")
            resume_value = "evet" if is_approved else "hayır"
            
            try:
                for event in graph.stream(Command(resume=resume_value), session.config):
                    for node_name, node_state in event.items():
                        if node_name.startswith("__"):
                            continue
                        session.current_trace.append(f"{node_name.upper()}: çalıştı")
                        if node_state.get("final_answer"):
                            session.history.append({"role": "user", "content": "(onay)"})
                            session.history.append({"role": "assistant", "content": node_state["final_answer"]})
            except Exception as e:
                print(f"Interrupt resume hatası: {e}")
            
            session.pending_interrupt = None
            return session.history, format_trace(session.current_trace), "Onay işlendi."
        
        # Normal mesaj
        state = {
            "messages": [HumanMessage(content=message)],
            "tool_calls": [],
            "current_agent": "supervisor",
            "iteration_count": 0,
            "final_answer": None,
            "pending_tool": None
        }
        
        session.current_trace = []
        final_answer = None
        
        for event in graph.stream(state, session.config):
            for node_name, node_state in event.items():
                if node_name.startswith("__"):
                    continue
                
                session.current_trace.append(f"{node_name.upper()}: çalıştı")
                
                # Tool çağrılarını izle
                if node_state.get("tool_calls"):
                    for tc in node_state["tool_calls"]:
                        session.current_trace.append(f"   → Tool: {tc.get('name', 'unknown')}")
                
                # Interrupt yakalama
                if "__interrupt__" in event:
                    interrupt_info = event["__interrupt__"][0]
                    value = interrupt_info.value
                    session.pending_interrupt = value
                    session.current_trace.append(f"⛔ Interrupt: {value.get('tool_name', 'unknown')}")
                    status = f"⛔ ONAY GEREKLİ: {value.get('tool_name', 'unknown')} - 'evet' veya 'hayır' yazın"
                    return session.history, format_trace(session.current_trace), status
                
                # Final answer - EVENT'TEN al (state objesi değişmiyor!)
                if node_state.get("final_answer"):
                    final_answer = node_state["final_answer"]
                
                # AI mesajlarını da kontrol et
                if not final_answer and node_state.get("messages"):
                    for m in reversed(node_state["messages"]):
                        content = getattr(m, 'content', '')
                        if content and not content.startswith("[") and len(content) > 5:
                            final_answer = content
                            break
        
        # Cevabı history'ye ekle
        if final_answer:
            session.history.append({"role": "user", "content": message})
            session.history.append({"role": "assistant", "content": final_answer})
        else:
            session.history.append({"role": "user", "content": message})
            session.history.append({"role": "assistant", "content": "Yanıt üretilemedi."})
        
        trace_text = format_trace(session.current_trace)
        status = "✅ Yanıt hazır."
        
        if session.pending_interrupt:
            status = f"⛔ ONAY GEREKLİ: {session.pending_interrupt.get('tool_name', 'unknown')}"
        
        return session.history, trace_text, status
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"UI HATASI: {error_detail}")
        return history, format_trace([]), f"❌ Hata: {str(e)}"


def clear_chat():
    """Chat'i temizler."""
    if hasattr(process_message, "sessions"):
        process_message.sessions.clear()
    return [], "", "✅ Yeni oturum başlatıldı."


# ── Gradio UI ─────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="🤖 Multi-Agent System") as demo:
        gr.Markdown("""
        # 🤖 Multi-Agent System with LangGraph
        
        **Model:** `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` (4-bit Quantization)
        
        **Agents:** Supervisor → Research | Code | Tool
        
        **Özellikler:** Human-in-the-loop onay • SQLite bellek • Türkçe
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 Konuşma",
                    height=500,
                    value=[],
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Mesajınız",
                        placeholder="Bir şeyler yazın... (Interrupt onayı için 'evet' veya 'hayır' yazın)",
                        scale=8,
                    )
                    send_btn = gr.Button("➤ Gönder", scale=1, variant="primary")
                
                status_text = gr.Textbox(
                    label="Durum",
                    interactive=False,
                    value="Hazır. Mesajınızı yazın."
                )
            
            with gr.Column(scale=1):
                trace_display = gr.Textbox(
                    label="🔍 Ajan Çalışma İzleme",
                    interactive=False,
                    lines=20,
                    value="Henüz çalışma izi yok.",
                )
                
                clear_btn = gr.Button("🗑️ Yeni Oturum", variant="secondary")
                
                gr.Markdown("""
                ### 📋 Kısayollar
                - **Web arama:** "Atatürk kimdir?"
                - **Matematik:** "15 çarpı 23 kaç eder?"
                - **Kod:** "Python'da 1'den 10'a kadar sayıları yazdır"
                - **Dosya:** "test.txt dosyasına merhaba yaz"
                
                ### ⚠️ Interrupt
                Kritik tool'lar öncesinde onay istenir.
                """)
        
        # Event handlers - stateless approach
        send_btn.click(
            fn=process_message,
            inputs=[msg_input, chatbot, gr.State(value="default-session")],
            outputs=[chatbot, trace_display, status_text],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )
        
        msg_input.submit(
            fn=process_message,
            inputs=[msg_input, chatbot, gr.State(value="default-session")],
            outputs=[chatbot, trace_display, status_text],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, trace_display, status_text],
        )
    
    return demo


def main():
    print("=" * 60)
    print("🤖 Multi-Agent System Gradio UI")
    print("   Model: ytu-ce-cosmos/Turkish-Gemma-9b-v0.1")
    print("=" * 60)
    print("\n📥 LLM hazırlanıyor...")
    get_llm_engine()
    print("✅ Model hazır!\n")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
