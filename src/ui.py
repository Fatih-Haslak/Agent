"""
Gradio Web UI for Multi-Agent System
=====================================
Chat interface with agent reasoning display and interrupt handling.
Compatible with Gradio 6.x
"""

import os
import sys
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
        self.thread_id = "gradio-session"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        # Gradio 6.x format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.history: List[Dict[str, str]] = []
        self.pending_interrupt: Dict[str, Any] = None
        self.current_trace: List[str] = []


# Global session store
sessions: Dict[str, ChatSession] = {}


def format_trace(trace: List[str]) -> str:
    """Execution trace'i formatlı metne dönüştürür."""
    if not trace:
        return "Henüz çalışma izi yok."
    lines = []
    for step in trace:
        emoji = {
            "supervisor": "🗺️",
            "research": "📚",
            "code": "💻",
            "tool": "🔧",
            "tools": "⚙️",
            "interrupt": "⛔",
            "finish": "✅",
        }.get(step.lower(), "•")
        lines.append(f"{emoji} {step}")
    return "\n".join(lines)


def process_message(message: str, session_id: str) -> Tuple[List[Dict[str, str]], str, str]:
    """Kullanıcı mesajını işler ve yanıt döndürür.
    
    Returns:
        (chat_history, trace_display, status)
    """
    if not message.strip():
        return [], "", "Boş mesaj gönderilemez."
    
    session = sessions.get(session_id)
    if not session:
        session = ChatSession()
        sessions[session_id] = session
    
    # Eğer bekleyen interrupt varsa, bu mesaj onay/ret olarak işlenir
    if session.pending_interrupt:
        approval = message.strip().lower()
        is_approved = approval in ("evet", "yes", "e", "y", "true", "1", "onayla")
        
        resume_value = "evet" if is_approved else "hayır"
        stream = graph.stream(Command(resume=resume_value), session.config)
        
        result = _collect_stream(stream, session)
        
        trace_text = format_trace(session.current_trace)
        status = f"Interrupt {'onaylandı' if is_approved else 'reddedildi'}."
        return session.history, trace_text, status
    
    # Normal mesaj akışı
    state = {
        "messages": [HumanMessage(content=message)],
        "tool_calls": [],
        "current_agent": "supervisor",
        "iteration_count": 0,
        "final_answer": None,
        "pending_tool": None
    }
    
    session.current_trace = []
    stream = graph.stream(state, session.config)
    
    result = _collect_stream(stream, session)
    
    # Yanıtı history'ye ekle (Gradio 6.x formatı)
    final_answer = result.get("final_answer", "")
    if not final_answer:
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            content = getattr(m, 'content', str(m))
            if content and not content.startswith("["):
                final_answer = content
                break
    
    # Gradio 6.x mesaj formatı
    session.history.append({"role": "user", "content": message})
    session.history.append({"role": "assistant", "content": final_answer or "Yanıt üretilemedi."})
    
    trace_text = format_trace(session.current_trace)
    
    if session.pending_interrupt:
        status = f"⛔ ONAY GEREKLİ: {session.pending_interrupt.get('tool_name', 'unknown')}"
    else:
        status = "✅ Yanıt hazır."
    
    return session.history, trace_text, status


def _collect_stream(stream, session: ChatSession) -> Dict[str, Any]:
    """Graph stream'ini toplar ve interrupt'ları yakalar."""
    result = {}
    
    for event in stream:
        if "__interrupt__" in event:
            interrupt_info = event["__interrupt__"][0]
            value = interrupt_info.value
            session.pending_interrupt = value
            session.current_trace.append(f"⛔ Interrupt: {value.get('tool_name', 'unknown')}")
            result["pending_interrupt"] = True
            break
        
        for node_name, node_state in event.items():
            if node_name.startswith("__"):
                continue
            
            session.current_trace.append(f"{node_name.upper()}: çalıştı")
            
            if node_state.get("final_answer"):
                result["final_answer"] = node_state["final_answer"]
            
            if node_state.get("messages"):
                result["messages"] = node_state.get("messages", [])
            
            if node_state.get("tool_calls"):
                for tc in node_state["tool_calls"]:
                    tool_name = tc.get("name", "unknown")
                    session.current_trace.append(f"   → Tool: {tool_name}")
    
    return result


def clear_session(session_id: str) -> Tuple[List[Dict[str, str]], str, str]:
    """Session'ı temizler."""
    sessions[session_id] = ChatSession()
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
        
        session_id = gr.State(value="default-gradio-session")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 Konuşma",
                    height=500,
                    type="messages",
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
                Kritik tool'lar (kod çalıştırma, dosya yazma, HTTP) öncesinde onay istenir.
                """)
        
        # Event handlers
        send_btn.click(
            fn=process_message,
            inputs=[msg_input, session_id],
            outputs=[chatbot, trace_display, status_text],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )
        
        msg_input.submit(
            fn=process_message,
            inputs=[msg_input, session_id],
            outputs=[chatbot, trace_display, status_text],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )
        
        clear_btn.click(
            fn=clear_session,
            inputs=[session_id],
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
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
