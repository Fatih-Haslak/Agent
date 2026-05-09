"""
Gradio Web UI for Multi-Agent System with Streaming
====================================================
Chat interface with streaming, ReAct pattern, and interrupt handling.

Anti-flicker fixes for Gradio 6.x:
- bubble_full_width=False on Chatbot
- Explicit CSS to lock container heights and prevent layout shifts
- container=False on inner components where possible
"""

import os
import sys
import re
import traceback
import gradio as gr
from typing import List, Dict, Any, Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from src.graph import graph
from src.config import get_llm_engine


# ── Session Management ──────────────────────────────────────────────

class SessionManager:
    _sessions: Dict[str, "ChatSession"] = {}

    @classmethod
    def get_or_create(cls, session_id: str) -> "ChatSession":
        if session_id not in cls._sessions:
            cls._sessions[session_id] = ChatSession()
        return cls._sessions[session_id]

    @classmethod
    def clear(cls):
        cls._sessions.clear()


class ChatSession:
    def __init__(self):
        self.thread_id = f"gradio-session-{id(self)}"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.trace: List[str] = []
        self.pending_interrupt: Dict[str, Any] = None


# ── Formatting Helpers ──────────────────────────────────────────────

def format_trace(trace: List[str]) -> str:
    if not trace:
        return "Henüz çalışma izi yok."
    lines = []
    for step in trace:
        emoji = {
            "supervisor": "🗺️", "research": "📚", "code": "💻",
            "tool": "🔧", "tools": "⚙️", "interrupt": "⛔", "finish": "✅",
            "think": "💭", "action": "🎯", "observe": "👁️",
        }.get(step.lower(), "•")
        lines.append(f"{emoji} {step}")
    return "\n".join(lines)


def clean_answer(text: str) -> str:
    """Yanıttan JSON ve markdown kalıntılarını temizler."""
    if not text:
        return text
    cleaned = re.sub(r'```json\s*\{[\s\S]*?\}\s*```', '', text).strip()
    cleaned = re.sub(r'\{[\s\S]*?\}\s*$', '', cleaned).strip()
    return cleaned or text


# ── Streaming Graph Runner ──────────────────────────────────────────

def process_message_stream(
    message: str,
    history: List[Dict[str, str]],
    session_id: str
) -> Generator[List[Dict[str, str]], str, str]:
    """Kullanıcı mesajını işler ve graph event'lerini stream eder.
    
    Chatbot titremesini önlemek için:
    - Ara adımlarda SADECE status_text ve trace_display güncellenir.
    - Chatbot'a yalnızca nihai cevap veya interrupt mesajı yazılır.
    """

    # Boş mesaj kontrolü
    if not message or not message.strip():
        yield history, format_trace([]), "Boş mesaj."
        return

    session = SessionManager.get_or_create(session_id)

    # ── Interrupt Onayı ─────────────────────────────────────────────
    if session.pending_interrupt:
        approval = message.strip().lower()
        is_approved = approval in ("evet", "yes", "e", "y", "true", "1", "onayla")
        resume_value = "evet" if is_approved else "hayır"

        current_history = list(history)
        current_history.append({"role": "user", "content": message})
        yield current_history.copy(), format_trace(session.trace), "🔄 Onay işleniyor..."

        try:
            trace = list(session.trace)
            for event in graph.stream(Command(resume=resume_value), session.config):
                for node_name, node_state in event.items():
                    if node_name.startswith("__"):
                        continue

                    trace.append(f"{node_name.upper()}: çalıştı")
                    status = f"🔄 {node_name.upper()} çalışıyor..."
                    yield current_history.copy(), format_trace(trace), status

                    if node_state.get("final_answer"):
                        final = clean_answer(node_state["final_answer"])
                        current_history.append({"role": "assistant", "content": final})
                        session.trace = trace
                        session.pending_interrupt = None
                        yield current_history.copy(), format_trace(trace), "✅ Yanıt hazır."
                        return

            current_history.append({"role": "assistant", "content": "İşlem tamamlandı."})
            session.trace = trace
            session.pending_interrupt = None
            yield current_history.copy(), format_trace(trace), "✅ İşlem tamamlandı."
            return

        except Exception as e:
            error_detail = traceback.format_exc()
            print(f"UI HATASI (interrupt resume): {error_detail}")
            current_history.append({"role": "assistant", "content": f"❌ Hata: {str(e)}"})
            session.pending_interrupt = None
            yield current_history.copy(), format_trace(session.trace), f"❌ Hata: {str(e)}"
            return

    # ── Normal Mesaj İşleme ─────────────────────────────────────────
    state = {
        "messages": [HumanMessage(content=message)],
        "tool_calls": [],
        "current_agent": "supervisor",
        "iteration_count": 0,
        "final_answer": None,
        "pending_tool": None
    }

    current_history = list(history)
    current_history.append({"role": "user", "content": message})
    yield current_history.copy(), format_trace([]), "🔄 Başlatılıyor..."

    try:
        trace = []
        for event in graph.stream(state, session.config):
            # Interrupt kontrolü
            if "__interrupt__" in event:
                interrupt_info = event["__interrupt__"][0]
                value = interrupt_info.value
                trace.append(f"⛔ Interrupt: {value.get('tool_name', 'unknown')}")
                status = f"⛔ ONAY GEREKLİ: {value.get('tool_name', 'unknown')} - 'evet' veya 'hayır' yazın"
                current_history.append({"role": "assistant", "content": status})
                session.trace = trace
                session.pending_interrupt = value
                yield current_history.copy(), format_trace(trace), status
                return

            for node_name, node_state in event.items():
                if node_name.startswith("__"):
                    continue

                trace.append(f"{node_name.upper()}: çalıştı")

                if node_state.get("tool_calls"):
                    for tc in node_state["tool_calls"]:
                        trace.append(f"   → Tool: {tc.get('name', 'unknown')}")

                status = f"🔄 {node_name.upper()} çalışıyor..."
                yield current_history.copy(), format_trace(trace), status

                if node_state.get("final_answer"):
                    final = clean_answer(node_state["final_answer"])
                    current_history.append({"role": "assistant", "content": final})
                    session.trace = trace
                    yield current_history.copy(), format_trace(trace), "✅ Yanıt hazır."
                    return

        current_history.append({"role": "assistant", "content": "Yanıt üretilemedi."})
        session.trace = trace
        yield current_history.copy(), format_trace(trace), "⚠️ Yanıt üretilemedi."

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"UI HATASI: {error_detail}")
        current_history.append({"role": "assistant", "content": f"❌ Hata: {str(e)}"})
        yield current_history.copy(), format_trace([]), f"❌ Hata: {str(e)}"


# ── UI Helpers ──────────────────────────────────────────────────────

def clear_chat():
    SessionManager.clear()
    return [], "", "✅ Yeni oturum başlatıldı."


# ── Anti-flicker CSS ────────────────────────────────────────────────

ANTI_FLICKER_CSS = """
/* Sabit yükseklikler ve taşmayı önle */
.chatbot-container {
    min-height: 500px !important;
    max-height: 500px !important;
    overflow-y: auto !important;
}

/* Gradio Chatbot içindeki mesaj baloncuklarının genişliğini sabitle */
.message-wrap {
    max-width: 90% !important;
    word-break: break-word !important;
}

/* Ana satır yüksekliğini kilitle */
.main-row {
    align-items: flex-start !important;
}

/* Scrollbar genişliğını sabitle, taşmayı engelle */
body {
    overflow-x: hidden !important;
}
"""


# ── Gradio UI ───────────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="🤖 Multi-Agent System", css=ANTI_FLICKER_CSS) as demo:
        gr.Markdown("""
        # 🤖 Multi-Agent System with LangGraph
        
        **Model:** `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` (4-bit Quantization)
        
        **Agents:** Supervisor → Research | Code | Tool
        
        **Özellikler:** ReAct Pattern • Streaming • Human-in-the-loop • Türkçe
        """)
        
        session_state = gr.State(value="default-session")

        with gr.Row(elem_classes="main-row"):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 Konuşma",
                    height=500,
                    elem_classes="chatbot-container",
                    value=None,
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
                    value="Hazır. Mesajınızı yazın.",
                )
            
            with gr.Column(scale=1):
                trace_display = gr.Textbox(
                    label="🔍 Ajan Çalışma İzleme (ReAct)",
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
        
        # Event handlers — STREAMING
        send_btn.click(
            fn=process_message_stream,
            inputs=[msg_input, chatbot, session_state],
            outputs=[chatbot, trace_display, status_text],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )
        
        msg_input.submit(
            fn=process_message_stream,
            inputs=[msg_input, chatbot, session_state],
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
    print("   Features: ReAct Pattern + Streaming")
    print("=" * 60)
    print("\n📥 LLM hazırlanıyor...")
    get_llm_engine()
    print("✅ Model hazır!\n")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7867,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
