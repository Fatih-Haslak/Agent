"""
Streamlit UI for Multi-Agent System with Streaming
===================================================
Replaces the Gradio UI with a stable Streamlit interface.
No flickering, no ASGI errors, no dependency hell.
"""

import os
import sys
import re
import traceback

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.graph import graph
from src.config import get_llm_engine


# =============================================================================
# 1. SAYFA YAPILANDIRMASI
# =============================================================================

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 2. MODEL YUKLEME (Cache ile)
# =============================================================================

@st.cache_resource(show_spinner="📥 LLM yükleniyor...")
def _load_llm():
    return get_llm_engine()


# =============================================================================
# 3. OTURUM DURUMU
# =============================================================================

def _init_session():
    """Streamlit session state'ini baslat."""
    defaults = {
        "messages": [],               # Chat gecmisi: [{"role": "user"/"assistant", "content": str}]
        "trace": [],                  # ReAct calisma adimlari: [str, str, ...]
        "thread_id": None,            # LangGraph checkpointer thread ID
        "pending_interrupt": None,    # Onay bekleyen tool bilgisi
        "processing": False,          # Graph su an calisiyor mu?
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# =============================================================================
# 4. YARDIMCI FONKSIYONLAR
# =============================================================================

def _clean_answer(text: str) -> str:
    """LLM yanitindan JSON tool call kalintilarini temizler."""
    if not text:
        return text
    cleaned = re.sub(r'```json\s*\{[\s\S]*?\}\s*```', '', text).strip()
    cleaned = re.sub(r'\{[\s\S]*?\}\s*$', '', cleaned).strip()
    return cleaned or text


def _format_trace(trace: list) -> str:
    """Trace adimlarini emoji'li metne cevirir."""
    emoji_map = {
        "supervisor": "🗺️", "research": "📚", "code": "💻",
        "tool": "🔧", "tools": "⚙️", "interrupt": "⛔", "finish": "✅",
    }
    lines = []
    for step in trace:
        emoji = emoji_map.get(step.lower().split(":")[0].strip(), "•")
        if step.startswith("   "):
            lines.append(f"  {emoji} {step.strip()}")
        else:
            lines.append(f"{emoji} {step}")
    return "\n".join(lines)


# =============================================================================
# 5. GRAPH CALISTIRMA (Streaming + Interrupt)
# =============================================================================

def _run_graph_with_progress(status_box):
    """LangGraph'i calistirir, status kutusunu canli gunceller.
    
    Args:
        status_box: st.status() nesnesi — write() ile icerigi guncellenir.
    
    Returns:
        str | None: final_answer varsa metin, yoksa None
    """
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    trace = list(st.session_state.trace)

    try:
        for event in graph.stream(st.session_state._graph_state, config):
            # Interrupt kontrolu
            if "__interrupt__" in event:
                value = event["__interrupt__"][0].value
                trace.append(f"⛔ Interrupt: {value.get('tool_name', '?')}")
                st.session_state.trace = trace
                st.session_state.pending_interrupt = value
                st.session_state.processing = False
                status_box.update(
                    label=f"⛔ Onay gerekli: {value.get('tool_name', '?')}",
                    state="error",
                )
                return None

            for node_name, node_state in event.items():
                if node_name.startswith("__"):
                    continue

                trace.append(f"{node_name.upper()}: çalıştı")

                if node_state.get("tool_calls"):
                    for tc in node_state["tool_calls"]:
                        trace.append(f"   → Tool: {tc.get('name', '?')}")

                # Status kutusunu canli guncelle
                status_box.write(_format_trace(trace[-4:]))

                if node_state.get("final_answer"):
                    final = _clean_answer(node_state["final_answer"])
                    st.session_state.trace = trace
                    st.session_state.messages.append({"role": "assistant", "content": final})
                    st.session_state.processing = False
                    status_box.update(label="✅ Yanıt hazır", state="complete")
                    return final

        # Eger final_answer bulunamadiysa
        st.session_state.trace = trace
        st.session_state.processing = False
        status_box.update(label="⚠️ Yanıt alınamadı", state="error")
        return None

    except Exception as e:
        st.session_state.processing = False
        status_box.update(label=f"❌ Hata: {str(e)}", state="error")
        tb = traceback.format_exc()
        st.error(tb)
        return None


# =============================================================================
# 6. MESAJ ISLEME
# =============================================================================

def _process_message(prompt: str):
    """Yeni kullanici mesajini isler ve graph'i baslatir."""
    # Kullanici mesajini chat'e ekle
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Thread ID olustur (ilk mesajda)
    if st.session_state.thread_id is None:
        st.session_state.thread_id = f"streamlit-{os.urandom(4).hex()}"

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Graph state
    st.session_state._graph_state = {
        "messages": [HumanMessage(content=prompt)],
        "tool_calls": [],
        "current_agent": "supervisor",
        "iteration_count": 0,
        "final_answer": None,
        "pending_tool": None,
    }

    st.session_state.processing = True
    st.session_state.trace = []

    # Status kutusu ile streaming
    with st.status("🔄 Ajanlar çalışıyor...", expanded=True) as status:
        _run_graph_with_progress(status)


def _resume_after_interrupt(approval: str):
    """Interrupt sonrasi graph'i resume eder."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state._graph_state = Command(resume=approval)
    st.session_state.processing = True
    st.session_state.pending_interrupt = None

    with st.status("🔄 Onay işleniyor...", expanded=True) as status:
        _run_graph_with_progress(status)


# =============================================================================
# 7. UI LAYOUT
# =============================================================================

# Baslik
st.markdown("""
# 🤖 Multi-Agent System with LangGraph
**Model:** `ytu-ce-cosmos/Turkish-Gemma-9b` (4-bit) | **Agents:** Supervisor → Research | Code | Tool
""")

# Iki sutun layout
left_col, right_col = st.columns([3, 1], gap="medium")

# ---- SOL SUTUN: SOHBET ----
with left_col:
    # Mevcut mesajlari goster
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- INTERRUPT ONAY BOLUMU ----
    if st.session_state.pending_interrupt and not st.session_state.processing:
        pending = st.session_state.pending_interrupt
        tool_name = pending.get("tool_name", "?")
        args = pending.get("args", pending.get("arguments", {}))

        st.warning(f"⚠️ **{tool_name}** aracı için onay gerekiyor!")
        if args:
            with st.expander("🔍 Argümanları göster"):
                st.code(str(args), language="text")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Onayla", use_container_width=True, type="primary"):
                _resume_after_interrupt("evet")
                st.rerun()
        with col_b:
            if st.button("❌ Reddet", use_container_width=True):
                _resume_after_interrupt("hayır")
                st.rerun()

        # Chat input gosterme (interrupt cozulene kadar beklet)
        st.stop()

    # ---- CHAT INPUT ----
    prompt = st.chat_input(
        "Bir şeyler yazın...",
        disabled=st.session_state.processing,
    )

    if prompt and not st.session_state.processing:
        _process_message(prompt)
        st.rerun()

# ---- SAG SUTUN: TRACE + KONTROLLER ----
with right_col:
    st.subheader("🔍 ReAct İzleme")

    # Trace gostermek icin bos bir kap
    trace_placeholder = st.empty()

    if st.session_state.trace:
        trace_placeholder.code(
            _format_trace(st.session_state.trace),
            language="text",
            line_numbers=True,
        )
    else:
        trace_placeholder.markdown("*Henüz çalışma izi yok.*")

    st.divider()

    if st.button("🗑️ Yeni Oturum", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # Kisayollar
    with st.expander("📋 Kısayollar", expanded=False):
        st.markdown("""
        - **Web arama:** "Atatürk kimdir?"
        - **Matematik:** "15 çarpı 23 kaç eder?"
        - **Kod:** "Python'da 1'den 10'a kadar sayıları yazdır"
        - **Dosya:** "test.txt dosyasına merhaba yaz"
        
        Kritik tool'lar öncesinde onay istenir.
        """)
