"""
Gradio Web UI for Multi-Agent System with Streaming
====================================================

Bu modul, LangGraph tabanli cok-ajanli sistem icin Gradio web arayuzu saglar.

Ozellikler:
-----------
- Streaming: Kullanici mesajini gonderdiginde graph event'leri anlik olarak
ekrana yansitilir (Supervisor -> Agent -> Tools -> ...)
- ReAct Pattern: Her ajani calisma adimlari (Dusun, Hareket Et, Gozlemle)
sag panelde izlenebilir
- Human-in-the-loop: Kritik tool'lar (code_exec, file_io, http_request)
calistirilmadan once kullanicidan onay ister
- Anti-flicker: Chatbot titremesini onlemek icin ara adimlarda sadece
status ve trace guncellenir, nihai cevap gelene kadar chatbot'a dokunulmaz
- Session Management: Her kullanici oturumu icin ayri LangGraph thread_id
ve interrupt durumu tutulur

Gradio Versiyon Notlari:
------------------------
- Gradio 5.12.0 kullanilmaktadir (daha stabil layout motoru icin 6.x'ten dusuruldu)
- gradio-client'ta bool schema hatasi vardir (get_type() fonksiyonu schema=True/False
  durumunda TypeError verir). Bu yuzden import siralamasinda monkey-patch uygulanir.
- Windows'ta localhost erisilebilirlik kontrolu basarisiz olabildiginden
  networking.url_ok bypass edilir.

Kullanim:
---------
    $env:PYTHONIOENCODING="utf-8"
    conda run -n fatih_ai python src/ui.py

Tarayicide http://127.0.0.1:7867 adresinden erisilir.
"""

import os
import sys
import re
import traceback
from typing import List, Dict, Any, Generator

# =============================================================================
# 1. GRADIO-CLIENT BUG FIX (Import Sirasinda Uygulanir)
# =============================================================================
# NOT: Bu patch'ler gradio_client kutuphanesinin import edilmesinden HEMEN SONRA
# uygulanmalidir. Aksi halde Gradio'nun ASGI uygulamasi (FastAPI/Starlette)
# schema parsing yaparken TypeError verir ve HTTP 500 doner.
#
# Hatanin kaynagi: gradio_client.utils.get_type() fonksiyonunda
#   if "const" in schema:
# satiri, schema=True (bool) oldugunda "argument of type 'bool' is not iterable"
# hatasi verir. Cunku Python'da "const" in True gecersizdir.
#
# Bu hata Gradio 4.x, 5.x ve 6.x versiyonlarinda gradio-client'in
# tum surumlerinde gorulmektedir.
# =============================================================================

import gradio_client.utils as _gcu

# Orijinal fonksiyonlari sakla
_original_get_type = _gcu.get_type
_original_json_schema = _gcu._json_schema_to_python_type


def _patched_get_type(schema):
    """
    gradio_client.utils.get_type() icin guvenlik patch'i.
    
    Gradio'nun API dokumantasyonu (api_info) olustururken component'lerin
    JSON schema'sini Python tipine cevirir. Eger schema bir dict degilse
    (ornegin bool True/False), orijinal fonksiyon coker.
    
    Args:
        schema: JSON schema dict veya diger tipler (bool, int, vb.)
    
    Returns:
        str: Python tipi tanimi ("Any" guvenlik fallback'i)
    """
    # Guvenlik kontrolu: schema dict degilse direkt "Any" don
    # Bu durum genellikle JSON Schema'da "additionalProperties": true/false
    # gibi boolean degerlerde ortaya cikar
    if not isinstance(schema, dict):
        return "Any"
    return _original_get_type(schema)


def _patched_json_schema(schema, defs):
    """
    gradio_client.utils._json_schema_to_python_type() icin guvenlik patch'i.
    
    Recursive olarak JSON schema'yi gezerken alt alanlarda da bool degerlerle
    karsilasilabilir. Bu fonksiyon da koruma saglar.
    
    Args:
        schema: JSON schema dict, bool veya diger
        defs: $defs referanslari
    
    Returns:
        str: Python tipi tanimi
    """
    if schema is True or schema is False or not isinstance(schema, dict):
        return "Any"
    return _original_json_schema(schema, defs)


# Monkey-patch'leri uygula
_gcu.get_type = _patched_get_type
_gcu._json_schema_to_python_type = _patched_json_schema

# =============================================================================
# 2. GRADIO IMPORT & WINDOWS LOCALHOST BYPASS
# =============================================================================
# Gradio'nun launch() metodu, server baslatildiktan sonra localhost'un
# erisilebilir olup olmadigini kontrol eder. Windows'ta bu kontrol
# (ozellikle VPN/proxy durumlarinda) basarisiz olabilir ve
# "ValueError: When localhost is not accessible..." hatasi verir.
# networking.url_ok fonksiyonu her zaman True donerek bu kontrol bypass edilir.
# =============================================================================

import gradio as gr
import gradio.networking as networking

# Windows localhost kontrolunu devre disi birak
networking.url_ok = lambda x: True

# Proje kokunu Python path'ine ekle (src/ altindaki moduller icin)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# LangChain/LangGraph mesaj tipleri ve interrupt komutu
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

# Proje icindeki moduller
from src.graph import graph           # Derlenmis LangGraph StateGraph
from src.config import get_llm_engine  # Lazy-load LLM (Turkish-Gemma-9b)


# =============================================================================
# 3. OTURUM YONETIMI (Session Management)
# =============================================================================
# Her kullanici oturumu icin ayri veri tutulur:
# - thread_id: LangGraph SQLite checkpointer'da bu oturumun mesaj gecmisi
# - trace: Ajan calisma adimlarinin listesi (ReAct izleme icin)
# - pending_interrupt: Human-in-the-loop bekleyen tool bilgisi
#
# SessionManager sinifi, Gradio'nun stateless yapisi uzerine stateful
# oturumlar insa etmek icin bir static dictionary kullanir.
# =============================================================================

class SessionManager:
    """
    Tum kullanici oturumlarini merkezi olarak yoneten sinif.
    
    Gradio her istegi ayri bir HTTP request olarak ele alir; bu yuzden
    oturum verilerini sunucu tarafinda (bellekte) tutmak gerekir.
    
    Attributes:
        _sessions (Dict[str, ChatSession]): session_id -> ChatSession eslemesi
    """
    _sessions: Dict[str, "ChatSession"] = {}

    @classmethod
    def get_or_create(cls, session_id: str) -> "ChatSession":
        """
        Var olan oturumu dondurur veya yeni olusturur.
        
        Args:
            session_id: Gradio State component'inden gelen benzersiz ID
        
        Returns:
            ChatSession: Bu oturuma ait session nesnesi
        """
        if session_id not in cls._sessions:
            cls._sessions[session_id] = ChatSession()
        return cls._sessions[session_id]

    @classmethod
    def clear(cls):
        """Tum oturumlari temizler (Yeni Oturum butonu icin)."""
        cls._sessions.clear()


class ChatSession:
    """
    Tek bir kullanici oturumunun durumunu tutan sinif.
    
    Attributes:
        thread_id (str): LangGraph checkpointer'da kullanilan benzersiz ID
        config (dict): LangGraph stream icin yapilandirma (thread_id icerir)
        trace (List[str]): Ajan calisma adimlarinin kronolojik listesi
        pending_interrupt (Dict[str, Any] | None): Onay bekleyen kritik tool
    """
    def __init__(self):
        # Gradio oturumunun Python nesne ID'sini kullanarak benzersiz thread_id olustur
        self.thread_id = f"gradio-session-{id(self)}"
        # LangGraph'in SQLite checkpointer'i icin yapilandirma
        self.config = {"configurable": {"thread_id": self.thread_id}}
        # ReAdiz izleme adimlari (ornegin: "SUPERVISOR: calisti", "→ Tool: wiki_search")
        self.trace: List[str] = []
        # None = interrupt yok; dict = onay bekleyen tool bilgisi
        self.pending_interrupt: Dict[str, Any] = None


# =============================================================================
# 4. YARDIMCI FONKSIYONLAR (Formatting Helpers)
# =============================================================================


def format_trace(trace: List[str]) -> str:
    """
    Ajan calisma adimlarini emojili, okunabilir metne cevirir.
    
    Sag panelde (trace_display) gosterilen calisma izi, her bir adima
    uygun emoji ekleyerek kullaniciya ajani izleme kolayligi saglar.
    
    Args:
        trace: Ajan adimlarinin string listesi
    
    Returns:
        str: Emoji eklenmis, satir satir ayrilmis calisma izi
    
    Ornek:
        >>> format_trace(["supervisor", "research", "   → Tool: wiki_search"])
        "🗺️ supervisor\n📚 research\n•    → Tool: wiki_search"
    """
    if not trace:
        return "Henüz çalışma izi yok."
    lines = []
    for step in trace:
        # Her node adi icin bir emoji haritasi
        emoji = {
            "supervisor": "🗺️",   # Karar verici, yonlendirici
            "research": "📚",     # Bilgi toplama
            "code": "💻",         # Kod yazma/calistirma
            "tool": "🔧",         # Tool agent
            "tools": "⚙️",        # Tool calistirma node'u
            "interrupt": "⛔",    # Onay bekleme
            "finish": "✅",       # Gorev tamamlandi
            "think": "💭",        # ReAct - Dusun
            "action": "🎯",       # ReAct - Hareket Et
            "observe": "👁️",      # ReAct - Gozlemle
        }.get(step.lower(), "•")  # Bilinmeyen adimlar icin bullet
        lines.append(f"{emoji} {step}")
    return "\n".join(lines)


def clean_answer(text: str) -> str:
    """
    LLM yanıtından JSON tool call kalıntılarını ve markdown bloklarını temizler.
    
    Turkish-Gemma-9b modeli bazen yanit olarak JSON tool cagrilari uretebilir.
    Bu fonksiyon kullaniciya gosterilmeden once bu yapilari temizler.
    
    Temizleme adimlari:
        1. Markdown icindeki JSON bloklarını kaldir: ```json {...}```
        2. Metin sonundaki tek basina JSON objelerini kaldir: {...}
    
    Args:
        text: LLM'den gelen ham yanit metni
    
    Returns:
        str: Temizlenmis, kullanici-okunabilir yanit
    
    Ornek:
        >>> clean_answer("Merhaba! ```json{\\"tool\\": \"wiki_search\\"}```")
        "Merhaba!"
    """
    if not text:
        return text
    # Adim 1: Markdown JSON kod bloklarini kaldir
    cleaned = re.sub(r'```json\s*\{[\s\S]*?\}\s*```', '', text).strip()
    # Adim 2: Metin sonundaki tek basina JSON objesini kaldir
    cleaned = re.sub(r'\{[\s\S]*?\}\s*$', '', cleaned).strip()
    # Eger temizleme sonucu bos kaldıysa orijinali don (guvenlik)
    return cleaned or text


# =============================================================================
# 5. STREAMING GRAPH RUNNER (Ana Mantik)
# =============================================================================
# Bu fonksiyon, Gradio'nun yield mekanizmasiyla calisan kalbidir.
# Kullanici mesaj gonderdiginde:
#   1. LangGraph state'i baslatilir (mesaj, tool_calls, current_agent, ...)
#   2. graph.stream(state, config) ile event'ler akar
#   3. Her event (node calismasi) icin yield edilerek Gradio'ya gonderilir
#   4. Gradio sadece degisen component'leri (status, trace) gunceller
#
# ANTI-FLICKER STRATEJISI:
# ------------------------
# Ara adimlarda (supervisor, research, tools calisirken) chatbot'a
# yer tutucu mesajlar eklenMEZ. Sadece nihai cevap (final_answer) veya
# interrupt durumunda chatbot'a yazilir. Bu sayede Gradio'nun her
# yield'da chatbot'u bastan render etmesi engellenir, titreme onlenir.
# =============================================================================

def process_message_stream(
    message: str,
    history: List[Dict[str, str]],
    session_id: str
) -> Generator[List[Dict[str, str]], str, str]:
    """
    Kullanici mesajini isler ve graph event'lerini Gradio'ya stream eder.
    
    Bu fonksiyon bir Python generator'dur. Gradio click/submit event handler'lari
    generator dondurdugunde, her `yield` ifadesi calistiginda UI anlik olarak
    guncellenir. Bu sayede kullanici "Supvervisor calisiyor...", "Research calisiyor..."
    gibi adimlari gercek zamanli izleyebilir.
    
    Anti-flicker prensibi:
        - Ara adimlarda sadece `status_text` ve `trace_display` guncellenir.
        - `chatbot`a yalnizca nihai cevap veya interrupt mesaji yazilir.
    
    Args:
        message: Kullanicinin girdigi metin
        history: Mevcut chat gecmisi (openai-format: [{"role": "user", "content": "..."}, ...])
        session_id: Gradio State component'inden gelen oturum ID'si
    
    Yields:
        Tuple[List[Dict], str, str]: (updated_chat_history, trace_text, status_text)
    
    Akis:
        1. Interrupt onayi bekleniyor mu? -> Onay isle
        2. Degilse -> Yeni mesaj ile LangGraph'i baslat
        3. graph.stream() event'lerini dinle:
           - Her node calistiginda trace'e ekle, status guncelle
           - __interrupt__ gelirse chatbot'a onay mesaji yaz
           - final_answer gelirse chatbot'a nihai cevap yaz
    """

    # -------------------------------------------------------------------------
    # 5.1 Bos Mesaj Kontrolu
    # -------------------------------------------------------------------------
    if not message or not message.strip():
        # Bos mesaj gonderilirse mevcut history'yi oldugu gibi dondur
        yield history, format_trace([]), "Boş mesaj."
        return

    # Oturumu al veya olustur
    session = SessionManager.get_or_create(session_id)

    # -------------------------------------------------------------------------
    # 5.2 INTERRUPT ONAYI (Human-in-the-loop)
    # -------------------------------------------------------------------------
    # Eger onceki mesajda kritik bir tool (code_exec, file_io, http_request)
    # calistirilmadan interrupt edildiyse, kullanicinin yeni mesaji bir ONAY'dir.
    # Ornegin kullanici "evet" veya "hayir" yazarak interrupt'i resume eder.
    # -------------------------------------------------------------------------
    if session.pending_interrupt:
        # Kullanici onayini normalize et (turkce/ingilizce/evet/hayir)
        approval = message.strip().lower()
        is_approved = approval in ("evet", "yes", "e", "y", "true", "1", "onayla")
        resume_value = "evet" if is_approved else "hayır"

        # Chatbot'a sadece kullanici mesajini ekle (onay mesaji)
        current_history = list(history)
        current_history.append({"role": "user", "content": message})

        # Ilk durum: onay isleniyor
        yield current_history.copy(), format_trace(session.trace), "🔄 Onay işleniyor..."

        try:
            # Mevcut trace'i kopyala (interrupt oncesi adimlari koru)
            trace = list(session.trace)

            # LangGraph interrupt'ini resume et
            # Command(resume=...) ile kullanici onayini graph'a ilet
            for event in graph.stream(Command(resume=resume_value), session.config):
                for node_name, node_state in event.items():
                    if node_name.startswith("__"):
                        continue

                    # Node calisti -> trace'e ekle
                    trace.append(f"{node_name.upper()}: çalıştı")
                    status = f"🔄 {node_name.upper()} çalışıyor..."

                    # Chatbot'a DOKUNMA — sadece status ve trace guncelle
                    yield current_history.copy(), format_trace(trace), status

                    # Node nihai cevap uretti mi?
                    if node_state.get("final_answer"):
                        final = clean_answer(node_state["final_answer"])
                        # Nihai cevabi chatbot'a yaz (anti-flicker: tek seferlik)
                        current_history.append({"role": "assistant", "content": final})
                        session.trace = trace
                        session.pending_interrupt = None
                        yield current_history.copy(), format_trace(trace), "✅ Yanıt hazır."
                        return

            # Eger final_answer yoksa (guvenlik neti)
            current_history.append({"role": "assistant", "content": "İşlem tamamlandı."})
            session.trace = trace
            session.pending_interrupt = None
            yield current_history.copy(), format_trace(trace), "✅ İşlem tamamlandı."
            return

        except Exception as e:
            # Interrupt resume sirasinda hata olursa
            error_detail = traceback.format_exc()
            print(f"UI HATASI (interrupt resume): {error_detail}")
            current_history.append({"role": "assistant", "content": f"❌ Hata: {str(e)}"})
            session.pending_interrupt = None
            yield current_history.copy(), format_trace(session.trace), f"❌ Hata: {str(e)}"
            return

    # -------------------------------------------------------------------------
    # 5.3 NORMAL MESAJ ISLEME (Yeni Graph Akisi)
    # -------------------------------------------------------------------------
    # LangGraph'in bekledigi baslangic state'i olustur.
    # AgentState TypedDict'inin tum alanlari verilmelidir.
    # -------------------------------------------------------------------------
    state = {
        "messages": [HumanMessage(content=message)],  # Kullanici mesaji
        "tool_calls": [],                              # Henuz tool cagrisi yok
        "current_agent": "supervisor",                 # Her zaman supervisor baslar
        "iteration_count": 0,                          # Dongu sayaci
        "final_answer": None,                          # Nihai yanit henuz yok
        "pending_tool": None                           # Bekleyen tool yok
    }

    # Chatbot'a sadece kullanici mesajini ekle
    # NOT: Placeholder assistant mesaji eklenMEZ. Chatbot bos kalir
    # nihai cevap gelene kadar. Bu titremeyi onler.
    current_history = list(history)
    current_history.append({"role": "user", "content": message})

    # Ilk durum: baslatiliyor
    yield current_history.copy(), format_trace([]), "🔄 Başlatılıyor..."

    try:
        trace = []

        # graph.stream: LangGraph StateGraph'i adim adim calistirir.
        # Her event bir dict'tir: {node_name: node_state}
        for event in graph.stream(state, session.config):

            # --- INTERRUPT KONTROLU ------------------------------------------------
            # LangGraph `interrupt()` fonksiyonu cagrildiginda __interrupt__ key'i
            # event'e eklenir. Bu durumda kullanicidan onay almak gerekir.
            # -----------------------------------------------------------------------
            if "__interrupt__" in event:
                interrupt_info = event["__interrupt__"][0]
                value = interrupt_info.value

                # Trace'e interrupt kaydet
                trace.append(f"⛔ Interrupt: {value.get('tool_name', 'unknown')}")

                # Kullaniciya onay mesaji olustur
                status = (
                    f"⛔ ONAY GEREKLİ: {value.get('tool_name', 'unknown')}"
                    f" - 'evet' veya 'hayır' yazın"
                )

                # ONEMLI: Interrupt mesaji chatbot'a yazilir cunku bu bir
                # kullanici etkilesimi gerektiren gercek durumdur (anti-flicker istisnasi)
                current_history.append({"role": "assistant", "content": status})
                session.trace = trace
                session.pending_interrupt = value
                yield current_history.copy(), format_trace(trace), status
                return  # Generator'u sonlandir, kullanici onayi bekle

            # --- NORMAL NODE CALISMASI ---------------------------------------------
            for node_name, node_state in event.items():
                if node_name.startswith("__"):
                    continue  # Dahili LangGraph event'lerini atla

                # Node calisti -> trace'e ekle
                trace.append(f"{node_name.upper()}: çalıştı")

                # Tool cagrilari var mi? (research, code, tool agent'lari)
                if node_state.get("tool_calls"):
                    for tc in node_state["tool_calls"]:
                        trace.append(f"   → Tool: {tc.get('name', 'unknown')}")

                # Durum metnini guncelle
                status = f"🔄 {node_name.upper()} çalışıyor..."

                # Anti-flicker: Chatbot'a DOKUNMA. Sadece status ve trace guncelle.
                # Kullanici chatbot'ta sadece kendi mesajini gorur, arka planda
                # calisiyor... yazisi sadece durum cubugunda gorunur.
                yield current_history.copy(), format_trace(trace), status

                # --- NIHAI CEVAP KONTROLU -----------------------------------------
                # Supervisor veya bir agent final_answer alanini set ettiyse
                # graph artik bitti demektir.
                # -------------------------------------------------------------------
                if node_state.get("final_answer"):
                    final = clean_answer(node_state["final_answer"])

                    # Nihai cevabi chatbot'a yaz (tek seferlik guncelleme)
                    current_history.append({"role": "assistant", "content": final})
                    session.trace = trace

                    yield current_history.copy(), format_trace(trace), "✅ Yanıt hazır."
                    return  # Gorev tamamlandi

        # --- GUVENLIK NETI -------------------------------------------------------
        # Eger graph.stream bitti ama hicbir node final_answer set etmediyse
        # (ornegin beklenmedik bir durumda dongu kirildiysa)
        # -------------------------------------------------------------------------
        current_history.append({"role": "assistant", "content": "Yanıt üretilemedi."})
        session.trace = trace
        yield current_history.copy(), format_trace(trace), "⚠️ Yanıt üretilemedi."

    except Exception as e:
        # Beklenmeyen hata durumu
        error_detail = traceback.format_exc()
        print(f"UI HATASI: {error_detail}")
        current_history.append({"role": "assistant", "content": f"❌ Hata: {str(e)}"})
        yield current_history.copy(), format_trace([]), f"❌ Hata: {str(e)}"


# =============================================================================
# 6. UI HELPERS
# =============================================================================


def clear_chat():
    """
    Tum oturumlari temizler ve UI'yi sifirlar.
    
    Returns:
        tuple: (bos_chat_history, bos_trace, baslangic_durum_mesaji)
    """
    SessionManager.clear()
    return [], "", "✅ Yeni oturum başlatıldı."


# =============================================================================
# 7. ANTI-FLICKER CSS (Gradio 5.x Layout Stabilizasyonu)
# =============================================================================
# Gradio'nun yerlesik CSS'sini override ederek container yuksekliklerini
# sabitler. Bu sayede her streaming adiminda layout'un yeniden hesaplanmasi
# engellenir, titreme onlenir.
#
# !important kullanimi: Gradio'nun kendi stillerini ezmek icin gereklidir.
# =============================================================================

ANTI_FLICKER_CSS = """
/* ================================================================
   ANTI-FLICKER CSS: Sayfa titremesini onlemek icin agresif onlemler.
   
   Sorun: F11 (tam ekran)'da duzeliyor, normal modda titriyor.
   Nedeni: Scrollbar'in gorunup kaybolmasi (~17px genislik degisimi)
   layout shift'e yol aciyor.
   
   Cozum:
     1. Scrollbar her zaman gorunur olsun (overflow-y: scroll)
     2. Scrollbar genisligi sabitlensin (scrollbar-gutter: stable)
     3. Sayfa scrollbar'i her zaman acik olsun
     4. width: 100vw scrollbar genisligini hesaba katar
   ================================================================ */

/* TUM SAYFA: Scrollbar her zaman acik olsun, yatay scroll yasak */
html {
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    width: 100vw !important;
}
body {
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    width: 100vw !important;
}

/* Scrollbar goruntuleme alanini sabitle (Chrome/Firefox modern) */
body, html {
    scrollbar-gutter: stable !important;
}

/* Chatbot container: scrollbar her zaman acik, yukseklik sabit */
.chatbot-container {
    min-height: 500px !important;
    max-height: 500px !important;
    overflow-y: scroll !important;
    scrollbar-gutter: stable !important;
}

/* Mesaj baloncuklarinin genisligini sabitle. */
.message-wrap {
    max-width: 90% !important;
    word-break: break-word !important;
}

/* Ana Row hizalamasi sabit */
.main-row {
    align-items: flex-start !important;
}
"""


# =============================================================================
# 8. GRADIO UI TANIMLAMASI
# =============================================================================
# Layout yapisi:
#   +----------------------------------------------------------+
#   |  [Markdown: Baslik]                                      |
#   +----------------------------------------------------------+
#   |  +------------------------+  +-------------------------+ |
#   |  | Chatbot                |  | Trace Display           | |
#   |  | (mesajlar)             |  | (ajani izleme)          | |
#   |  +------------------------+  |                         | |
#   |  | [Input] [Gonder]       |  | [Yeni Oturum]           | |
#   |  +------------------------+  | [Kisayollar]            | |
#   |  | Status Text            |  | [Interrupt aciklamasi]  | |
#   |  +------------------------+  +-------------------------+ |
#   +----------------------------------------------------------+
#
# Component'ler:
#   - chatbot: gr.Chatbot(type="messages") -> OpenAI formatinda mesajlar
#   - msg_input: gr.Textbox -> Kullanici girdisi
#   - send_btn: gr.Button -> Gonder butonu
#   - status_text: gr.Textbox -> Anlik durum mesaji (streaming icin)
#   - trace_display: gr.Textbox -> ReAct calisma izi
#   - clear_btn: gr.Button -> Oturumu sifirla
#   - session_state: gr.State -> Kullanici oturum ID'si (gizli)
# =============================================================================

def create_ui():
    """
    Gradio arayuzunu olusturur ve event handler'lari baglar.
    
    Returns:
        gr.Blocks: Gradio demo nesnesi (launch edilmis degil)
    """
    with gr.Blocks(title="🤖 Multi-Agent System", css=ANTI_FLICKER_CSS) as demo:
        # --- Baslik Bolumu -----------------------------------------------------
        gr.Markdown("""
        # 🤖 Multi-Agent System with LangGraph
        
        **Model:** `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` (4-bit Quantization)
        
        **Agents:** Supervisor → Research | Code | Tool
        
        **Özellikler:** ReAct Pattern • Streaming • Human-in-the-loop • Türkçe
        """)

        # Gizli oturum durumu. Gradio'nun State component'i sayfa yenilense
        # bile degerini sunucu tarafinda tutar.
        session_state = gr.State(value="default-session")

        # --- Ana Layout (2 sutunlu) -------------------------------------------
        with gr.Row(elem_classes="main-row"):
            # SOL SUTUN: Chat alani (scale=3 -> daha genis)
            with gr.Column(scale=3):
                # Chatbot: type="messages" -> OpenAI format ({"role": "user", "content": "..."})
                # height=500 -> sabit yukseklik
                # elem_classes="chatbot-container" -> CSS hedefleme icin
                # value=[] -> bos baslangic degeri
                chatbot = gr.Chatbot(
                    label="💬 Konuşma",
                    height=500,
                    elem_classes="chatbot-container",
                    type="messages",
                    value=[],
                )

                # Input satiri: Textbox + Gonder butonu
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Mesajınız",
                        placeholder=(
                            "Bir şeyler yazın... "
                            "(Interrupt onayı için 'evet' veya 'hayır' yazın)"
                        ),
                        scale=8,  # Textbox genisligi
                    )
                    send_btn = gr.Button(
                        "➤ Gönder",
                        scale=1,
                        variant="primary",  # Vurgulu buton (renkli)
                    )

                # Durum cubugu: streaming adimlarini gosterir
                status_text = gr.Textbox(
                    label="Durum",
                    interactive=False,  # Salt okunur
                    value="Hazır. Mesajınızı yazın.",
                )

            # SAG SUTUN: Calisma izi ve kisayollar (scale=1 -> daha dar)
            with gr.Column(scale=1):
                # ReAct calisma izleme kutusu
                trace_display = gr.Textbox(
                    label="🔍 Ajan Çalışma İzleme (ReAct)",
                    interactive=False,
                    lines=20,  # 20 satirlik yukseklik
                    value="Henüz çalışma izi yok.",
                )

                # Oturumu sifirla butonu
                clear_btn = gr.Button(
                    "🗑️ Yeni Oturum",
                    variant="secondary",  # Daha az vurgulu
                )

                # Sag sutun yardim metni (Markdown)
                gr.Markdown("""
                ### 📋 Kısayollar
                - **Web arama:** "Atatürk kimdir?"
                - **Matematik:** "15 çarpı 23 kaç eder?"
                - **Kod:** "Python'da 1'den 10'a kadar sayıları yazdır"
                - **Dosya:** "test.txt dosyasına merhaba yaz"
                
                ### ⚠️ Interrupt
                Kritik tool'lar öncesinde onay istenir.
                """)

        # ----------------------------------------------------------------------
        # EVENT HANDLERS (Olay Isleyicileri)
        # ----------------------------------------------------------------------
        # Gradio'da bir butona tiklama veya Enter basma olayina fonksiyon baglamak
        # icin .click() veya .submit() kullanilir.
        #
        # NOT: process_message_stream bir generator'dur. Gradio generator'lari
        # otomatik olarak yield eden degerleri arayuze akar. Her yield'da
        # outputs'ta belirtilen component'ler guncellenir.
        # ----------------------------------------------------------------------

        # Send butonuna tiklaninca
        send_btn.click(
            fn=process_message_stream,      # Calistirilacak fonksiyon (generator)
            inputs=[
                msg_input,                  # Fonksiyonun 1. argumani: message
                chatbot,                    # Fonksiyonun 2. argumani: history
                session_state,              # Fonksiyonun 3. argumani: session_id
            ],
            outputs=[
                chatbot,                    # 1. yield: guncellenmis chat history
                trace_display,              # 2. yield: guncellenmis trace metni
                status_text,                # 3. yield: guncellenmis durum metni
            ],
        ).then(
            # Fonksiyon bittikten sonra input kutusunu temizle
            fn=lambda: "",
            outputs=msg_input,
        )

        # Enter tusuna basilinca (Textbox submit event'i)
        msg_input.submit(
            fn=process_message_stream,
            inputs=[msg_input, chatbot, session_state],
            outputs=[chatbot, trace_display, status_text],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        # Yeni Oturum butonuna tiklaninca
        clear_btn.click(
            fn=clear_chat,
            inputs=[],                      # Hicbir input yok
            outputs=[chatbot, trace_display, status_text],
        )

    return demo


# =============================================================================
# 9. ANA PROGRAM (Entry Point)
# =============================================================================

def main():
    """
    Uygulamayi baslatir.
    
    Akis:
        1. LLM'i lazy-load et (ilk cagirimda model yuklenir, ~20-30 sn)
        2. Gradio UI'yi olustur
        3. Web sunucusunu baslat (127.0.0.1:7867)
    """
    print("=" * 60)
    print("🤖 Multi-Agent System Gradio UI")
    print("   Model: ytu-ce-cosmos/Turkish-Gemma-9b-v0.1")
    print("   Features: ReAct Pattern + Streaming")
    print("=" * 60)
    print("\n📥 LLM hazırlanıyor...")

    # LLM'i baslat (ModelLoader.load() ile 4-bit quantization)
    get_llm_engine()

    print("✅ Model hazır!\n")

    # Arayuzu olustur
    demo = create_ui()

    # Web sunucusunu baslat
    # server_name: 127.0.0.1 (yalnizca yerel erisim)
    # server_port: 7867
    # share=False: Public link olusturma (gizlilik icin)
    # show_error=True: Hata detaylarini goster
    demo.launch(
        server_name="127.0.0.1",
        server_port=7867,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    # Script dogrudan calistirildiginda main()'i baslat
    main()
