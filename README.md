# Multi-Agent System with LangGraph + Turkish-Gemma-9b

Endüstriyel seviyede çoklu agent sistemi. **Local LLM** ile çalışır, **human-in-the-loop** onay mekanizmalı, **Türkçe** doğal dil destekli.

---

## 🎯 Kim Ne Kararını Veriyor?

| Katman | Kim | Karar |
|--------|-----|-------|
| **1. Routing** | 🗺️ **Supervisor** | "Bu görev hangi agent'a gitsin?" (research / code / tool / finish) |
| **2. Tool Seçimi** | 📚💻🔧 **Alt Agent** | "Bu görevi hangi tool ile yapayım?" (wiki_search / calculator / code_exec ...) |
| **3. Tool Çalıştırma** | ⚙️ **Tool Executor** | "Bu tool kritik mi? → Onay gerekli mi?" |
| **4. Cevap Üretimi** | 🗺️ **Supervisor** | "Tool sonucu var → Direkt cevap ver" veya "Tekrar routing yap" |

---

## 🏗️ DETAYLI MİMARİ AKIŞI

```mermaid
flowchart TD
    subgraph Kullanici["👤 Kullanıcı"]
        UserMsg["Mesaj: 'Atatürk kimdir?'"]
    end
    
    subgraph Seviye1["SEVİYE 1: SUPERVISOR ROUTING"]
        Supervisor["🗺️ SUPERVISOR AGENT<br/>Karar: Hangi agent çalışsın?"]
        
        Supervisor -->|"Bilgi sorusu → research"| Agent1
        Supervisor -->|"Matematik/API → tool"| Agent2
        Supervisor -->|"Kod/Dosya → code"| Agent3
        Supervisor -->|"Sohbet/Selam → finish"| CevapUret
        Supervisor -->|"Tool sonucu var → finish"| CevapUret
    end
    
    subgraph Seviye2["SEVİYE 2: AGENT TOOL SEÇİMİ"]
        Agent1["📚 RESEARCH AGENT<br/>'Bilgi toplamam lazım...<br/>Hangi tool?'"]
        Agent2["🔧 TOOL AGENT<br/>'Hesaplama/API lazım...<br/>Hangi tool?'"]
        Agent3["💻 CODE AGENT<br/>'Kod yazmam lazım...<br/>Hangi tool?'"]
        
        Agent1 -->|"wiki_search(query='Atatürk')"| ToolSecim1["🛠️ Tool: wiki_search"]
        Agent1 -->|"web_search(query='Atatürk')"| ToolSecim2["🛠️ Tool: web_search"]
        
        Agent2 -->|"calculator(expr='15*23')"| ToolSecim3["🛠️ Tool: calculator"]
        Agent2 -->|"http_request(GET, 'api.com')"| ToolSecim4["🛠️ Tool: http_request<br/>⚠️ Kritik!"]
        
        Agent3 -->|"code_exec(python, 'print(1+1)')"| ToolSecim5["🛠️ Tool: code_exec<br/>⚠️ Kritik!"]
        Agent3 -->|"file_io(read, 'test.txt')"| ToolSecim6["🛠️ Tool: file_io"]
        Agent3 -->|"file_io(write, 'test.txt')"| ToolSecim7["🛠️ Tool: file_io<br/>⚠️ Kritik!"]
    end
    
    subgraph Seviye3["SEVİYE 3: TOOL YÜRÜTME"]
        ToolExec["⚙️ TOOL EXECUTOR<br/>Tool'u çalıştır"]
        
        KritikKontrol{"Kritik Tool mu?"}
        
        ToolExec --> KritikKontrol
        
        KritikKontrol -->|"Hayır<br/>(wiki_search, calculator, file_io read)"| SonucNormal["✅ Sonuç hazır"]
        KritikKontrol -->|"Evet<br/>(code_exec, http_request, file_io write)"| Interrupt["⛔ INTERRUPT<br/>Kullanıcı onayı bekle"]
        
        Interrupt -->|"Kullanıcı: evet"| SonucOnay["✅ Tool çalıştır"]
        Interrupt -->|"Kullanıcı: hayır"| SonucIptal["❌ İptal edildi"]
    end
    
    subgraph Seviye4["SEVİYE 4: CEVAP ÜRETİMİ"]
        CevapUret["📝 CEVAP ÜRETİMİ"]
        SonucNormal --> Supervisor
        SonucOnay --> Supervisor
        SonucIptal --> Supervisor
    end
    
    UserMsg --> Supervisor
    CevapUret -->|"Atatürk, Türkiye Cumhuriyeti'nin kurucusudur..."| Kullanici
    
    style Supervisor fill:#6c5ce7,color:#fff,stroke:#fff,stroke-width:3px
    style Agent1 fill:#00b894,color:#fff
    style Agent2 fill:#e17055,color:#fff
    style Agent3 fill:#0984e3,color:#fff
    style Interrupt fill:#d63031,color:#fff,stroke:#fff,stroke-width:2px
    style CevapUret fill:#fdcb6e,color:#000
```

---

## 🔍 KARAR MEKANİZMASI DETAYLI AÇIKLAMA

### 1️⃣ Supervisor Nasıl Routing Kararı Verir?

```
Kullanıcı: "Atatürk kimdir?"
    ↓
Supervisor LLM (Turkish-Gemma-9b)
Prompt: "Bu görev hangi agent'a gitsin?"
    ↓
Çıktı: {"next": "research", "reason": "Bilgi sorusu"}
    ↓
Karar: RESEARCH agent'ına gönder
```

### 2️⃣ Research Agent Nasıl Tool Seçer?

```
Research Agent LLM (Turkish-Gemma-9b)
Prompt: "Sergen Yalçın kimdir? Araştırma yap."
    ↓
Çıktı: {"tool": "wiki_search", "args": {"query": "Sergen Yalçın"}}
    ↓
Karar: wiki_search tool'unu çağır
```

### 3️⃣ Tool Executor Nasıl Çalışır?

```
wiki_search(query="Sergen Yalçın")
    ↓
Kritik mi? → Hayır (güvenli tool)
    ↓
Direkt çalıştır
    ↓
Sonuç: "Sergen Yalçın, 5 Kasım 1972 doğumlu..."
    ↓
Supervisor'a dön
```

### 4️⃣ Supervisor Sonucu Nasıl Değerlendirir?

```
Supervisor: "Tool sonucu var mı?"
    ↓
Evet → "Direkt cevap üret, tekrar agent'a gönderme"
    ↓
LLM: "Sergen Yalçın, Türk teknik direktör..."
    ↓
Kullanıcıya yanıt
```

---

## 🛠️ TOOL KATMANI

| Tool | Kim Seçer | Kritik? | Örnek Kullanım |
|------|-----------|---------|----------------|
| `wiki_search` | 📚 Research Agent | ❌ | `"Atatürk kimdir"` → Wikipedia'dan detaylı bilgi |
| `web_search` | 📚 Research Agent | ❌ | `"Son dakika haberler"` → Web araması |
| `summarize` | 📚 Research Agent | ❌ | Uzun metni özetleme |
| `calculator` | 🔧 Tool Agent | ❌ | `"15 * 23"` → 345 |
| `http_request` | 🔧 Tool Agent | ⚠️ **Evet** | `"GET https://api.com"` → Onay gerektirir |
| `code_exec` | 💻 Code Agent | ⚠️ **Evet** | `"python: print(1+1)"` → Onay gerektirir |
| `file_io` (read) | 💻 Code Agent | ❌ | `"read test.txt"` → Güvenli |
| `file_io` (write) | 💻 Code Agent | ⚠️ **Evet** | `"write test.txt"` → Onay gerektirir |

---

## 🧠 BELLEK KATMANI

```mermaid
flowchart LR
    subgraph STM["Short-term Memory<br/>(RAM - Anlık)"]
        Messages["messages[]<br/>Konuşma geçmişi<br/>Son 6-10 mesaj"]
    end
    
    subgraph LTM["Long-term Memory<br/>(SQLite - Kalıcı)"]
        DB["checkpoints.sqlite<br/>Thread-based depolama<br/>Oturumlar arası hatırlama"]
    end
    
    AgentState["AgentState<br/>TypedDict"] --> Messages
    AgentState --> DB
    
    style STM fill:#74b9ff,color:#000
    style LTM fill:#a29bfe,color:#000
```

---

## 🤖 MODEL

| Özellik | Değer |
|---------|-------|
| Model | `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` |
| Kuantizasyon | 4-bit BitsAndBytes (NF4, double_quant) |
| compute_dtype | `torch.float16` |
| Framework | PyTorch + HuggingFace Transformers |
| VRAM Kullanımı | ~8 GB |

---

## 🚀 KURULUM

### Gereksinimler
- Python 3.10+
- CUDA destekli GPU (önerilen)
- 8 GB+ VRAM

### Adımlar

```bash
# 1. Repoyu klonla
git clone git@github.com:Fatih-Haslak/Agent.git
cd Agent

# 2. Conda env oluştur
conda create -n agent_env python=3.11
conda activate agent_env

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Çalıştır
# Terminal modu:
python src/main.py

# Web UI modu:
python src/ui.py
# Tarayıcıda: http://localhost:7861
```

---

## 📂 PROJE YAPISI

```
src/
├── agents/
│   ├── supervisor.py       # 🗺️ ROUTING: Hangi agent'a gitsin?
│   ├── research.py         # 📚 TOOL SEÇİMİ: wiki_search/web_search?
│   ├── code.py             # 💻 TOOL SEÇİMİ: code_exec/file_io?
│   └── tool.py             # 🔧 TOOL SEÇİMİ: calculator/http_request?
├── tools/
│   ├── wiki_search.py      # 📚 Türkçe Wikipedia API
│   ├── web_search.py       # 🌐 DuckDuckGo/ddgs arama
│   ├── code_exec.py        # 💻 Kod çalıştırma
│   ├── file_io.py          # 📁 Dosya işlemleri
│   ├── calculator.py       # 🧮 Matematiksel hesaplama
│   ├── http_request.py     # 🌐 HTTP istekleri
│   └── executor.py         # ⚙️ TOOL YÜRÜTÜCÜ + Kritik kontrol
├── nodes/
│   ├── tools_node.py       # Tool çalıştırma
│   └── interrupt_node.py   # ⛔ Human-in-the-loop onay
├── memory/
│   ├── short_term.py       # In-state messages
│   └── long_term.py        # SQLite checkpointer
├── graph/
│   └── workflow.py         # LangGraph StateGraph
├── state/
│   └── agent_state.py      # TypedDict tanımı
├── config.py               # LLM yapılandırması
├── main.py                 # CLI entry point
└── ui.py                   # Gradio web arayüzü (port 7861)
```

---

## 🧪 DOĞRULANAN TEST SONUÇLARI

| Senaryo | Supervisor Kararı | Agent Tool Seçimi | Sonuç |
|---------|-------------------|-------------------|-------|
| **"Merhaba"** | `finish` (sohbet) | — | "Merhaba! İyiyim..." ✅ |
| **"15×23?"** | `tool` (matematik) | `calculator` | "345" ✅ |
| **"Sergen Yalçın?"** | `research` (bilgi) | `wiki_search` | "5 Kasım 1972 doğumlu..." ✅ |

---

## ⚠️ HUMAN-IN-THE-LOOP

**Kritik tool'lar öncesi onay ister:**
- `code_exec` (kod çalıştırma)
- `file_io` yazma/silme (okuma güvenli)
- `http_request` (harici API çağrısı)

---

## GitHub

Remote: `git@github.com:Fatih-Haslak/Agent.git`
