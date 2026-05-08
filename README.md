# Multi-Agent System with LangGraph + Turkish-Gemma-9b

Endüstriyel seviyede çoklu agent sistemi. **Local LLM** ile çalışır, **human-in-the-loop** onay mekanizmalı, **Türkçe** doğal dil destekli.

---

## 🏗️ Mimari Akış Diyagramı

```mermaid
flowchart TD
    User([👤 Kullanıcı]) -->|Mesaj| Start
    
    subgraph LangGraph["🔄 LangGraph StateGraph"]
        Start([START]) --> Supervisor
        
        subgraph SupervisorNode["🗺️ Supervisor Agent"]
            Supervisor["Supervisor<br/>Görev analizi + Routing"]
            Supervisor -->|"finish<br/>(selamlama/sohbet)"| EndNode
            Supervisor -->|"research<br/>(bilgi sorusu)"| Research
            Supervisor -->|"code<br/>(kod/dosya)"| Code
            Supervisor -->|"tool<br/>(matematik/API)"| Tool
            Supervisor -->|"ToolMessage<br/>(sonuç var)"| DirectAnswer["📝 Doğrudan Cevap Üret"]
            DirectAnswer --> EndNode
        end
        
        subgraph ResearchNode["📚 Research Agent"]
            Research["Research Agent<br/>Prompt-based JSON tool calling"]
            Research -->|"tool çağrısı var"| ToolExec
            Research -->|"doğrudan yanıt"| Supervisor
        end
        
        subgraph CodeNode["💻 Code Agent"]
            Code["Code Agent<br/>Prompt-based JSON tool calling"]
            Code -->|"tool çağrısı var"| ToolExec
            Code -->|"doğrudan yanıt"| Supervisor
        end
        
        subgraph ToolNode["🔧 Tool Agent"]
            Tool["Tool Agent<br/>Prompt-based JSON tool calling"]
            Tool -->|"tool çağrısı var"| ToolExec
            Tool -->|"doğrudan yanıt"| Supervisor
        end
        
        subgraph ToolExecution["⚙️ Tool Execution Layer"]
            ToolExec["Tool Executor"]
            ToolExec -->|"Kritik tool?"| InterruptCheck{"Kritik mi?"}
            ToolExec -->|"Güvenli tool"| Supervisor
            
            InterruptCheck -->|"Evet<br/>(code_exec, file_io yazma, http_request)"| Interrupt["⛔ Interrupt Node<br/>Human-in-the-Loop"]
            InterruptCheck -->|"Hayır"| Supervisor
            
            Interrupt -->|"evet"| ToolRun["🔨 Tool Çalıştır"]
            Interrupt -->|"hayır"| Supervisor
            ToolRun --> Supervisor
        end
    end
    
    EndNode([END]) -->|Yanıt| User
    
    style Supervisor fill:#6c5ce7,color:#fff
    style Research fill:#00b894,color:#fff
    style Code fill:#0984e3,color:#fff
    style Tool fill:#e17055,color:#fff
    style ToolExec fill:#fdcb6e,color:#000
    style Interrupt fill:#d63031,color:#fff
    style DirectAnswer fill:#00cec9,color:#000
    style EndNode fill:#6c5ce7,color:#fff
```

---

## 🛠️ Araç Katmanı (Tools)

```mermaid
flowchart LR
    subgraph Tools["🧰 Kullanılabilir Araçlar"]
        direction TB
        
        subgraph ResearchTools["📚 Research Tools"]
            Wiki["wiki_search<br/>Türkçe Wikipedia API<br/>GÜVENİLİR + DETAYLI"]
            Web["web_search<br/>DuckDuckGo/ddgs<br/>Genel web arama"]
            Sum["summarize<br/>Metin özetleme"]
        end
        
        subgraph CodeTools["💻 Code Tools"]
            CodeExec["code_exec<br/>Python/Bash/JS çalıştır<br/>⚠️ Kritik"]
            FileIO["file_io<br/>Dosya okuma/yazma/silme<br/>⚠️ Yazma/Silme Kritik"]
        end
        
        subgraph UtilityTools["🔧 Utility Tools"]
            Calc["calculator<br/>Matematiksel hesaplama"]
            HTTP["http_request<br/>HTTP GET/POST/PUT/DELETE<br/>⚠️ Kritik"]
        end
    end
    
    style Wiki fill:#00b894,color:#fff
    style Web fill:#74b9ff,color:#000
    style Sum fill:#a29bfe,color:#000
    style CodeExec fill:#e17055,color:#fff
    style FileIO fill:#e17055,color:#fff
    style Calc fill:#fdcb6e,color:#000
    style HTTP fill:#e17055,color:#fff
```

---

## 🧠 Bellek Katmanı

```mermaid
flowchart TB
    subgraph Memory["🧠 Bellek Sistemi"]
        subgraph STM["Short-term Memory"]
            Messages["messages[]<br/>LangGraph add_messages reducer<br/>Konuşma geçmişi (son 6-10 mesaj)"]
        end
        
        subgraph LTM["Long-term Memory"]
            SQLite["SQLite Checkpointer<br/>data/checkpoints.sqlite<br/>Thread-based kalıcı depolama"]
        end
    end
    
    AgentState["AgentState<br/>(TypedDict)"] --> Messages
    AgentState --> SQLite
    
    style STM fill:#74b9ff,color:#000
    style LTM fill:#a29bfe,color:#000
```

---

## 🤖 Model

| Özellik | Değer |
|---------|-------|
| Model | `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` |
| Kuantizasyon | 4-bit BitsAndBytes (NF4, double_quant) |
| compute_dtype | `torch.float16` |
| Framework | PyTorch + HuggingFace Transformers |
| VRAM Kullanımı | ~8 GB |

---

## 🚀 Kurulum

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
# Tarayıcıda: http://localhost:7860
```

---

## 📂 Proje Yapısı

```
src/
├── agents/
│   ├── supervisor.py       # 🗺️ Görev analizi + routing
│   ├── research.py         # 📚 Wikipedia + web arama
│   ├── code.py             # 💻 Kod yazma/çalıştırma
│   └── tool.py             # 🔧 Matematik + HTTP
├── tools/
│   ├── wiki_search.py      # 📚 Türkçe Wikipedia API
│   ├── web_search.py       # 🌐 DuckDuckGo/ddgs arama
│   ├── code_exec.py        # 💻 Kod çalıştırma
│   ├── file_io.py          # 📁 Dosya işlemleri
│   ├── calculator.py       # 🧮 Matematiksel hesaplama
│   ├── http_request.py     # 🌐 HTTP istekleri
│   └── executor.py         # ⚙️ Tool yürütücü + kritik kontrol
├── nodes/
│   ├── tools_node.py       # Tool çalıştırma
│   └── interrupt_node.py   # ⛔ Human-in-the-loop
├── memory/
│   ├── short_term.py       # In-state messages
│   └── long_term.py        # SQLite checkpointer
├── graph/
│   └── workflow.py         # LangGraph StateGraph
├── state/
│   └── agent_state.py      # TypedDict tanımı
├── config.py               # LLM yapılandırması
├── main.py                 # CLI entry point
└── ui.py                   # Gradio web arayüzü
```

---

## 🧪 Test Sonuçları

| Senaryo | Akış | Sonuç |
|---------|------|-------|
| **Selamlama** | SUPERVISOR → finish | "Merhaba! İyiyim, teşekkür ederim." ✅ |
| **Matematik** | SUPERVISOR → TOOL → TOOLS → SUPERVISOR | "15 × 23 = 345" ✅ |
| **Wikipedia** | SUPERVISOR → RESEARCH → TOOLS → SUPERVISOR | "Sergen Yalçın, 5 Kasım 1972 doğumlu..." ✅ |

---

## ⚠️ Human-in-the-Loop

Kritik tool'lar öncesi onay ister:
- `code_exec` (kod çalıştırma)
- `file_io` yazma/silme (okuma güvenli)
- `http_request` (harici API çağrısı)

---

## GitHub

Remote: `git@github.com:Fatih-Haslak/Agent.git`
