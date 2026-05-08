# Multi-Agent System with LangGraph + Turkish-Gemma-9b

Diagramdaki mimariye tam uygun, **LangGraph** tabanlı çoklu agent sistemi. Şimdi **lokal Turkish-Gemma-9b** modeli ile çalışıyor!

## 🏗️ Mimarisi

```
Kullanıcı
    ↓
Supervisor Agent  →  routing kararı (research | code | tool | finish)
    ↓
Research Agent    →  web_search, summarize
Code Agent        →  code_exec, file_io
Tool Agent        →  calculator, http_request
    ↓
Tool Executor     →  kritik tool varsa Interrupt Node
    ↓
Interrupt Node    →  Human-in-the-loop onay (evet/hayır)
    ↓
Yanıt → Kullanıcı
```

## 🤖 Model

| Özellik | Değer |
|---------|-------|
| Model | `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` |
| Kuantizasyon | 4-bit BitsAndBytes (NF4, double_quant) |
| compute_dtype | `torch.float16` |
| Framework | PyTorch + HuggingFace Transformers |

## 🧠 Bellek Katmanı

- **Short-term Memory**: `AgentState["messages"]` içinde tutulur, `add_messages` reducer ile yönetilir.
- **Long-term Memory**: `SQLite` checkpointer ile `data/checkpoints.sqlite` dosyasında kalıcı olarak saklanır.

## 🚀 Kurulum

### Gereksinimler
- Python 3.10+
- CUDA destekli GPU (önerilen)
- 8 GB+ VRAM (4-bit kuantizasyon ile)

### Bağımlılıklar

```bash
# Conda env önerilir (Python 3.11)
conda create -n agent_env python=3.11
conda activate agent_env

pip install -r requirements.txt
```

## 🖥️ Çalıştırma

```bash
python src/main.py
```

İlk çalıştırmada model Hugging Face'den otomatik indirilecektir (~5-6 GB).

## 📂 Proje Yapısı

```
src/
├── agents/           # Supervisor, Research, Code, Tool agent'ları
├── tools/            # Araçlar ve executor
├── nodes/            # Tools node & Interrupt node
├── memory/           # Short-term & Long-term memory
├── graph/            # LangGraph StateGraph & workflow
├── state/            # AgentState (TypedDict)
├── config.py         # LLM yapılandırması (ModelLoader + LLMEngine)
└── main.py           # CLI entry point
```

## ⚠️ Human-in-the-Loop

`code_exec`, `http_request` ve `file_io` (write/delete/append) çağrıları öncesinde sistem durur ve kullanıcıdan onay ister.

## 🧪 Testler

```bash
pytest tests/
```

## GitHub

Remote: `git@github.com:Fatih-Haslak/Agent.git`
