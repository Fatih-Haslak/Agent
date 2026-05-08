# Multi-Agent System with LangGraph

Diagramdaki mimariye tam uygun, **LangGraph** tabanlı çoklu agent sistemi.

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

## 🧠 Bellek Katmanı

- **Short-term Memory**: `AgentState["messages"]` içinde tutulur, `add_messages` reducer ile yönetilir.
- **Long-term Memory**: `SQLite` checkpointer ile `data/checkpoints.sqlite` dosyasında kalıcı olarak saklanır.

## 🚀 Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env
# .env dosyasına OPENAI_API_KEY'i ekle
```

## 🖥️ Çalıştırma

```bash
python src/main.py
```

## 📂 Proje Yapısı

```
src/
├── agents/           # Supervisor, Research, Code, Tool agent'ları
├── tools/            # Araçlar ve executor
├── nodes/            # Tools node & Interrupt node
├── memory/           # Short-term & Long-term memory
├── graph/            # LangGraph StateGraph & workflow
├── state/            # AgentState (TypedDict)
├── config.py         # LLM yapılandırması
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
