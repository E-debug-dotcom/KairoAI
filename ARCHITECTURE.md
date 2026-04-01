# Local Multi-Purpose AI System — Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│              (HTTP Client / Streamlit UI / Automation)                  │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ HTTP Requests
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI API LAYER                                │
│                                                                         │
│   POST /task          POST /resume/optimize     POST /assistant/query   │
│   POST /job/apply     POST /code/generate       GET  /history           │
│                                                                         │
│                      [ api/routes/*.py ]                                │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TASK ROUTER                                     │
│                   [ core/task_router.py ]                               │
│                                                                         │
│   task_type: "resume"  ──►  Resume Module                              │
│   task_type: "job"     ──►  Job Application Module                     │
│   task_type: "assist"  ──►  General Assistant Module                   │
│   task_type: "code"    ──►  Code Generation Module                     │
└──────┬──────────┬──────────────┬──────────────┬───────────────────────┘
       │          │              │              │
       ▼          ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  RESUME  │ │   JOB    │ │ASSISTANT │ │   CODE   │
│  MODULE  │ │ APP MOD. │ │  MODULE  │ │  MODULE  │
│          │ │          │ │          │ │          │
│ handler  │ │ handler  │ │ handler  │ │ handler  │
│ analyzer │ │          │ │          │ │          │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │             │            │             │
     └─────────────┴────────────┴─────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐
│   PROMPT    │  │  ANALYSIS ENGINE │  │    PARSERS      │
│  MANAGER   │  │                  │  │                 │
│             │  │ keyword_extract  │  │ pdf_parser      │
│ per-module  │  │ similarity_score │  │ docx_parser     │
│ templates   │  │ missing_keywords │  │ text_handler    │
└──────┬──────┘  └────────┬─────────┘  └────────┬────────┘
       │                  │                      │
       └──────────────────┼──────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CORE LLM SERVICE                                  │
│                    [ core/llm_service.py ]                              │
│                                                                         │
│   • Manages Ollama HTTP calls                                           │
│   • Configurable model (mistral / llama3 / etc.)                       │
│   • Streaming support                                                   │
│   • Retry logic + error handling                                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ HTTP (localhost:11434)
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OLLAMA RUNTIME                                  │
│                  [ Local LLM — Mistral / LLaMA 3 ]                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    CROSS-CUTTING CONCERNS                               │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │  OUTPUT FORMAT  │  │  STORAGE (SQLite) │  │   LOGGER (utils)      │  │
│  │  structured JSON│  │  requests/results │  │   per-component logs  │  │
│  │  raw + clean    │  │  job apps history │  │   rotating file log   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

Data Flow (Resume Optimize example):
─────────────────────────────────────
Client ──► POST /resume/optimize (PDF + JD text)
        ──► Parser extracts resume text
        ──► Analysis Engine scores keywords
        ──► Prompt Manager builds prompt
        ──► LLM Service calls Ollama
        ──► Output Formatter structures result
        ──► Storage saves request + result
        ──► JSON response returned to client
```
