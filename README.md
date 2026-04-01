# LocalAI System

A modular, production-ready local AI system powered by Ollama (Mistral / LLaMA 3).
Handles multiple task types through a unified FastAPI interface.

---

## Architecture Overview

```
Client -> FastAPI -> Task Router -> Module Handler -> LLM Service -> Ollama
                                        |
                              Parsers / Analysis / Prompts / Storage
```

See `ARCHITECTURE.md` for the full diagram.

---

## Modules

| Module | Endpoint Prefix | Sub-tasks |
|---|---|---|
| Resume Optimization | `/api/v1/resume` | optimize, analyze, cover_letter, star_rewrite |
| General Assistant | `/api/v1/assistant` | query, explain |
| Job Application | `/api/v1/job` | question, screening |
| Code Generation | `/api/v1/code` | generate, review, explain |
| Generic Dispatcher | `/api/v1/task` | any of the above via task_type |

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- At least one local model pulled

---

## Setup

### 1. Install Ollama and pull a model

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull mistral

# OR use LLaMA 3
ollama pull llama3

# Start the Ollama server (runs on localhost:11434)
ollama serve
```

### 2. Clone and set up the project

```bash
cd ai_system
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure (optional)

Copy the example env file and adjust settings:

```bash
cp .env.example .env
```

Key settings in `.env`:

```env
DEFAULT_MODEL=mistral          # or llama3
OLLAMA_BASE_URL=http://localhost:11434
API_PORT=8000
LOG_LEVEL=INFO
DEBUG=false
```

### 4. Run the server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at: `http://localhost:8000`
Swagger UI: `http://localhost:8000/api/v1/docs`
ReDoc: `http://localhost:8000/api/v1/redoc`

---

## Quick Start Examples

### Health check

```bash
curl http://localhost:8000/health
```

### Ask a technical question

```bash
curl -X POST http://localhost:8000/api/v1/assistant/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I audit failed logins in Windows Server using PowerShell?"}'
```

### Optimize a resume (text input)

```bash
curl -X POST http://localhost:8000/api/v1/resume/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jane Doe\nIT Analyst...",
    "job_description": "We are hiring a Senior IT Specialist..."
  }'
```

### Upload a resume PDF

```bash
curl -X POST http://localhost:8000/api/v1/resume/optimize/upload \
  -F "resume_file=@resume.pdf" \
  -F "job_description=Senior IT Specialist role requiring Azure AD..."
```

### Generate a PowerShell script

```bash
curl -X POST http://localhost:8000/api/v1/code/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Export all disabled AD accounts to CSV",
    "language": "powershell",
    "requirements": ["Include error handling", "Accept output path parameter"]
  }'
```

### Generic task dispatcher

```bash
curl -X POST http://localhost:8000/api/v1/task/ \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "code",
    "sub_task": "generate",
    "payload": {
      "task_description": "Python script to monitor disk usage and alert if above 85%",
      "language": "python"
    }
  }'
```

---

## Project Structure

```
ai_system/
├── main.py                    # FastAPI app + startup wiring
├── config.py                  # All settings (env-configurable)
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md
│
├── api/
│   └── routes/
│       ├── task.py            # Generic /task dispatcher
│       ├── resume.py          # Resume-specific routes + file upload
│       └── other_routes.py    # Assistant, job app, code, history
│
├── core/
│   ├── llm_service.py         # Ollama HTTP wrapper (all modules use this)
│   ├── task_router.py         # Routes task_type -> handler
│   └── output_formatter.py   # Standardized JSON response envelopes
│
├── modules/
│   ├── resume/handler.py      # Resume optimization pipeline
│   ├── assistant/handler.py   # General Q&A and explanations
│   ├── job_application/handler.py
│   └── code_gen/handler.py
│
├── prompts/
│   ├── prompt_manager.py      # Template registry + renderer
│   ├── resume_prompts.py      # Resume-specific prompt templates
│   └── assistant_prompts.py   # Assistant, job app, code templates
│
├── parsers/
│   └── document_parser.py     # PDF, DOCX, text extraction
│
├── analysis/
│   └── engine.py              # TF-IDF keywords + cosine similarity
│
├── storage/
│   └── database.py            # SQLite via SQLAlchemy
│
├── utils/
│   ├── logger.py              # Rotating file + console logger
│   └── helpers.py             # Text cleaning, truncation, etc.
│
└── examples/
    └── example_requests.json  # Sample requests for all endpoints
```

---

## Adding a New Module

The system is designed for easy extension. To add a new module (e.g., "email_writer"):

**1. Create the handler:**

```python
# modules/email_writer/handler.py
class EmailWriterHandler:
    async def handle(self, payload: dict) -> dict:
        # ... your logic
        pass

email_writer_handler = EmailWriterHandler()
```

**2. Add prompt templates:**

```python
# prompts/email_prompts.py
def register_email_prompts():
    prompt_manager.register("email", "compose", "Write an email about {topic}...")
```

**3. Register in main.py startup:**

```python
from modules.email_writer.handler import email_writer_handler
router.register("email", email_writer_handler.handle)
```

**4. Add API routes (optional):**

```python
@email_route.post("/compose")
async def compose_email(request: EmailRequest):
    ...
```

That's it. The task router, output formatter, and storage all work automatically.

---

## Future Integration: Selenium / Playwright

The job application module is designed to feed a browser automation layer.
The `application_id` returned from `/job/screening` can be used to:

```python
# automation/job_submitter.py (future module)
from storage.database import db

app = db.get_job_applications()[-1]
# Use Playwright to navigate to app["url"] and fill fields
# from app["application_answers"]
```

Enable by uncommenting the Playwright/Selenium lines in `requirements.txt`
and implementing `modules/automation/handler.py`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Switching Models

Change `DEFAULT_MODEL` in `.env` or override per-request:

```python
from core.llm_service import LLMService
llm = LLMService(model="llama3")
response = llm.complete("Your prompt here")
```

Available local models: `ollama list`
