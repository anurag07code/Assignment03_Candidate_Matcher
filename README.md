# GenAI Candidate Matcher

A web application that matches candidate resumes against job descriptions using hybrid retrieval (dense embeddings + BM25), skill-based scoring, and LLM-generated insights. Optimized for GenAI Engineer roles.

## Features

- **PDF resume ingestion** — Upload multiple resumes; extracts text via PyPDF
- **Hybrid retrieval** — Chroma (vector DB) + BM25 for semantic and keyword search
- **Skill extraction** — Detects 25+ tracked skills (Python, LLM, RAG, LangChain, Docker, AWS, etc.) and years of experience
- **Fit scoring** — Combines JD alignment and candidate capability into a 0–100 score
- **LLM insights** — OpenRouter-powered analysis (strengths, gaps, verdict) per candidate
- **Narrative output** — Results presented as an LLM-style markdown report

## Prerequisites

- Python 3.10+
- [OpenRouter](https://openrouter.ai) API key (for LLM insights)
- [Google AI](https://ai.google.dev) API key (for Gemini embeddings)

## Installation

```bash
# Clone or navigate to the project
cd "C.1-A03-Candidate Matcher"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
# Required: OpenRouter (for LLM insights)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Required: Google AI (for embeddings)
GOOGLE_API_KEY=your-google-ai-key

# Optional: OpenRouter model (default: openrouter/free)
OPENROUTER_MODEL=openrouter/free
```

## Usage

### Start the server

```bash
python app.py
```

The app runs at **http://127.0.0.1:5000** (Flask debug mode).

### Workflow

1. **Paste job description** — Enter the role requirements in the textarea (e.g., GenAI Engineer with Python, RAG, LangChain).
2. **Upload resumes** — Drop or select PDF files.
3. **Build matcher index** — Ingests resumes, chunks text, builds vector + BM25 indexes.
4. **Run matching** — Scores candidates against the JD and generates LLM insights.

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check; `matcher_ready` indicates index is built |
| `/upload` | POST | Upload PDF resumes (form-data, key: `files`) |
| `/matcher/build` | POST | Build candidate matcher index from uploaded resumes |
| `/matcher/match` | POST | Run matching. Body: `{ "job_description": "...", "top_k": 10 }` |

## Project structure

```
C.1-A03-Candidate Matcher/
├── app.py              # Flask routes (upload, build, match)
├── processor.py        # Ingestion, scoring, LLM calls
├── requirements.txt
├── .env                # API keys (create manually, see Configuration)
├── static/
│   └── style.css       # UI styles
└── templates/
    └── index.html      # Single-page UI
```

## How it works

1. **Ingestion** — PDFs are loaded, split into 800-token chunks with 100 overlap, embedded via Gemini, and stored in Chroma + BM25.
2. **Scoring** — Each resume is scored by:
   - **JD alignment (70%)** — How many of the candidate's skills appear in the job description
   - **Capability (30%)** — How many tracked skills the candidate has
   - **Experience** — Years extracted via regex, normalized (cap at 5 years)
3. **Insights** — LLM analyzes each candidate vs. the JD and returns strengths, gaps, and a short verdict.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Chroma is not installed` | Ensure `langchain-chroma` is installed; on Windows, you may need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| `LLM error` | Check `OPENROUTER_API_KEY` and model availability |
| Embedding errors | Verify `GOOGLE_API_KEY` is set and has Gemini access |
| Fit score 0 for everyone | Ensure the job description mentions relevant skills (e.g., Python, RAG) |

## License

Internal use / assignment project.
