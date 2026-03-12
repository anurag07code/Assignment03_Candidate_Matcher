# Intelligent Multi-Source Research Assistant

An Advanced RAG system built with Flask and LangChain that synthesizes information from PDFs, CSVs, and Markdown files.

## 🚀 Key Features
- **Multi-Source Ingestion:** Processes structured (CSV) and unstructured (PDF, MD) data.
- **Agentic Routing:** Uses an Agent Dispatcher to classify intent into Summary, Comparative, or Factual QA.
- **Source Attribution:** Every response includes source filenames and a Confidence Score.
- **Local Vector Store:** Utilizes ChromaDB for persistent embedding storage.

## 🛠️ Architecture
1. **Frontend:** Flask-based UI for file staging and querying.
2. **Knowledge Base:** Documents are chunked (1000 tokens, 150 overlap) and embedded using `gemini-embedding-001`.
3. **Dispatcher:** An LLM-based router evaluates the query and triggers one of three specialized tools:
   - `Summary Tool`: Aggregates document overviews.
   - `Comparative Tool`: Correlates metrics across sources.
   - `Factual QA Tool`: Provides specific details with citations.
