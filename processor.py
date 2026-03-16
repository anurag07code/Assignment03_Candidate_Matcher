import os
import json
import requests
import shutil
import urllib3
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = "chroma_db"

# Global keyword retriever built at ingestion time for hybrid retrieval
bm25_retriever: BM25Retriever | None = None


def call_llm(system_prompt: str, user_content: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    }
    
    try:
        # Disable warnings properly for both old and new urllib3 versions
        urllib3.disable_warnings()
        
        # verify=False bypasses the SSL check
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            verify=False,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error calling LLM: {str(e)}"




def ingest_documents(directory_path: str):
    all_documents = []
    
    # Process all files in the uploads folder together
    for filename in os.listdir(directory_path):
        path = os.path.join(directory_path, filename)

        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".csv"):
            loader = CSVLoader(path)
        elif path.endswith(".json"):
            from langchain_community.document_loaders import JSONLoader
            loader = JSONLoader(path)
        elif path.endswith(".txt"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(path)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(path)
        elif path.endswith(".docx"):
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(path)
        elif path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
        else:
            continue
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename
        all_documents.extend(docs)

    if not all_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_documents)

    # WINDOWS FIX: Clear the directory safely
    if os.path.exists(CHROMA_PATH):
        # We use a try-except because Chroma might still have a lock
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            print("Database locked. Creating new collection inside existing DB.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # Build a keyword/BM25 retriever for hybrid retrieval
    global bm25_retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)

    return vector_db

def _hybrid_retrieve(
    query: str, vector_db, k_dense: int = 6, k_bm25: int = 6
) -> List[tuple[Document, float]]:
    """
    Hybrid retrieval: dense (Chroma) + keyword (BM25).
    Returns a deduplicated, ranked list of Documents.
    """
    dense_docs_with_scores = vector_db.similarity_search_with_score(
        query, k=k_dense
    )

    keyword_docs: List[Document] = []
    if bm25_retriever is not None:
        # In LangChain with Pydantic v2, BM25Retriever exposes `invoke`
        # (which internally calls `_get_relevant_documents`).
        try:
            keyword_docs = bm25_retriever.invoke(query)[:k_bm25]
        except Exception:
            keyword_docs = []

    combined: Dict[str, Dict[str, Any]] = {}

    def _make_key(doc: Document) -> str:
        src = (
            doc.metadata.get("source_file")
            or doc.metadata.get("source")
            or "unknown"
        )
        return f"{src}|{hash(doc.page_content)}"

    # Add dense docs with score (rank)
    for rank, (doc, _score) in enumerate(dense_docs_with_scores):
        key = _make_key(doc)
        combined[key] = {"doc": doc, "dense_rank": rank, "keyword_rank": None}

    # Add keyword docs
    for rank, doc in enumerate(keyword_docs):
        key = _make_key(doc)
        if key in combined:
            combined[key]["keyword_rank"] = rank
        else:
            combined[key] = {
                "doc": doc,
                "dense_rank": None,
                "keyword_rank": rank,
            }

    # Rank by combined ranks (lower is better)
    def _score(entry: Dict[str, Any]) -> float:
        d = entry["dense_rank"]
        k = entry["keyword_rank"]
        if d is None and k is None:
            return 1e9
        if d is None:
            return k + 3
        if k is None:
            return d
        return (d + k) / 2

    ranked = sorted(combined.values(), key=_score)

    # Normalized score 0–100 based on hybrid rank (position in ranked list)
    n = len(ranked) or 1
    scored: List[tuple[Document, float]] = []
    for idx, e in enumerate(ranked):
        # earlier docs get higher scores
        score_pct = 100.0 * (n - idx) / n
        scored.append((e["doc"], round(score_pct, 1)))
    return scored


def _build_citations(
    docs_with_scores: List[tuple[Document, float]]
) -> List[Dict[str, Any]]:
    """
    Build lightweight citation objects with normalized hybrid scores and location hints.
    """
    citations: List[Dict[str, Any]] = []
    for idx, (d, score) in enumerate(docs_with_scores):
        src = (
            d.metadata.get("source_file")
            or d.metadata.get("source")
            or "unknown"
        )
        preview = d.page_content[:260].replace("\n", " ").strip()

        # Best-effort location: page for PDFs, row for CSV, etc.
        page = d.metadata.get("page")
        row = d.metadata.get("row")
        location = None
        if page is not None:
            try:
                # Many loaders are 0-based; present as 1-based
                page_num = int(page)
                location = f"page {page_num + 1}"
            except Exception:
                location = f"page {page}"
        elif row is not None:
            location = f"row {row}"

        citations.append(
            {
                "source_file": src,
                "score": score,
                "location": location,
                "preview": preview,
                "chunk_index": idx,
            }
        )
    return citations


def factual_qa_tool(query: str, vector_db):
    docs_with_scores = _hybrid_retrieve(
        query, vector_db, k_dense=6, k_bm25=6
    )[:6]
    docs = [d for d, _ in docs_with_scores]
    context = "\n\n".join(
        [
            f"[{d.metadata.get('source_file')}]\n{d.page_content}"
            for d in docs
        ]
    )
    prompt = (
        "You are a precise research assistant.\n"
        "Answer the user's question strictly based on the provided context.\n"
        "Prefer clear text, tables, and simple diagrams (ASCII) when useful.\n"
        "If information is missing, say so explicitly."
    )
    answer = call_llm(prompt, f"Context:\n{context}\n\nQuestion:\n{query}")
    citations = _build_citations(docs_with_scores)
    overall_score = citations[0]["score"] if citations else 0.0
    return {
        "tool": "factual_qa",
        "overall_confidence": overall_score,
        "answer": answer,
        "citations": citations,
    }


def comparative_tool(query: str, vector_db):
    docs_with_scores = _hybrid_retrieve(
        query, vector_db, k_dense=8, k_bm25=8
    )[:8]
    docs = [d for d, _ in docs_with_scores]
    context = "\n\n".join(
        [
            f"Source: {d.metadata.get('source_file')}\nContent:\n{d.page_content}"
            for d in docs
        ]
    )
    prompt = (
        "You are a comparative analytics assistant.\n"
        "Compare, contrast, and correlate information across the context.\n"
        "Use structured output: bullet lists and, where appropriate, markdown tables.\n"
        "Call out agreements, conflicts, and trends. If data is insufficient, say so."
    )
    answer = call_llm(
        prompt, f"Context:\n{context}\n\nComparative question:\n{query}"
    )
    citations = _build_citations(docs_with_scores)
    overall_score = citations[0]["score"] if citations else 0.0
    return {
        "tool": "comparative",
        "overall_confidence": overall_score,
        "answer": answer,
        "citations": citations,
    }


def summary_tool(query: str, vector_db):
    docs_with_scores = _hybrid_retrieve(
        query or "overall summary", vector_db, k_dense=10, k_bm25=10
    )[:10]
    docs = [d for d, _ in docs_with_scores]
    context = "\n\n".join(
        [
            f"[{d.metadata.get('source_file')}]\n{d.page_content}"
            for d in docs
        ]
    )
    prompt = (
        "You are a summarization assistant.\n"
        "Produce a concise, well-structured summary of the topic or documents.\n"
        "Use sections and bullet points where appropriate. Avoid hallucinating."
    )
    answer = call_llm(
        prompt, f"Context:\n{context}\n\nSummarize for:\n{query or 'overall'}"
    )
    citations = _build_citations(docs_with_scores)
    overall_score = citations[0]["score"] if citations else 0.0
    return {
        "tool": "summary",
        "overall_confidence": overall_score,
        "answer": answer,
        "citations": citations,
    }


def agent_dispatcher(user_query: str, vector_db):
    """
    Lightweight agent dispatcher that routes to summary, comparative, or QA tools.
    Falls back to a clarification prompt if intent is ambiguous.
    """
    intent_prompt = (
        "Classify the user's request.\n"
        "Reply with ONLY one of: SUMMARY, COMPARATIVE, QA.\n"
        "SUMMARY: user asks to summarize or overview.\n"
        "COMPARATIVE: user asks to compare, contrast, correlate.\n"
        "QA: direct factual question."
    )
    intent_raw = call_llm(intent_prompt, user_query)
    intent = intent_raw.strip().upper()

    if "SUMMARY" in intent:
        return summary_tool(user_query, vector_db)
    if "COMPARATIVE" in intent:
        return comparative_tool(user_query, vector_db)
    if "QA" in intent:
        return factual_qa_tool(user_query, vector_db)

    # Clarification agent (fallback)
    clarification_answer = (
        "I’m not fully sure what you need.\n\n"
        "Could you clarify whether you want a **summary**, a **comparison**, "
        "or a **direct factual answer** about your documents?"
    )
    return {
        "tool": "clarification",
        "overall_confidence": "low",
        "answer": clarification_answer,
        "citations": [],
    }