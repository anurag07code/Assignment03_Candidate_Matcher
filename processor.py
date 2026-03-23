import os
import re
import requests
import urllib3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    Chroma = None  # type: ignore[assignment]

load_dotenv()
urllib3.disable_warnings()


# =========================
# LLM CALL
# =========================
def call_llm(system: str, user: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {
        "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    res = requests.post(url, headers=headers, json=data, verify=False)
    if res.status_code == 200:
        try:
            return res.json()["choices"][0]["message"]["content"]
        except Exception:
            return "LLM error"
    return "LLM error"


# =========================
# INGEST DOCUMENTS
# =========================
def ingest(folder: str):
    if Chroma is None:
        raise ImportError(
            "Chroma is not installed. Install `langchain-chroma` (and its native deps) "
            "or update the project dependencies to enable vector DB ingestion."
        )
    docs = []

    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            pages = loader.load()

            for p in pages:
                p.metadata["source"] = file
            docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectordb = Chroma.from_documents(chunks, embeddings)

    bm25 = BM25Retriever.from_documents(chunks)

    return vectordb, bm25, chunks


# =========================
# METADATA EXTRACTION
# =========================
_TRACKED_SKILLS = [
    "python", "llm", "rag", "langchain", "docker", "aws",
    "java", "javascript", "sql", "react", "machine learning", "ml", "ai", "nlp",
    "tensorflow", "pytorch", "fastapi", "flask", "kubernetes", "gcp", "azure",
    "openai", "gemini", "vector", "embeddings", "api", "git", "linux",
]


def extract_metadata(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    found_skills = [s for s in _TRACKED_SKILLS if s in t]

    # experience: "5+ years", "3 years", "2 years of"
    exp_match = re.search(r"(\d+)\+?\s*years", t)
    experience = int(exp_match.group(1)) if exp_match else 0

    return {"skills": found_skills, "experience": experience}


# =========================
# HYBRID RETRIEVAL
# =========================
def hybrid_search(query: str, vectordb, bm25, k: int = 8, source: str | None = None):
    """Retrieve docs relevant to query. Optionally filter by candidate source."""
    try:
        if source:
            dense = vectordb.similarity_search(query, k=k, filter={"source": source})
            keyword_raw = bm25.invoke(query)
            keyword = [d for d in keyword_raw if d.metadata.get("source") == source][:k]
        else:
            dense = vectordb.similarity_search(query, k=k)
            keyword = bm25.invoke(query)[:k]
    except Exception:
        dense = vectordb.similarity_search(query, k=k)
        keyword = bm25.invoke(query)[:k]
        if source:
            keyword = [d for d in keyword if d.metadata.get("source") == source][:k]
    combined = list({id(doc): doc for doc in dense + keyword}.values())
    return combined


# =========================
# SCORING
# =========================
# Broader keyword variants so JD phrases like "large language models" still match
_SKILL_VARIANTS: Dict[str, List[str]] = {
    "python": ["python"],
    "llm": ["llm", "large language model", "large-language model", "llms"],
    "rag": ["rag", "retrieval augmented", "retrieval-augmented"],
    "langchain": ["langchain", "lang chain"],
    "docker": ["docker", "container"],
    "aws": ["aws", "amazon web"],
    "machine learning": ["machine learning", "ml", "deep learning"],
    "ml": ["machine learning", "ml", "deep learning"],
    "ai": ["ai", "artificial intelligence"],
    "nlp": ["nlp", "natural language"],
    "api": ["api", "apis", "rest"],
}


def score_candidate(text: str, jd: str) -> Tuple[float, Dict[str, Any]]:
    meta = extract_metadata(text)
    jd_lower = (jd or "").strip().lower()
    num_skills = len(meta["skills"])

    # JD alignment: how many of the candidate's skills appear in the JD (with variants)
    matches = 0
    for s in meta["skills"]:
        variants = _SKILL_VARIANTS.get(s, [s])
        if any(v in jd_lower for v in variants):
            matches += 1

    # skill_score combines JD match + raw capability (so candidates with skills never get 0)
    # JD match: 0-1 based on how many of candidate's skills the JD wants
    # capability: 0-1 based on how many relevant skills candidate has (normalized by ~8)
    jd_match = matches / max(num_skills, 1) if num_skills else 0
    capability = min(num_skills / 8.0, 1.0)  # 8+ skills = full capability score
    skill_score = 0.7 * jd_match + 0.3 * capability

    # experience score (0–1, cap at 5 years)
    exp_score = min(meta["experience"] / 5, 1.0)

    # combined score 0–100; ensure at least 1.0 when candidate has any skills or experience
    raw = (0.7 * skill_score + 0.3 * exp_score) * 100
    if num_skills > 0 or meta["experience"] > 0:
        raw = max(raw, 1.0)  # no hard zero for candidates with signal
    final_score = min(raw, 100.0)
    return round(final_score, 2), meta


# =========================
# LLM INSIGHTS
# =========================
def generate_insights(text: str, jd: str) -> str:
    prompt = """
Analyze this candidate vs job description.

Give:
- Strengths
- Gaps
- Final verdict (short)

Keep it concise.
"""

    return call_llm(prompt, f"JD:\n{jd}\n\nResume:\n{(text or '')[:2000]}")


def _stage_from_score(score: float) -> str:
    if score < 25:
        return "exclude"
    if score < 60:
        return "borderline"
    return "strong"


# =========================
# FLASK APP COMPATIBILITY
# (keeps your current routes working)
# =========================
@dataclass
class CandidateMatcherIndex:
    vectordb: Any
    bm25: Any
    chunks: Any
    resumes: List[str]


def build_candidate_matcher_index(resumes_dir: str) -> CandidateMatcherIndex:
    vectordb, bm25, chunks = ingest(resumes_dir)
    resumes = sorted({(d.metadata.get("source") or "unknown") for d in chunks})
    return CandidateMatcherIndex(vectordb=vectordb, bm25=bm25, chunks=chunks, resumes=resumes)


def run_candidate_matching(idx: CandidateMatcherIndex, job_description: str, top_k: int = 10) -> Dict[str, Any]:
    # group chunks by candidate (fallback when hybrid returns nothing)
    candidates: Dict[str, str] = {}
    for doc in idx.chunks:
        src = doc.metadata.get("source") or "unknown"
        candidates.setdefault(src, "")
        candidates[src] += (doc.page_content or "") + "\n"

    jd_skills = extract_metadata(job_description)["skills"]
    items: List[Dict[str, Any]] = []

    for candidate_id, text in candidates.items():
        # hybrid search: get relevant chunks for this candidate vs JD
        relevant_docs = hybrid_search(job_description, idx.vectordb, idx.bm25, k=8, source=candidate_id)
        context = " ".join([d.page_content for d in relevant_docs]).strip() if relevant_docs else text

        score, meta = score_candidate(context, job_description)

        if len(meta["skills"]) == 0 and meta["experience"] == 0:
            continue

        gaps = [s for s in jd_skills if s not in meta["skills"]]
        insights = generate_insights(context, job_description)

        items.append(
            {
                "candidate_id": candidate_id,
                "final_fit_score": float(score),
                "stage": _stage_from_score(float(score)),
                "low_signal": False,
                "confidence": "medium",
                "strengths": meta.get("skills", []),
                "gaps": gaps,
                "explanation": insights,
                "details": {
                    "skills": meta.get("skills", []),
                    "experience": meta.get("experience", 0),
                    "gaps": gaps,
                },
            }
        )

    items.sort(key=lambda x: float(x.get("final_fit_score", 0.0)), reverse=True)

    limit = max(1, int(top_k) if top_k is not None else 10)
    results = []
    for i, item in enumerate(items[:limit], start=1):
        out = dict(item)
        out["rank"] = i
        results.append(out)

    # Build LLM-style narrative for display
    narrative_lines = [
        "## Candidate matching report\n",
        "Here are the ranked candidates based on the job description:\n",
    ]
    for r in results:
        fit = r.get("final_fit_score", 0)
        name = r.get("candidate_id", "Unknown")
        strengths = ", ".join(r.get("strengths") or []) or "-"
        gaps = ", ".join(r.get("gaps") or []) or "-"
        expl = (r.get("explanation") or "").strip()
        narrative_lines.append(f"### {r['rank']}. **{name}** — {fit:.1f}/100 fit\n")
        narrative_lines.append(f"**Strengths:** {strengths}\n")
        narrative_lines.append(f"**Gaps:** {gaps}\n")
        if expl:
            narrative_lines.append(f"\n{expl}\n")
        narrative_lines.append("\n---\n\n")

    return {"results": results, "narrative": "".join(narrative_lines)}

