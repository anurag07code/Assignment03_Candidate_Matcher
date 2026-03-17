import os
import json
import requests
import shutil
import urllib3
import re
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    Chroma = None  # type: ignore[assignment]
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = "chroma_db"

def call_llm(system_prompt: str, user_content: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
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
# ============================================================
# Candidate Matcher (GenAI Engineer) — Hybrid retrieval + scoring
# ============================================================

_CURRENT_YEAR = datetime.utcnow().year


def _safe_int(x: Any, default: int | None = None) -> int | None:
    try:
        return int(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _extract_years_experience(text: str) -> int | None:
    # Common patterns: "5+ years", "3 years of experience", "over 7 years"
    patterns = [
        r"(\d{1,2})\s*\+\s*years",
        r"(\d{1,2})\s*years?\s+of\s+experience",
        r"over\s+(\d{1,2})\s*years",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return _safe_int(m.group(1))
    return None


def _extract_latest_year(text: str) -> int | None:
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
    if not years:
        return None
    try:
        return max(int(y) for y in years)
    except Exception:
        return None


def _normalize_skill(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\(\)\[\]\{\},;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["python", "py"],
    "llm": ["llm", "large language model", "large-language model"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "langchain": ["langchain"],
    "vector database": ["vector db", "vector database", "vectordb", "chroma", "faiss", "pinecone", "weaviate", "milvus"],
    "embeddings": ["embeddings", "embedding"],
    "prompt engineering": ["prompt engineering", "prompting"],
    "evaluation": ["eval", "evaluation", "llm evals", "ragas"],
    "mlops": ["mlops", "model monitoring", "monitoring", "deployment"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "s3", "lambda", "ecs", "eks", "bedrock"],
    "gcp": ["gcp", "vertex ai", "vertex", "cloud run"],
    "azure": ["azure", "azure openai"],
    "openai": ["openai", "gpt-4", "gpt-4o", "gpt"],
    "gemini": ["gemini", "google genai"],
    "huggingface": ["hugging face", "huggingface", "transformers"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "sql": ["sql", "postgres", "postgresql", "mysql"],
    "etl": ["etl", "pipelines", "airflow", "dagster"],
    "security": ["pii", "security", "gdpr", "soc2", "redaction"],
}


def _skill_presence(text: str, canonical: str) -> bool:
    needles = _SKILL_ALIASES.get(canonical, [canonical])
    t = text.lower()
    return any(n.lower() in t for n in needles)


def _extract_skills_rule_based(text: str) -> List[str]:
    found = []
    for canonical in _SKILL_ALIASES.keys():
        if _skill_presence(text, canonical):
            found.append(canonical)
    return sorted(set(found))


def _default_genai_engineer_jd() -> str:
    return (
        "Job Title: GenAI Engineer\n\n"
        "We are looking for a GenAI Engineer to design and deploy LLM-powered products. "
        "You will build RAG pipelines, evaluation harnesses, and production APIs.\n\n"
        "Must-Have:\n"
        "- Python, strong software engineering fundamentals\n"
        "- LLM/RAG: embeddings, vector databases (Chroma/FAISS), retrieval + reranking\n"
        "- Frameworks: LangChain (or equivalent), tool/function calling\n"
        "- API development: FastAPI/Flask, testing, structured outputs (JSON)\n\n"
        "Important:\n"
        "- LLM evaluation, prompt engineering, grounding/attribution\n"
        "- Data processing, chunking, metadata extraction, hybrid search (BM25 + dense)\n"
        "- Cloud & deployment: Docker, CI/CD, AWS/GCP/Azure\n\n"
        "Nice-to-Have:\n"
        "- Fine-tuning, open-source models (HF), GPUs\n"
        "- MLOps, monitoring, security/redaction, compliance\n\n"
        "Implicit signals:\n"
        "- Recent GenAI work (last 12–18 months), shipped projects, measurable impact."
    )


def _parse_jd_into_clusters(job_description: str) -> Dict[str, Any]:
    """
    Parse a job description into weighted skill clusters.
    Uses an LLM if available; falls back to heuristics.
    Output format:
      {
        "clusters": [
           {"name": "must_have", "weight": 0.45, "skills": [...]},
           {"name": "important", "weight": 0.30, "skills": [...]},
           {"name": "nice_to_have", "weight": 0.15, "skills": [...]},
           {"name": "implicit", "weight": 0.10, "skills": [...]}
        ]
      }
    """
    jd = (job_description or "").strip()
    if not jd:
        jd = _default_genai_engineer_jd()

    # Try LLM for structured cluster extraction (best effort).
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        system = (
            "You are an expert technical recruiter for GenAI Engineering roles.\n"
            "Extract skill clusters from the job description.\n"
            "Return STRICT JSON with keys: clusters.\n"
            "clusters is a list of {name, weight, skills}.\n"
            "Allowed names: must_have, important, nice_to_have, implicit.\n"
            "weights must sum to 1.0 exactly.\n"
            "skills must be concise canonical names (e.g., python, rag, langchain, vector database, fastapi, docker, evaluation).\n"
            "Do not include any prose outside JSON."
        )
        raw = call_llm(system, jd)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("clusters"), list):
                return parsed
        except Exception:
            pass

    # Heuristic fallback: seed from known skills and fixed weights.
    clusters = [
        {
            "name": "must_have",
            "weight": 0.45,
            "skills": ["python", "llm", "rag", "embeddings", "vector database", "langchain", "fastapi"],
        },
        {
            "name": "important",
            "weight": 0.30,
            "skills": ["evaluation", "prompt engineering", "etl", "docker", "aws", "gcp", "azure", "sql"],
        },
        {
            "name": "nice_to_have",
            "weight": 0.15,
            "skills": ["huggingface", "pytorch", "tensorflow", "kubernetes", "mlops", "security"],
        },
        {
            "name": "implicit",
            "weight": 0.10,
            "skills": ["recent genai work", "shipped products", "measurable impact"],
        },
    ]
    return {"clusters": clusters}


def _validate_and_normalize_cluster_weights(jd_clusters: Dict[str, Any]) -> Dict[str, Any]:
    clusters = jd_clusters.get("clusters") if isinstance(jd_clusters, dict) else None
    if not isinstance(clusters, list) or not clusters:
        jd_clusters = _parse_jd_into_clusters(_default_genai_engineer_jd())
        clusters = jd_clusters["clusters"]

    total = 0.0
    cleaned = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip().lower()
        if name not in {"must_have", "important", "nice_to_have", "implicit"}:
            continue
        weight = float(c.get("weight") or 0.0)
        skills = c.get("skills") or []
        if not isinstance(skills, list):
            skills = []
        skills_norm = []
        for s in skills:
            ss = _normalize_skill(str(s))
            if ss:
                skills_norm.append(ss)
        weight = max(0.0, weight)
        total += weight
        cleaned.append({"name": name, "weight": weight, "skills": sorted(set(skills_norm))})

    if not cleaned:
        cleaned = _parse_jd_into_clusters(_default_genai_engineer_jd())["clusters"]
        return _validate_and_normalize_cluster_weights({"clusters": cleaned})

    # Normalize to sum to 1.0
    if total <= 0:
        for c in cleaned:
            c["weight"] = 1.0 / len(cleaned)
        return {"clusters": cleaned}

    for c in cleaned:
        c["weight"] = c["weight"] / total

    # Final rounding fix to ensure exact 1.0 sum (for UI/requirements)
    s = sum(c["weight"] for c in cleaned)
    delta = 1.0 - s
    cleaned[0]["weight"] = cleaned[0]["weight"] + delta
    return {"clusters": cleaned}


def _resume_metadata(text: str, source_file: str) -> Dict[str, Any]:
    yrs = _extract_years_experience(text)  # may be None
    latest_year = _extract_latest_year(text)  # may be None
    recency_years = None
    if latest_year is not None:
        recency_years = max(0, _CURRENT_YEAR - latest_year)

    skills = _extract_skills_rule_based(text)
    return {
        "source_file": source_file,
        "years_experience": yrs,
        "latest_year": latest_year,
        "recency_years": recency_years,
        "skills": skills,
    }


def _metadata_filter_pass(meta: Dict[str, Any]) -> bool:
    # Conservative early filter: at least one of these signals should exist.
    skills = meta.get("skills") or []
    yrs = meta.get("years_experience")
    latest = meta.get("latest_year")
    return bool(skills) or (yrs is not None and yrs >= 1) or (latest is not None)


@dataclass
class CandidateMatcherIndex:
    vector_db: Any
    bm25: BM25Retriever | None
    resumes: List[Dict[str, Any]]  # {id, text, metadata}


def _build_citations(docs_with_scores: List[tuple[Document, float]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for idx, (d, score) in enumerate(docs_with_scores):
        src = d.metadata.get("source_file") or d.metadata.get("source") or "unknown"
        preview = (d.page_content or "")[:260].replace("\n", " ").strip()
        page = d.metadata.get("page")
        location = None
        if page is not None:
            try:
                location = f"page {int(page) + 1}"
            except Exception:
                location = f"page {page}"
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


def build_candidate_matcher_index(resumes_dir: str) -> CandidateMatcherIndex:
    """
    Ingest resumes, extract metadata, build dense embeddings + sparse BM25,
    and store a local vector DB for retrieval at match-time.
    """
    if Chroma is None:
        raise ImportError(
            "Chroma is not installed. Install requirements (and on Windows/Python 3.12 you may need Microsoft C++ Build Tools) "
            "to build chroma-hnswlib."
        )
    # Load resumes from directory using existing loaders where possible.
    all_docs: List[Document] = []
    resumes: List[Dict[str, Any]] = []

    for filename in os.listdir(resumes_dir):
        path = os.path.join(resumes_dir, filename)
        if os.path.isdir(path):
            continue

        loader = None
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(path)
        elif path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
        elif path.endswith(".docx"):
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue

        docs = loader.load()
        # Collapse multi-page into one resume text for scoring; still keep chunks for retrieval.
        full_text = "\n".join(d.page_content for d in docs if d.page_content).strip()
        meta = _resume_metadata(full_text, filename)
        meta["resume_id"] = filename
        if not _metadata_filter_pass(meta):
            # Keep, but marked as low-signal.
            meta["low_signal"] = True
        resumes.append({"id": filename, "text": full_text, "metadata": meta})

        for d in docs:
            d.metadata["source_file"] = filename
            d.metadata["resume_id"] = filename
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No resumes found to index.")

    # Chunk for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = text_splitter.split_documents(all_docs)

    # Clear safely
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            pass

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # BM25 over chunks for sparse matching
    bm25 = BM25Retriever.from_documents(chunks)

    return CandidateMatcherIndex(
        vector_db=vector_db,
        bm25=bm25,
        resumes=resumes,
    )


def _hybrid_retrieve_for_resume(query: str, idx: CandidateMatcherIndex, resume_id: str, k_dense: int = 6, k_bm25: int = 6) -> List[tuple[Document, float]]:
    """
    Hybrid retrieval scoped to a single resume via metadata filter.
    Ensemble: dense + bm25. Returns docs with normalized scores (0-100).
    """
    # Dense scoped
    dense = idx.vector_db.similarity_search_with_score(
        query,
        k=k_dense,
        filter={"resume_id": resume_id},
    )
    # BM25: retrieve globally, then filter to resume_id
    keyword_docs: List[Document] = []
    if idx.bm25 is not None:
        try:
            keyword_docs = idx.bm25.invoke(query)
        except Exception:
            keyword_docs = []
    keyword_docs = [d for d in keyword_docs if (d.metadata.get("resume_id") == resume_id)][:k_bm25]

    combined: Dict[str, Dict[str, Any]] = {}

    def _key(d: Document) -> str:
        return f"{d.metadata.get('resume_id','?')}|{hash(d.page_content)}"

    for r, (d, _s) in enumerate(dense):
        combined[_key(d)] = {"doc": d, "dense_rank": r, "keyword_rank": None}

    for r, d in enumerate(keyword_docs):
        k = _key(d)
        if k in combined:
            combined[k]["keyword_rank"] = r
        else:
            combined[k] = {"doc": d, "dense_rank": None, "keyword_rank": r}

    def _ens(entry: Dict[str, Any]) -> float:
        d = entry["dense_rank"]
        k = entry["keyword_rank"]
        if d is None and k is None:
            return 1e9
        if d is None:
            return k + 3
        if k is None:
            return d
        return (d + k) / 2

    ranked = sorted(combined.values(), key=_ens)
    n = len(ranked) or 1
    out: List[tuple[Document, float]] = []
    for i, e in enumerate(ranked):
        out.append((e["doc"], round(100.0 * (n - i) / n, 1)))
    return out


def skill_match_scoring_tool(
    candidate: Dict[str, Any],
    idx: CandidateMatcherIndex,
    jd_clusters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Scores candidate across weighted skill clusters and outputs structured JSON.
    """
    text = candidate["text"]
    meta = candidate["metadata"]
    clusters = jd_clusters["clusters"]

    cluster_scores = []
    evidence = []

    for c in clusters:
        skills = c.get("skills") or []
        hits = []
        misses = []
        for s in skills:
            # Allow implicit signals to be scored by keywords.
            present = _skill_presence(text, s) if s in _SKILL_ALIASES else (s.lower() in text.lower())
            (hits if present else misses).append(s)

        # Retrieval-based evidence for this cluster
        q = " ".join(skills[:12]) if skills else c["name"]
        retrieved = _hybrid_retrieve_for_resume(q, idx, meta["resume_id"], k_dense=4, k_bm25=4)[:4]
        citations = _build_citations(retrieved)
        evidence.append({"cluster": c["name"], "query": q, "citations": citations})

        denom = max(1, len(skills))
        raw = len(hits) / denom
        cluster_scores.append(
            {
                "cluster": c["name"],
                "weight": float(c["weight"]),
                "score": round(100.0 * raw, 1),
                "hits": hits,
                "misses": misses,
            }
        )

    # Experience + recency signals
    yrs = meta.get("years_experience")
    latest_year = meta.get("latest_year")
    recency_years = meta.get("recency_years")

    exp_score = 0.5
    if yrs is not None:
        exp_score = _clamp01(_sigmoid((yrs - 3) / 1.4))  # ~3y baseline

    rec_score = 0.5
    if recency_years is not None:
        rec_score = _clamp01(_sigmoid((1.5 - recency_years) / 0.8))  # prefer <= ~1-2 years

    base = 0.0
    for cs in cluster_scores:
        base += (cs["weight"] * (cs["score"] / 100.0))

    # Final score blends skills + signals
    final = (0.80 * base) + (0.12 * exp_score) + (0.08 * rec_score)
    final = _clamp01(final)

    return {
        "candidate_id": candidate["id"],
        "metadata": {
            "years_experience": yrs,
            "latest_year": latest_year,
            "recency_years": recency_years,
            "skills_detected": meta.get("skills") or [],
        },
        "cluster_scores": cluster_scores,
        "signals": {
            "experience_score_0_1": round(exp_score, 3),
            "recency_score_0_1": round(rec_score, 3),
        },
        "final_fit_score": round(100.0 * final, 1),
        "evidence": evidence,
    }


def comparative_ranking_tool(
    top_candidates: List[Dict[str, Any]],
    jd_clusters: Dict[str, Any],
) -> List[str]:
    """
    Performs pairwise-ish refinement. Best effort LLM; fallback to keep current ordering.
    Returns ordered list of candidate_ids.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or len(top_candidates) <= 1:
        return [c["id"] for c in top_candidates]

    # Provide compact summaries to the LLM to reduce token use.
    payload = []
    for c in top_candidates:
        meta = c.get("metadata") or {}
        payload.append(
            {
                "candidate_id": c["id"],
                "years_experience": meta.get("years_experience"),
                "latest_year": meta.get("latest_year"),
                "skills_detected": (meta.get("skills") or [])[:18],
                "snippet": (c.get("text") or "")[:900],
            }
        )

    system = (
        "You are ranking candidates for a GenAI Engineer role.\n"
        "Given candidates with detected skills, experience, and snippets, return STRICT JSON:\n"
        "{ \"ordered_candidate_ids\": [..] }\n"
        "Rank by job-relevant skill depth and recency, not verbosity.\n"
        "Do not include any prose outside JSON."
    )
    user = json.dumps({"candidates": payload, "job_clusters": jd_clusters}, ensure_ascii=False)
    raw = call_llm(system, user)
    try:
        parsed = json.loads(raw)
        ids = parsed.get("ordered_candidate_ids")
        if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
            # Keep only known ids, preserve missing at end
            known = {c["id"] for c in top_candidates}
            out = [i for i in ids if i in known]
            for c in top_candidates:
                if c["id"] not in out:
                    out.append(c["id"])
            return out
    except Exception:
        pass

    return [c["id"] for c in top_candidates]


def match_explanation_generator(scored: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces readable summaries: strengths, gaps, final fit score, confidence.
    """
    # Strengths/gaps from cluster breakdown
    strengths = []
    gaps = []
    for c in scored["cluster_scores"]:
        if c["score"] >= 70:
            strengths.extend(c["hits"][:8])
        if c["score"] <= 40:
            gaps.extend(c["misses"][:8])

    strengths = sorted(set(strengths))[:12]
    gaps = sorted(set(gaps))[:12]

    # Confidence heuristic: high when evidence + signals present
    yrs = scored["metadata"].get("years_experience")
    rec = scored["metadata"].get("recency_years")
    skill_n = len(scored["metadata"].get("skills_detected") or [])
    confidence = "medium"
    if skill_n >= 8 and (rec is None or rec <= 2):
        confidence = "high"
    if skill_n <= 3 and (yrs is None or yrs < 2):
        confidence = "low"

    # Optional LLM narrative (best effort, still grounded by structured fields)
    narrative = None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        system = (
            "Write a concise candidate-to-job match explanation.\n"
            "Use ONLY the structured JSON provided.\n"
            "Output 5-8 bullet points with: strengths, gaps, and a final recommendation.\n"
            "No hallucinations; if unknown, say unknown."
        )
        user = json.dumps(scored, ensure_ascii=False)
        narrative = call_llm(system, user).strip()

    return {
        "strengths": strengths,
        "gaps": gaps,
        "confidence": confidence,
        "summary": narrative,
    }


def _dispatcher_stage(scored_fit: float) -> str:
    if scored_fit < 25:
        return "exclude"
    if scored_fit < 60:
        return "borderline"
    return "strong"


def run_candidate_matching(
    idx: CandidateMatcherIndex,
    job_description: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    End-to-end:
    - early filters
    - structured scoring
    - comparative refinement for borderline group
    - explanations for strong group
    """
    jd_clusters = _validate_and_normalize_cluster_weights(_parse_jd_into_clusters(job_description))

    scored = []
    # Score ALL uploaded resumes; low-signal ones will naturally get low confidence/score.
    for c in idx.resumes:
        s = skill_match_scoring_tool(c, idx, jd_clusters)
        stage = _dispatcher_stage(float(s["final_fit_score"]))
        s["stage"] = stage
        if c.get("metadata", {}).get("low_signal"):
            s["low_signal"] = True
        scored.append(s)

    # Base ranking by final_fit_score
    scored.sort(key=lambda x: float(x["final_fit_score"]), reverse=True)

    # Comparative refinement on top segment (borderline + strong mixed)
    top_segment_ids = [s["candidate_id"] for s in scored[: max(3, min(12, len(scored)))]]
    top_segment_candidates = [c for c in idx.resumes if c["id"] in set(top_segment_ids)]
    refined_order = comparative_ranking_tool(top_segment_candidates, jd_clusters)

    # Apply refined order to scored list for those ids; keep others afterwards
    id_to_scored = {s["candidate_id"]: s for s in scored}
    refined_scored = [id_to_scored[i] for i in refined_order if i in id_to_scored]
    remaining = [s for s in scored if s["candidate_id"] not in set(refined_order)]
    scored = refined_scored + remaining

    # Explanations for ALL candidates (including excluded/low-signal) for full transparency.
    results = []
    rank = 1
    limit = max(1, top_k)
    for s in scored[:limit]:
        explanation = match_explanation_generator(s)
        results.append(
            {
                "rank": rank,
                "candidate_id": s["candidate_id"],
                "final_fit_score": s["final_fit_score"],
                "stage": s.get("stage"),
                "low_signal": bool(s.get("low_signal")),
                "confidence": explanation.get("confidence", "medium"),
                "strengths": explanation.get("strengths", []),
                "gaps": explanation.get("gaps", []),
                "explanation": explanation.get("summary"),
                "details": s,
            }
        )
        rank += 1

    return {
        "job_clusters": jd_clusters,
        "excluded": [],
        "results": results,
    }