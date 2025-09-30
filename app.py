import os
import re
import time
import uuid
from difflib import SequenceMatcher
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_SEARCH_MIN_INTERVAL = float(os.getenv("GOOGLE_SEARCH_MIN_INTERVAL", "2.0"))
GOOGLE_SEARCH_MAX_RETRIES = int(os.getenv("GOOGLE_SEARCH_MAX_RETRIES", "1"))

_search_lock = Lock()
_last_search_time = 0.0

_WORD_RE = re.compile(r"[\w']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def enforce_search_rate_limit() -> None:
    if GOOGLE_SEARCH_MIN_INTERVAL <= 0:
        return

    global _last_search_time
    with _search_lock:
        now = time.monotonic()
        wait_for = (_last_search_time + GOOGLE_SEARCH_MIN_INTERVAL) - now
        if wait_for > 0:
            time.sleep(wait_for)
            now = time.monotonic()
        _last_search_time = now


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def generate_queries(topic: str) -> List[str]:
    clean_topic = topic.strip()
    if not clean_topic:
        return []

    llm = get_query_llm()
    queries: List[str] = []
    seen: set[str] = set()

    if llm is not None:
        prompt = [
            SystemMessage(
                content=(
                    "You create web search engine queries to help research complex topics. "
                    "Return between 1 and 5 focused, diverse search queries that would help "
                    "someone gather high-quality information about the topic. "
                    "Respond with one query per line and no additional commentary."
                )
            ),
            HumanMessage(content=f"Topic: {clean_topic}"),
        ]

        try:
            response = llm.invoke(prompt)
            raw_output = getattr(response, "content", "")
            for line in raw_output.splitlines():
                candidate = line.strip()
                if not candidate:
                    continue
                candidate = re.sub(r"^[0-9]+[\).\-]*\s*", "", candidate)
                candidate = candidate.lstrip("-• ").strip()
                if not candidate:
                    continue
                normalized = " ".join(candidate.split()).lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                queries.append(candidate)
                if len(queries) >= 5:
                    break
        except Exception:
            queries = []
            seen.clear()

    if not queries:
        fallback = " ".join(clean_topic.split())
        return [fallback]

    return queries


def snippet_score(text: str, topic_tokens: List[str]) -> float:
    """Score how relevant a piece of text is to the topic using token overlap and fuzziness."""
    if not text:
        return 0.0
    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0

    text_set = set(text_tokens)
    topic_set = set(topic_tokens)

    overlap = text_set & topic_set
    overlap_score = len(overlap) / max(len(topic_set), 1)

    matcher = SequenceMatcher(None, " ".join(topic_tokens), " ".join(text_tokens))
    similarity_score = matcher.ratio()

    return 0.6 * overlap_score + 0.4 * similarity_score


def score_result(result: Dict[str, Any], topic_tokens: List[str]) -> float:
    title = result.get("title", "")
    snippet = result.get("snippet") or ""

    snippet_component = snippet_score(snippet, topic_tokens)
    title_component = snippet_score(title, topic_tokens)

    return snippet_component + 0.3 * title_component


def google_search(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    missing = [
        name
        for name, value in (("GOOGLE_API_KEY", GOOGLE_API_KEY), ("GOOGLE_CSE_ID", GOOGLE_CSE_ID))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Google Custom Search credentials: " + ", ".join(missing)
        )

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": max_results,
        "safe": "active",
    }

    payload: Dict[str, Any] = {}
    last_error: Optional[Exception] = None
    max_attempts = max(GOOGLE_SEARCH_MAX_RETRIES, 1)

    for attempt in range(1, max_attempts + 1):
        enforce_search_rate_limit()

        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.HTTPError as exc:  # pragma: no cover - network failure
            status_code = exc.response.status_code if exc.response else None
            last_error = exc
            if status_code == 429 and attempt < max_attempts:
                backoff = GOOGLE_SEARCH_MIN_INTERVAL * max(2 ** (attempt - 1), 1)
                time.sleep(backoff)
                continue
            raise
        except requests.RequestException as exc:  # pragma: no cover - network failure
            last_error = exc
            if attempt < max_attempts:
                time.sleep(GOOGLE_SEARCH_MIN_INTERVAL * attempt)
                continue
            raise
    else:  # pragma: no cover - loop exhausted
        if last_error:
            raise last_error
        raise RuntimeError("Custom Search request failed without a response.")

    items = payload.get("items") or []

    normalized_results = []
    for item in items:
        normalized_results.append(
            {
                "title": item.get("title", "Untitled result"),
                "snippet": item.get("snippet", ""),
                "url": item.get("link"),
                "source": "google",
            }
        )

    return normalized_results


def duckduckgo_search(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    normalized_results: List[Dict[str, Any]] = []

    with DDGS() as ddgs:
        responses = ddgs.text(
            query,
            region="wt-wt",
            safesearch="moderate",
            timelimit=None,
            max_results=max_results,
        )
        for item in responses:
            url = item.get("href") or item.get("url")
            if not url:
                continue
            normalized_results.append(
                {
                    "title": item.get("title") or item.get("heading") or "Untitled result",
                    "snippet": item.get("body") or item.get("excerpt") or "",
                    "url": url,
                    "source": "duckduckgo",
                }
            )

    return normalized_results


def perform_search(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    # return duckduckgo_search(query, max_results=max_results)
    google_exception: Optional[Exception] = None

    try:
        results = google_search(query, max_results=max_results)
        if results:
            return results
    except Exception as exc:
        google_error = True
        google_exception = exc
    else:
        google_error = False

    try:
        fallback_results = duckduckgo_search(query, max_results=max_results)
        if fallback_results:
            return fallback_results
    except Exception:
        if google_error:
            if google_exception:
                raise google_exception
            raise

    if google_error:
        # If Google failed and fallback returned nothing, raise to trigger upstream error handling.
        if google_exception:
            raise google_exception
        raise RuntimeError("All search providers failed for query: " + query)

    return []


def fetch_page_content(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0; +https://example.com)"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    cleaned = " ".join(fragment.strip() for fragment in text.split())
    return cleaned


def build_chroma_documents(
    results: List[Dict[str, Any]], warnings: List[str]
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for result in results:
        url = result.get("url")
        if not url:
            continue

        page_text = ""
        try:
            page_text = fetch_page_content(url)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to fetch {url}: {exc}")

        snippet = result.get("snippet") or ""
        combined_text = (page_text or snippet).strip()
        if not combined_text:
            continue

        print(f"Fetched {len(combined_text)} characters from {url}")
        print(combined_text[:200] + "...")
        splitter = get_text_splitter()
        chunks = splitter.split_text(combined_text)
        if not chunks:
            continue

        base_id = uuid.uuid5(uuid.NAMESPACE_URL, url).hex
        title = result.get("title") or "Untitled result"

        for index, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{base_id}-{index}",
                    "content": chunk,
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                }
            )

    return documents


class ExtractiveSummaryLLM(LLM):
    """Fallback LLM that extracts key sentences when no API-based model is configured."""

    max_sentences: int = 8

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        question = ""
        context_section = prompt

        if "Question:" in prompt:
            before_question, after_question = prompt.split("Question:", maxsplit=1)
            context_section = before_question
            if "Helpful Answer:" in after_question:
                question, _ = after_question.split("Helpful Answer:", maxsplit=1)
                question = question.strip()
            else:
                question = after_question.strip()

        cleaned_lines: List[str] = []
        for line in context_section.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith("use the following pieces of context"):
                continue
            if lowered.startswith("context:"):
                stripped = stripped.split(":", maxsplit=1)[-1].strip()
            cleaned_lines.append(stripped)

        combined_context = " ".join(cleaned_lines).strip()
        if not combined_context:
            return "[ExtractiveSummaryLLM] Not enough information was retrieved to draft an answer."

        hits = [{"text": combined_context, "metadata": {}}]
        return synthesize_answer(question, hits)

    @property
    def _llm_type(self) -> str:
        return "extractive_summary"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


@lru_cache(maxsize=1)
def get_llm() -> Any:
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.5)
    return ExtractiveSummaryLLM()


@lru_cache(maxsize=1)
def get_query_llm() -> Optional[Any]:
    llm = get_llm()
    if isinstance(llm, ChatOpenAI):
        return llm
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
        )
    return None


@lru_cache(maxsize=1)
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def create_vectorstore(documents: List[Dict[str, Any]]) -> Optional[Chroma]:
    if not documents:
        return None

    langchain_docs: List[Document] = []
    for doc in documents:
        content = doc.get("content", "").strip()
        if not content:
            continue
        langchain_docs.append(
            Document(
                page_content=content,
                metadata={
                    "title": doc.get("title") or "Untitled result",
                    "url": doc.get("url"),
                    "snippet": doc.get("snippet", ""),
                    "id": doc.get("id"),
                },
            )
        )

    if not langchain_docs:
        return None

    embeddings = get_embeddings()
    collection_name = f"research_{uuid.uuid4().hex}"

    return Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name=collection_name,
    )


def run_retrieval_qa(
    vectorstore: Optional[Chroma], question: str, top_k: int = 5
) -> Tuple[str, List[Document]]:
    if vectorstore is None:
        return "[run_retrieval_qa:vectorstore] Not enough information was retrieved to draft an answer.", []

    retriever = vectorstore.as_retriever(search_kwargs={"k": max(top_k, 1)})
    llm = get_llm()
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt=prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = qa_chain.invoke({"input": question})
    answer_text = (result.get("answer") or "").strip()
    if not answer_text:
        answer_text = "[run retrieval qa] Not enough information was retrieved to draft an answer."

    source_documents = result.get("source_documents") or []
    return answer_text, source_documents


def synthesize_answer(topic: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "[synthesize_answer:hits] Not enough information was retrieved to draft an answer."

    sentences: List[str] = []
    seen = set()

    for hit in hits:
        text = hit.get("text", "")
        for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
            candidate = sentence.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            sentences.append(candidate)
            if len(sentences) >= 8:
                break
        if len(sentences) >= 8:
            break

    if not sentences:
        return "[synthesize_answer:sentences] Not enough information was retrieved to draft an answer."
    body = " ".join(sentences)
    
    return body



@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/research", methods=["POST"])
def research() -> Any:
    payload = request.get_json(silent=True) or {}
    topic = (payload.get("message") or "").strip()

    if not topic:
        return jsonify({"error": "Please provide a topic to research."}), 400

    queries = generate_queries(topic)
    if not queries:
        return jsonify({"error": "Unable to create search queries for the provided topic."}), 400

    topic_tokens = tokenize(topic)

    aggregated_results = []
    seen_urls = set()

    errors = []

    for query in queries:
        try:
            search_results = perform_search(query)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Search failed for '{query}': {exc}")
            continue

        scored = []
        for result in search_results:
            url = result.get("href") or result.get("url")
            if not url:
                continue

            score = score_result(result, topic_tokens)
            scored.append(
                {
                    "title": result.get("title", "Untitled result"),
                    "snippet": result.get("snippet") or "",
                    "url": url,
                    "score": score,
                    "query": query,
                }
            )

        top_results = sorted(scored, key=lambda item: item["score"], reverse=True)[:2]

        for item in top_results:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            aggregated_results.append(item)

    top_ten = sorted(aggregated_results, key=lambda item: item["score"], reverse=True)[:10]
    
    chroma_documents = build_chroma_documents(top_ten, errors)
    print(f"\nBuilt {len(chroma_documents)} Chroma documents from {len(top_ten)} results\n")
    
    vectorstore = create_vectorstore(chroma_documents)
    print(f"\nCreated vectorstore: {vectorstore is not None}\n")
    
    answer, source_documents = run_retrieval_qa(vectorstore, topic, top_k=5)
    print(f"\nGenerated answer with {len(source_documents)} source documents\n")
    
    answer_sources = [
        {
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("url"),
            "snippet": doc.metadata.get("snippet", ""),
        }
        for doc in source_documents
    ]

    response: Dict[str, Any] = {
        "topic": topic,
        "queries": queries,
        "results": top_ten,
        "answer": answer,
        "answer_sources": answer_sources,
    }

    if errors:
        response["warnings"] = errors

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
