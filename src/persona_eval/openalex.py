"""Fetch source documents from the OpenAlex API with disk caching."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from tqdm import tqdm

from persona_eval.annotations import AnnotationEntry

logger = logging.getLogger(__name__)

OPENALEX_ID_RE = re.compile(r"(W\d+)")


def _extract_work_id(openalex_url: str) -> str | None:
    """Extract the work ID (e.g., 'W3111255098') from an OpenAlex URL or ID."""
    m = OPENALEX_ID_RE.search(openalex_url)
    return m.group(1) if m else None


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        for idx in indices:
            positions.append((idx, word))
    positions.sort()
    return " ".join(word for _, word in positions)


class OpenAlexClient:
    """Client for fetching paper data from OpenAlex with disk caching."""

    def __init__(self, cache_dir: str | Path = "cache", email: str | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum 0.1s between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)

    def get_work(self, openalex_url: str) -> dict | None:
        """Fetch a work record, using cache if available."""
        import requests

        work_id = _extract_work_id(openalex_url)
        if not work_id:
            logger.warning("Could not parse OpenAlex ID from: %s", openalex_url)
            return None

        cache_path = self.cache_dir / f"{work_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        self._rate_limit()
        url = f"https://api.openalex.org/works/{work_id}"
        params = {}
        if self.email:
            params["mailto"] = self.email

        try:
            resp = requests.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", work_id, e)
            return None

        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data

    def get_abstract(self, openalex_url: str) -> str | None:
        """Fetch the abstract for a single paper."""
        work = self.get_work(openalex_url)
        if not work:
            return None

        # Try abstract_inverted_index first (most common)
        inverted = work.get("abstract_inverted_index")
        if inverted:
            return _reconstruct_abstract(inverted)

        # Some works have abstract directly
        abstract = work.get("abstract")
        if abstract:
            return abstract

        logger.debug("No abstract found for %s", openalex_url)
        return None

    def get_title(self, openalex_url: str) -> str | None:
        """Fetch the title for a single paper."""
        work = self.get_work(openalex_url)
        if not work:
            return None
        title = work.get("title")
        if title:
            return title.strip()
        logger.debug("No title found for %s", openalex_url)
        return None

    def get_source_text(self, paper_ids: list[str]) -> str:
        """Fetch and concatenate abstracts for all papers in a query.

        Returns a single string with abstracts separated by newlines.
        """
        abstracts = []
        for pid in paper_ids:
            abstract = self.get_abstract(pid)
            if abstract:
                abstracts.append(abstract)
            else:
                logger.debug("Missing abstract for %s", pid)

        if not abstracts:
            logger.warning("No abstracts found for any of %d papers", len(paper_ids))
            return ""

        return "\n\n".join(abstracts)

    def get_reference_text(self, paper_ids: list[str]) -> str:
        """Fetch and concatenate titles for all papers in a query.

        Used as reference text for reference-based metrics (ROUGE, BERTScore).
        Returns a single string with titles separated by newlines.
        """
        titles = []
        for pid in paper_ids:
            title = self.get_title(pid)
            if title:
                titles.append(title)
            else:
                logger.debug("Missing title for %s", pid)

        if not titles:
            logger.warning("No titles found for any of %d papers", len(paper_ids))
            return ""

        return "\n".join(titles)

    def get_texts_batch(
        self,
        entries: list[AnnotationEntry],
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Fetch source and reference texts for all unique queries.

        Returns (source_texts, reference_texts) where:
            source_texts: {query_index: concatenated_abstracts}
            reference_texts: {query_index: concatenated_titles}
        """
        # Deduplicate: many annotators share the same query
        query_papers: dict[int, list[str]] = {}
        for entry in entries:
            if entry.query_index not in query_papers:
                query_papers[entry.query_index] = entry.paper_ids

        source_texts = {}
        reference_texts = {}
        for qi, paper_ids in tqdm(
            sorted(query_papers.items()),
            desc="Fetching source documents",
        ):
            source_texts[qi] = self.get_source_text(paper_ids)
            reference_texts[qi] = self.get_reference_text(paper_ids)

        return source_texts, reference_texts
