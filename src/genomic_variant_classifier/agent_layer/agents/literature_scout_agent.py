"""
literature_scout_agent.py — Literature & Feature Research Scout
===============================================================
Monitors PubMed, bioRxiv, and ClinGen for new publications on variant
pathogenicity, extracts proposed features or scoring methods, and surfaces
candidates for feature engineering review.

Messages emitted (outbox)
--------------------------
  FEATURE_CANDIDATE_ADDED (to TrainingLifecycleAgent)
      Emitted once per newly extracted feature candidate, immediately after
      it is written to SharedState. This gives TrainingLifecycleAgent real-
      time awareness of new candidates without waiting for its next scheduled
      run to poll the queue.

      Payload: {
          "candidate_name":    "<feature or score name>",
          "literature_source": "PubMed" | "bioRxiv" | "ClinGen",
          "pmid_or_doi":       "<identifier>",
          "paper_title":       "<str>",
          "relevance_score":   <float 0.0–1.0>,
          "extracted_at":      "<iso timestamp>"
      }
      Priority          : NORMAL
      Requires approval : False  (informational — TrainingLifecycle
                                  queues it for human review, does not
                                  auto-incorporate it)

Processing order inside run()
------------------------------
  1. Check if 7-day minimum interval has elapsed (existing logic).
  2. Fetch papers from PubMed, bioRxiv, ClinGen (existing logic).
  3. Score and filter candidates (existing logic).
  4. For each NEW candidate (not already in SharedState queue):
       a. Write to SharedState["literature"]["feature_candidates"] (existing).
       b. [NEW] Emit FEATURE_CANDIDATE_ADDED to TrainingLifecycleAgent.
  5. Render HTML digest (existing logic).
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# feedparser and requests are imported lazily inside their respective fetch
# methods so that the agent module loads cleanly even if they are not yet
# installed in the environment.

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.agent_layer.config import (
    BIORXIV_RSS_FEEDS,
    CLINGEN_API_BASE,
    LITERATURE_CANDIDATE_MIN_SCORE,
    LITERATURE_DIGEST_DIR,  # was REPORT_DIR
    LITERATURE_FEATURE_PATTERNS,
    LITERATURE_KNOWN_TOOLS,
    LITERATURE_MAX_PAPERS_PER_RUN,  # was LITERATURE_MAX_RESULTS
    LITERATURE_MIN_RELEVANCE,  # was LITERATURE_RELEVANCE_THRESHOLD
    LITERATURE_JOURNAL_ALLOWLIST,
    LITERATURE_JOURNAL_BOOST,
    LITERATURE_PUBMED_QUERIES,
    LITERATURE_RELEVANCE_KEYWORDS,  # was LITERATURE_KEYWORDS
    LITERATURE_ZENODO_QUERIES,
    ZENODO_API_BASE,
    NCBI_API_KEY,
    NCBI_EUTILS_BASE,
)
from genomic_variant_classifier.agent_layer.message_bus import FEATURE_CANDIDATE_ADDED, PRIORITY_NORMAL
from genomic_variant_classifier.agent_layer.shared_state import SharedState

# Minimum days between scout runs (not in config — defined here).
_LITERATURE_INTERVAL_DAYS = 7

# Compile feature extraction patterns from config.
# Config patterns use named group (?P<name>...).
_TRAINING_AGENT = "TrainingLifecycleAgent"

# Compile feature extraction patterns from config.
# Config patterns use named group (?P<n>...).
_COMPILED_PATTERNS = [re.compile(p, re.I) for p in LITERATURE_FEATURE_PATTERNS]


def _el_text(el) -> str:
    return el.text if (el is not None and el.text is not None) else ""


def _parse_pubmed_pub_date(article) -> str:
    """ISO-ish date from a PubmedArticle: ArticleDate -> PubDate (Year[-Month])
    -> MedlineDate. Returns '' if none found."""
    ad = article.find(".//ArticleDate")
    if ad is not None:
        y, m, d = _el_text(ad.find("Year")), _el_text(ad.find("Month")), _el_text(ad.find("Day"))
        if y:
            parts = [y]
            if m:
                parts.append(m.zfill(2))
            if d:
                parts.append(d.zfill(2))
            return "-".join(parts)
    pd_el = article.find(".//Journal/JournalIssue/PubDate")
    if pd_el is not None:
        y, m = _el_text(pd_el.find("Year")), _el_text(pd_el.find("Month"))
        if y:
            return f"{y}-{m}" if m else y
        medline = _el_text(pd_el.find("MedlineDate"))
        if medline:
            return medline
    return ""


def _parse_pubmed_article(article) -> dict:
    """Parse one <PubmedArticle> into a paper dict incl. journal/authors/publication_date."""
    pmid = _el_text(article.find(".//PMID"))
    title = _el_text(article.find(".//ArticleTitle"))
    abstract = " ".join((t.text or "") for t in article.findall(".//AbstractText")).strip()
    journal = (_el_text(article.find(".//Journal/Title"))
               or _el_text(article.find(".//Journal/ISOAbbreviation")))
    authors = []
    for au in article.findall(".//AuthorList/Author"):
        last = _el_text(au.find("LastName"))
        if last:
            authors.append(f"{last} {_el_text(au.find('Initials'))}".strip())
        else:
            coll = _el_text(au.find("CollectiveName"))
            if coll:
                authors.append(coll)
    return {
        "source": "PubMed",
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "journal": journal,
        "authors": "; ".join(authors),
        "publication_date": _parse_pubmed_pub_date(article),
    }


def _strip_html(text: str) -> str:
    """Strip HTML tags, then decode entities (order matters so &lt; survives)."""
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()


def _parse_zenodo_hit(hit: dict) -> dict:
    """Parse one Zenodo /api/records hit into a provenance-complete paper dict."""
    meta = hit.get("metadata", {}) or {}
    doi = hit.get("doi") or meta.get("doi") or (f"zenodo:{hit.get('id')}" if hit.get("id") else "")
    title = meta.get("title") or hit.get("title", "")
    abstract = _strip_html(meta.get("description", ""))
    links = hit.get("links", {}) or {}
    url = (links.get("html") or hit.get("doi_url")
           or (f"https://doi.org/{doi}" if doi and not str(doi).startswith("zenodo:") else ""))
    authors = "; ".join(c.get("name", "") for c in meta.get("creators", []) if c.get("name"))
    return {
        "source": "Zenodo",
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "url": url,
        "journal": "Zenodo",
        "authors": authors,
        "publication_date": meta.get("publication_date", ""),
    }


def _journal_relevance_boost(journal: str) -> float:
    """Modest boost when the paper's journal is a high-signal allow-listed venue."""
    j = (journal or "").lower()
    if j and any(name in j for name in LITERATURE_JOURNAL_ALLOWLIST):
        return LITERATURE_JOURNAL_BOOST
    return 0.0


class LiteratureScoutAgent(BaseAgent):
    """
    Monitors genomic literature and surfaces new feature candidates,
    notifying TrainingLifecycleAgent in real time via the MessageBus.
    """

    def __init__(self, shared_state: SharedState) -> None:
        super().__init__(shared_state)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        # ----------------------------------------------------------
        # Step 1: Check run interval (existing logic)
        # ----------------------------------------------------------
        if not self._should_run_literature():
            self.logger.info(
                "Literature scout not due (< %d days since last run). "
                "Use --pipeline literature to force.",
                _LITERATURE_INTERVAL_DAYS,
            )
            result = {"action": "skipped", "reason": "interval_not_elapsed"}
            self._log_finish(result)
            return result

        # ----------------------------------------------------------
        # Step 2: Fetch papers from all sources (existing logic)
        # ----------------------------------------------------------
        self._log_section("Fetching papers")
        pubmed_papers = self._fetch_pubmed()
        biorxiv_papers = self._fetch_biorxiv()
        clingen_papers = self._fetch_clingen()
        zenodo_papers = self._fetch_zenodo()
        all_papers = pubmed_papers + biorxiv_papers + clingen_papers + zenodo_papers
        self.logger.info("Total papers fetched: %d", len(all_papers))

        # ----------------------------------------------------------
        # Step 3: Score, filter, extract candidates (existing logic)
        # ----------------------------------------------------------
        self._log_section("Extracting feature candidates")
        section = self._get_section("literature")
        existing_ids = {
            c.get("pmid_or_doi") for c in section.get("feature_candidates", [])
        }
        existing_names = {
            c.get("name", "").lower() for c in section.get("feature_candidates", [])
        }

        new_candidates: list[dict] = []
        papers_processed = 0

        for paper in all_papers:
            score = self._relevance_score(paper)
            if score < LITERATURE_MIN_RELEVANCE:
                continue
            papers_processed += 1

            paper_id = paper.get("pmid") or paper.get("doi") or paper.get("url", "")
            if paper_id in existing_ids:
                continue  # already processed this paper

            candidates = self._extract_candidates(paper)
            for candidate_name in candidates:
                if candidate_name.lower() in existing_names:
                    continue
                if candidate_name.lower() in {
                    t.lower() for t in LITERATURE_KNOWN_TOOLS
                }:
                    continue

                now = datetime.now(timezone.utc).isoformat()
                candidate = {
                    "name": candidate_name,
                    "pmid_or_doi": paper_id,
                    "paper_title": paper.get("title", ""),
                    "literature_source": paper.get("source", "unknown"),
                    "authors": paper.get("authors", ""),
                    "publication_date": paper.get("publication_date", ""),
                    "journal": paper.get("journal", ""),
                    "relevance_score": round(score, 3),
                    "extracted_at": now,
                    "reviewed": False,
                    "incorporated": False,
                }
                new_candidates.append(candidate)
                existing_names.add(candidate_name.lower())

        # ----------------------------------------------------------
        # Step 4a: Write new candidates to SharedState (existing logic)
        # ----------------------------------------------------------
        if new_candidates:
            state = self._state.load()
            lit = state.setdefault("literature", {})
            queue = lit.setdefault("feature_candidates", [])
            queue.extend(new_candidates)
            lit["last_run"] = datetime.now(timezone.utc).isoformat()
            self._state.save(state)
            self.logger.info(
                "%d new feature candidate(s) added to queue.", len(new_candidates)
            )

            # ----------------------------------------------------------
            # Step 4b [NEW]: Emit FEATURE_CANDIDATE_ADDED per new candidate
            # ----------------------------------------------------------
            if not dry_run:
                for candidate in new_candidates:
                    self._emit_candidate(candidate)
            else:
                for candidate in new_candidates:
                    self.logger.info(
                        "  [dry-run] Would send FEATURE_CANDIDATE_ADDED → %s  "
                        "[candidate=%s  source=%s]",
                        _TRAINING_AGENT,
                        candidate["name"],
                        candidate["literature_source"],
                    )
        else:
            self.logger.info("No new feature candidates found.")
            # Still update last_run so the interval resets
            self._update_section(
                "literature",
                {"last_run": datetime.now(timezone.utc).isoformat()},
            )

        # ----------------------------------------------------------
        # Step 5: Render HTML digest (existing logic)
        # ----------------------------------------------------------
        digest_path = None
        if new_candidates and not dry_run:
            digest_path = self._render_digest(new_candidates)

        result = {
            "action": "literature_scout",
            "papers_fetched": len(all_papers),
            "papers_relevant": papers_processed,
            "new_candidates": len(new_candidates),
            "digest": digest_path,
            "messages_sent": len(new_candidates) if not dry_run else 0,
        }
        self._log_finish(result)
        return result

    # ------------------------------------------------------------------
    # NEW: emit FEATURE_CANDIDATE_ADDED
    # ------------------------------------------------------------------

    def _emit_candidate(self, candidate: dict) -> None:
        """
        Send a FEATURE_CANDIDATE_ADDED message to TrainingLifecycleAgent.

        does not require approval — TrainingLifecycle stores it for human
        review without acting on it automatically.
        """
        payload = {
            "candidate_name": candidate["name"],
            "literature_source": candidate["literature_source"],
            "pmid_or_doi": candidate.get("pmid_or_doi"),
            "paper_title": candidate.get("paper_title", ""),
            "authors": candidate.get("authors", ""),
            "publication_date": candidate.get("publication_date", ""),
            "journal": candidate.get("journal", ""),
            "relevance_score": candidate.get("relevance_score", 0.0),
            "extracted_at": candidate.get("extracted_at"),
        }
        self.send_message(
            to=_TRAINING_AGENT,
            subject=FEATURE_CANDIDATE_ADDED,
            payload=payload,
            priority=PRIORITY_NORMAL,
            requires_approval=False,
        )
        self.logger.info(
            "→ FEATURE_CANDIDATE_ADDED sent to %s  [candidate=%s]",
            _TRAINING_AGENT,
            candidate["name"],
        )

    # ------------------------------------------------------------------
    # Run interval check — unchanged
    # ------------------------------------------------------------------

    def _should_run_literature(self) -> bool:
        from datetime import timedelta

        section = self._get_section("literature")
        last_run = section.get("last_run")
        if not last_run:
            return True
        try:
            last_dt = datetime.fromisoformat(last_run)
            return (datetime.now(timezone.utc) - last_dt) >= timedelta(
                days=_LITERATURE_INTERVAL_DAYS
            )
        except ValueError:
            return True

    # ------------------------------------------------------------------
    # PubMed fetch — unchanged
    # ------------------------------------------------------------------

    def _fetch_pubmed(self) -> list[dict]:
        self.logger.info("Fetching PubMed papers …")
        papers: list[dict] = []
        try:
            import requests

            for query, max_results in LITERATURE_PUBMED_QUERIES:
                params: dict[str, Any] = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "sort": "date",
                    "retmode": "json",
                }
                if NCBI_API_KEY:
                    params["api_key"] = NCBI_API_KEY

                resp = requests.get(
                    f"{NCBI_EUTILS_BASE}/esearch.fcgi",
                    params=params,
                    timeout=20,
                )
                resp.raise_for_status()
                ids = resp.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    continue

                fetch_params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                }
                if NCBI_API_KEY:
                    fetch_params["api_key"] = NCBI_API_KEY

                fetch_resp = requests.get(
                    f"{NCBI_EUTILS_BASE}/efetch.fcgi",
                    params=fetch_params,
                    timeout=30,
                )
                fetch_resp.raise_for_status()
                root = ET.fromstring(fetch_resp.content)

                for article in root.findall(".//PubmedArticle"):
                    papers.append(_parse_pubmed_article(article))

            self.logger.info("PubMed: %d paper(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("PubMed fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # bioRxiv fetch — unchanged
    # ------------------------------------------------------------------

    def _fetch_biorxiv(self) -> list[dict]:
        self.logger.info("Fetching bioRxiv papers …")
        papers: list[dict] = []
        try:
            import feedparser

            for feed_url in BIORXIV_RSS_FEEDS:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:LITERATURE_MAX_PAPERS_PER_RUN]:
                    papers.append(
                        {
                            "source": "bioRxiv",
                            "doi": getattr(entry, "id", ""),
                            "title": getattr(entry, "title", ""),
                            "abstract": getattr(entry, "summary", ""),
                            "url": getattr(entry, "link", ""),
                            "journal": "bioRxiv",
                            "authors": getattr(entry, "author", ""),
                            "publication_date": getattr(entry, "published", "")
                            or getattr(entry, "updated", ""),
                        }
                    )
            self.logger.info("bioRxiv: %d paper(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("bioRxiv fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # ClinGen fetch — unchanged
    # ------------------------------------------------------------------

    def _fetch_clingen(self) -> list[dict]:
        self.logger.info("Fetching ClinGen gene validity data …")
        papers: list[dict] = []
        try:
            import requests

            resp = requests.get(
                f"{CLINGEN_API_BASE}.json" "?limit=20&sort=scoreDate&direction=DESC",
                timeout=20,
            )
            resp.raise_for_status()
            for record in resp.json().get("gene_validity_list", []):
                papers.append(
                    {
                        "source": "ClinGen",
                        "doi": record.get("uuid", ""),
                        "title": (
                            f"{record.get('gene', '')} — "
                            f"{record.get('disease', '')} "
                            f"({record.get('classification', '')})"
                        ),
                        "abstract": record.get("notes", ""),
                        "url": record.get("url", ""),
                        "journal": "ClinGen",
                        "authors": "",
                        "publication_date": record.get("scoreDate", "")
                        or record.get("date", ""),
                    }
                )
            self.logger.info("ClinGen: %d record(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("ClinGen fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # Zenodo fetch
    # ------------------------------------------------------------------

    def _fetch_zenodo(self) -> list[dict]:
        self.logger.info("Fetching Zenodo records ...")
        papers: list[dict] = []
        try:
            import requests

            for query, size in LITERATURE_ZENODO_QUERIES:
                resp = requests.get(
                    f"{ZENODO_API_BASE}/records",
                    params={"q": query, "size": size, "sort": "mostrecent"},
                    timeout=20,
                )
                resp.raise_for_status()
                for hit in resp.json().get("hits", {}).get("hits", []):
                    papers.append(_parse_zenodo_hit(hit))
            self.logger.info("Zenodo: %d record(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("Zenodo fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # Relevance scoring — unchanged
    # ------------------------------------------------------------------

    def _relevance_score(self, paper: dict) -> float:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        score = 0.0
        for kw in LITERATURE_RELEVANCE_KEYWORDS:
            kw_lower = kw.lower()
            score += text.count(kw_lower) * (
                0.3 if kw_lower in paper.get("title", "").lower() else 0.1
            )
        score += _journal_relevance_boost(paper.get("journal", ""))
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Feature candidate extraction — unchanged
    # ------------------------------------------------------------------

    def _extract_candidates(self, paper: dict) -> list[str]:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        candidates: list[str] = []
        for pattern in _COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    name = match.group("name").strip().rstrip(".,;:")
                except IndexError:
                    continue
                if 3 <= len(name) <= 60 and name not in candidates:
                    candidates.append(name)
        return candidates

    # ------------------------------------------------------------------
    # HTML digest rendering — unchanged
    # ------------------------------------------------------------------

    def _render_digest(self, candidates: list[dict]) -> str | None:
        try:
            report_dir = Path(LITERATURE_DIGEST_DIR)
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            report_path = report_dir / f"literature_digest_{timestamp}.html"

            rows = "".join(
                f"<tr>"
                f"<td>{html.escape(c['name'])}</td>"
                f"<td>{html.escape(c.get('literature_source', ''))}</td>"
                f"<td><a href='{c.get('pmid_or_doi', '')}' target='_blank'>"
                f"{html.escape(c.get('paper_title', ''))[:80]}…</a></td>"
                f"<td>{c.get('relevance_score', 0):.2f}</td>"
                f"</tr>"
                for c in candidates
            )
            report_path.write_text(
                f"""<!DOCTYPE html><html><head>
<meta charset='utf-8'>
<title>Literature Digest {timestamp}</title>
<style>
  body {{font-family:sans-serif;padding:1rem}}
  table {{border-collapse:collapse;width:100%}}
  th,td {{border:1px solid #ccc;padding:6px 10px;text-align:left}}
  th {{background:#f0f0f0}}
</style></head><body>
<h2>Literature Digest — {timestamp}</h2>
<p>{len(candidates)} new feature candidate(s) surfaced.</p>
<table>
<tr><th>Candidate</th><th>Source</th><th>Paper</th><th>Relevance</th></tr>
{rows}
</table></body></html>""",
                encoding="utf-8",
            )
            self.logger.info("Literature digest written: %s", report_path)
            return str(report_path)

        except Exception as exc:
            self.logger.warning("Digest render failed: %s", exc)
            return None
