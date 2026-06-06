"""
remarl/sim/oracle.py
--------------------
Ground-truth quality scorer for REMARL training episodes.

The Oracle knows the correct answer (ground_truth_reqs from the scenario).
It scores the final SRS produced by the REMARL agent pipeline against
that ground truth using three complementary signals:

  1. Coverage   – semantic similarity between SRS and each ground-truth req
  2. Precision  – how much of the SRS is relevant (not hallucinated)
  3. Conflict   – whether known conflicts were identified and resolved

These three signals are combined into a single episode reward scalar
that the RL policies use for their delayed reward update.

During training:  oracle.score(final_workspace, scenario) is called once
                  per episode at termination.
During inference: oracle is not used (no ground truth available).
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────

@dataclass
class OracleResult:
    coverage_score: float       # [0,1] — how many ground-truth reqs were captured
    precision_score: float      # [0,1] — how relevant the SRS content is
    conflict_score: float       # [0,1] — conflict detection/resolution quality
    nfr_score: float            # [0,1] — non-functional requirements coverage
    total_reward: float         # weighted sum of above
    covered_reqs: List[str]     # which ground-truth reqs were found
    missed_reqs: List[str]      # which ground-truth reqs were missed
    hallucinated_count: int     # sentences in SRS not matching any ground truth
    details: dict               # per-req similarity scores for debugging


# ─────────────────────────────────────────────
#  Oracle
# ─────────────────────────────────────────────

class Oracle:
    """
    Scores a completed SRS against the ground-truth scenario.

    Uses sentence-transformers for semantic matching so that
    paraphrased requirements still score well.

    Weights (configurable):
      coverage  : 0.50  — most important training signal
      precision : 0.20  — penalise hallucination
      conflicts : 0.15  — reward finding contradictions
      nfr       : 0.15  — reward non-functional req coverage

    Args:
        model_name: sentence-transformers model to use.
                    "all-MiniLM-L6-v2" is fast and good enough.
                    "all-mpnet-base-v2" is slower but more accurate.
        coverage_threshold: cosine similarity above which a req is
                            considered "covered". Default 0.65.
        weights: tuple of (coverage, precision, conflict, nfr).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        coverage_threshold: float = 0.65,
        weights: Tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15),
    ):
        self.threshold = coverage_threshold
        self.w_cov, self.w_prec, self.w_conf, self.w_nfr = weights
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

        # Lazy import so the file is importable without sentence-transformers
        # installed (useful for unit tests with mock embeddings)
        self._model = None
        self._model_name = model_name

    # ── public API ───────────────────────────────────────────────────────

    def score(self, final_workspace, scenario) -> OracleResult:
        """
        Score the final workspace SRS against the scenario ground truth.

        Args:
            final_workspace: SharedWorkspace object (must have .get_srs())
            scenario:        Scenario object from ScenarioGenerator

        Returns:
            OracleResult with total_reward in [0, 1]
        """
        srs_text = self._extract_srs_text(final_workspace)

        if not srs_text or len(srs_text.strip()) < 50:
            logger.warning("Empty or very short SRS — returning zero reward.")
            return self._zero_result()

        coverage, covered, missed, details = self._score_coverage(
            srs_text, scenario.ground_truth_reqs
        )
        precision = self._score_precision(srs_text, scenario.ground_truth_reqs)
        conflict = self._score_conflict_handling(
            final_workspace, scenario.conflicts
        )
        nfr = self._score_nfr(srs_text, scenario.nfr)

        total = (
            self.w_cov  * coverage  +
            self.w_prec * precision +
            self.w_conf * conflict  +
            self.w_nfr  * nfr
        )
        total = float(np.clip(total, 0.0, 1.0))

        # Count hallucinated sentences (in SRS but not near any ground-truth)
        hallucinated = self._count_hallucinations(srs_text, scenario.ground_truth_reqs)

        result = OracleResult(
            coverage_score=coverage,
            precision_score=precision,
            conflict_score=conflict,
            nfr_score=nfr,
            total_reward=total,
            covered_reqs=covered,
            missed_reqs=missed,
            hallucinated_count=hallucinated,
            details=details,
        )

        logger.debug(
            f"Oracle result | domain={scenario.domain} | "
            f"cov={coverage:.3f} prec={precision:.3f} "
            f"conf={conflict:.3f} nfr={nfr:.3f} "
            f"total={total:.3f}"
        )
        return result

    def score_batch(self, workspaces, scenarios) -> List[OracleResult]:
        """Score a batch of episodes."""
        return [self.score(ws, sc) for ws, sc in zip(workspaces, scenarios)]

    def score_text(self, srs_text: str, user_stories: str = "") -> dict:
        """
        Score a completed SRS text heuristically — no ground truth required.
        Used in the HITL pipeline (inference mode) where no Scenario object exists.

        Scoring dimensions:
          - structure:    presence of IEEE 830 section markers
          - shall_density: density of well-formed requirement statements
          - nfr_coverage: presence of non-functional requirement keywords
          - clarity:      absence of ambiguous language
          - traceability: presence of requirement IDs (FR-xxx, NFR-xxx)

        Returns:
            dict with keys: structure, shall_density, nfr_coverage,
                            clarity, traceability, overall  — all in [0, 1]
        """
        if not srs_text or len(srs_text.strip()) < 50:
            return {k: 0.0 for k in
                    ["structure", "shall_density", "nfr_coverage",
                     "clarity", "traceability", "overall"]}

        text_lower = srs_text.lower()
        words = srs_text.split()
        word_count = max(len(words), 1)

        # 1. Structure score — IEEE 830 section headers present
        ieee_sections = [
            "introduction", "overall description", "system features",
            "functional requirement", "non-functional", "external interface",
            "appendix", "glossary", "traceability", "verification"
        ]
        found_sections = sum(1 for s in ieee_sections if s in text_lower)
        structure = min(1.0, found_sections / len(ieee_sections))

        # 2. Shall density — good requirement statements
        shall_count = text_lower.count("shall") + text_lower.count("must") + text_lower.count("will")
        shall_density = min(1.0, (shall_count / word_count) * 80)

        # 3. NFR coverage — non-functional keywords
        nfr_keywords = [
            "performance", "security", "scalab", "reliab", "availab",
            "usability", "maintainab", "portab", "response time",
            "throughput", "latency", "encrypt", "authenticat", "authoriz"
        ]
        nfr_hits = sum(1 for kw in nfr_keywords if kw in text_lower)
        nfr_coverage = min(1.0, nfr_hits / 6.0)

        # 4. Clarity — penalise vague language
        vague_words = [
            "etc", "and/or", "appropriate", "as necessary", "user-friendly",
            "fast", "efficient", "some", "many", "usually", "ideally",
            "might", "could", "may consider"
        ]
        vague_count = sum(text_lower.count(w) for w in vague_words)
        vague_rate = (vague_count / word_count) * 100
        clarity = max(0.0, 1.0 - min(1.0, vague_rate * 0.2))

        # 5. Traceability — requirement ID patterns (FR-001, NFR-002, REQ-003 etc.)
        import re
        req_ids = re.findall(r"\b(?:FR|NFR|REQ|UC|STK)[-\s]?\d{2,4}\b", srs_text, re.IGNORECASE)
        traceability = min(1.0, len(req_ids) / 10.0)

        # Weighted overall
        overall = (
            0.25 * structure +
            0.25 * shall_density +
            0.20 * nfr_coverage +
            0.15 * clarity +
            0.15 * traceability
        )

        return {
            "structure":     round(structure, 3),
            "shall_density": round(shall_density, 3),
            "nfr_coverage":  round(nfr_coverage, 3),
            "clarity":       round(clarity, 3),
            "traceability":  round(traceability, 3),
            "overall":       round(overall, 3),
        }

    # ── private scoring methods ──────────────────────────────────────────

    def _score_coverage(
        self,
        srs_text: str,
        ground_truth_reqs: List[str],
    ) -> Tuple[float, List[str], List[str], dict]:
        """
        For each ground-truth requirement, find the best-matching
        sentence or paragraph in the SRS via cosine similarity.

        Returns:
            (avg_coverage, covered_list, missed_list, per_req_details)
        """
        model = self._get_model()
        srs_sentences = self._split_sentences(srs_text)

        if not srs_sentences:
            return 0.0, [], ground_truth_reqs, {}

        # Encode everything at once for efficiency
        srs_embeddings = model.encode(srs_sentences, batch_size=32, show_progress_bar=False)
        req_embeddings = model.encode(ground_truth_reqs, batch_size=32, show_progress_bar=False)

        # Cosine similarity matrix: (n_reqs, n_sentences)
        from sentence_transformers import util
        sim_matrix = util.cos_sim(req_embeddings, srs_embeddings).numpy()  # (R, S)

        details = {}
        covered = []
        missed = []
        scores = []

        for i, req in enumerate(ground_truth_reqs):
            best_score = float(sim_matrix[i].max())
            best_sentence = srs_sentences[int(sim_matrix[i].argmax())]
            scores.append(best_score)
            details[req] = {
                "best_score": best_score,
                "matched_sentence": best_sentence,
                "covered": best_score >= self.threshold,
            }
            if best_score >= self.threshold:
                covered.append(req)
            else:
                missed.append(req)

        avg_coverage = float(np.mean(scores))
        return avg_coverage, covered, missed, details

    def _score_precision(
        self,
        srs_text: str,
        ground_truth_reqs: List[str],
    ) -> float:
        """
        Precision: fraction of SRS sentences that are semantically
        close to at least one ground-truth requirement.

        Low precision = agent hallucinated many irrelevant requirements.
        """
        model = self._get_model()
        srs_sentences = self._split_sentences(srs_text)

        if not srs_sentences:
            return 0.0

        srs_embeddings = model.encode(srs_sentences, batch_size=32, show_progress_bar=False)
        req_embeddings = model.encode(ground_truth_reqs, batch_size=32, show_progress_bar=False)

        from sentence_transformers import util
        sim_matrix = util.cos_sim(srs_embeddings, req_embeddings).numpy()  # (S, R)

        # A sentence is "relevant" if it matches any ground-truth req above threshold
        relevant = np.any(sim_matrix >= self.threshold, axis=1)
        return float(relevant.mean())

    def _score_conflict_handling(
        self,
        workspace,
        conflicts: List[dict],
    ) -> float:
        """
        Score whether the agents detected and addressed known conflicts.

        Checks:
          1. Was the conflict mentioned in the error report?
          2. Was a resolution documented in the final SRS?

        Returns 1.0 if no conflicts exist (no penalty for absent conflicts).
        """
        if not conflicts:
            return 1.0

        error_report = workspace.get("error_report", "").lower()
        final_srs = self._extract_srs_text(workspace).lower()

        scores = []
        for conflict in conflicts:
            req_a = conflict["req_a"].lower()
            req_b = conflict["req_b"].lower()
            conflict_type = conflict.get("type", "")

            # Check if key terms from both conflicting requirements appear
            # in the error report (detection signal)
            key_terms_a = set(req_a.split()[:4])  # first 4 words
            key_terms_b = set(req_b.split()[:4])
            detected_a = any(t in error_report for t in key_terms_a)
            detected_b = any(t in error_report for t in key_terms_b)
            detected = (detected_a and detected_b)

            # Check if resolution language appears in the SRS
            resolution_words = ["shall", "agreed", "resolved", "compromise", "priority", "unless"]
            resolution_mentioned = any(w in final_srs for w in resolution_words)

            score = 0.0
            if detected:
                score += 0.6
            if resolution_mentioned:
                score += 0.4

            scores.append(score)

        return float(np.mean(scores))

    def _score_nfr(self, srs_text: str, nfr_list: List[str]) -> float:
        """
        Score non-functional requirement coverage.
        NFRs are shorter and more precise than functional reqs,
        so we use a slightly lower threshold.
        """
        if not nfr_list:
            return 1.0

        model = self._get_model()
        srs_sentences = self._split_sentences(srs_text)

        if not srs_sentences:
            return 0.0

        srs_emb = model.encode(srs_sentences, batch_size=32, show_progress_bar=False)
        nfr_emb = model.encode(nfr_list, batch_size=32, show_progress_bar=False)

        from sentence_transformers import util
        sim_matrix = util.cos_sim(nfr_emb, srs_emb).numpy()

        nfr_threshold = self.threshold - 0.05  # slightly more lenient for NFRs
        scores = [float(sim_matrix[i].max()) for i in range(len(nfr_list))]
        covered = sum(1 for s in scores if s >= nfr_threshold)
        return covered / len(nfr_list)

    def _count_hallucinations(
        self,
        srs_text: str,
        ground_truth_reqs: List[str],
    ) -> int:
        """
        Count SRS sentences that don't match any ground-truth requirement.
        Used for logging/debugging, not included in reward.
        """
        model = self._get_model()
        srs_sentences = self._split_sentences(srs_text)
        if not srs_sentences:
            return 0

        srs_emb = model.encode(srs_sentences, batch_size=32, show_progress_bar=False)
        req_emb = model.encode(ground_truth_reqs, batch_size=32, show_progress_bar=False)

        from sentence_transformers import util
        sim_matrix = util.cos_sim(srs_emb, req_emb).numpy()

        hallucinated = np.all(sim_matrix < self.threshold, axis=1)
        return int(hallucinated.sum())

    # ── utility helpers ──────────────────────────────────────────────────

    def _extract_srs_text(self, workspace) -> str:
        """Pull all relevant text from the workspace into one string."""
        if hasattr(workspace, "get_srs"):
            return workspace.get_srs()
        # Fallback: concatenate all text fields
        parts = []
        for key in ["req_draft", "req_model", "srs_document", "user_stories"]:
            val = workspace.get(key, "") if hasattr(workspace, "get") else ""
            if val:
                parts.append(str(val))
        return "\n".join(parts)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentence-level units for scoring.
        Requirements documents are bulleted/numbered, so we split
        on both punctuation and list markers.
        """
        if not text:
            return []

        # Split on line breaks first (most RE docs are line-by-line)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # Further split long lines on sentence boundaries
        sentences = []
        for line in lines:
            # Remove numbering like "1." "FR-01:" etc.
            cleaned = re.sub(r"^\s*[\d\w\-]+[\.\)]\s*", "", line)
            # Split on ". " only if both sides have content
            parts = re.split(r"(?<=[a-z])\.\s+(?=[A-Z])", cleaned)
            sentences.extend(p.strip() for p in parts if len(p.strip()) > 10)

        return sentences[:200]  # cap at 200 sentences for performance

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded sentence-transformers model: {self._model_name}")
        return self._model

    def _zero_result(self) -> OracleResult:
        return OracleResult(
            coverage_score=0.0,
            precision_score=0.0,
            conflict_score=0.0,
            nfr_score=0.0,
            total_reward=0.0,
            covered_reqs=[],
            missed_reqs=[],
            hallucinated_count=0,
            details={},
        )


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Minimal mock workspace for testing
    class MockWorkspace:
        def get(self, key, default=""):
            data = {
                "req_draft": """
                    The system shall allow users to register and log in using email and password.
                    The system shall display a product catalogue with search and filter functionality.
                    The system shall provide a shopping cart that persists across sessions.
                    The system shall process payments via credit card, debit card, and PayPal.
                    The system shall send order confirmation emails to buyers.
                    The system shall allow buyers to leave ratings and reviews.
                """,
                "error_report": "Conflict detected: refund timeline conflict between buyer and seller requirements.",
            }
            return data.get(key, default)

    from sim.scenario_gen import ScenarioGenerator
    gen = ScenarioGenerator("data/scenarios/")
    scenario = gen.sample(domain="e_commerce_marketplace")

    oracle = Oracle()
    workspace = MockWorkspace()
    result = oracle.score(workspace, scenario)

    print("\n── Oracle Result ──")
    print(f"Coverage  : {result.coverage_score:.3f}")
    print(f"Precision : {result.precision_score:.3f}")
    print(f"Conflict  : {result.conflict_score:.3f}")
    print(f"NFR       : {result.nfr_score:.3f}")
    print(f"TOTAL     : {result.total_reward:.3f}")
    print(f"Covered   : {len(result.covered_reqs)}/{len(scenario.ground_truth_reqs)} reqs")
    print(f"Missed    : {len(result.missed_reqs)}")
    print(f"Hallucinated sentences: {result.hallucinated_count}")
    if result.missed_reqs:
        print("\nMissed requirements:")
        for r in result.missed_reqs:
            print(f"  - {r}")
