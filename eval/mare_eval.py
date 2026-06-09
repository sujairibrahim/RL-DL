"""
remarl/eval/mare_eval.py
========================
Mathematical evaluation for REMARL matching MARE's benchmark methodology.

MARE Paper Evaluation (what we replicate):
  - Precision, Recall, F1 of extracted requirement entities/relations
    against gold-standard annotations
  - Evaluated on: Problem Diagrams, Use Case Diagrams, Goal Models
  - Compared against baselines: EPD, IT4RE, HAGM

This module implements THREE evaluation tiers:

  Tier 1 — Requirement Coverage F1  (primary, matches MARE Table 3)
     Treat each ground-truth requirement as a "gold entity".
     For each generated SRS requirement, match against gold via:
       - Exact string match
       - Token overlap (BLEU-1 / Jaccard)
       - Semantic match (sentence-BERT cosine ≥ threshold)
     Compute P, R, F1 exactly as MARE does.

  Tier 2 — Entity/Relation Extraction F1  (matches MARE Table 2)
     Extract domain entities and relations from the SRS.
     Match against scenario.domain_entities gold set.
     Compute P, R, F1 per extraction type.

  Tier 3 — NFR Coverage F1  (extends MARE)
     Match generated NFR statements against scenario.nfr gold.
     Separate F1 for each NFR category (performance, security, etc.)

FIXES APPLIED
─────────────
FIX [1] CRITICAL — NFR precision was computed with the wrong formula.
        n_covered counted gold NFRs found (a recall numerator), but was
        then divided by len(nfr_sections) (a prediction denominator).
        This mixed up the P and R roles and could produce values > 1.0.
        Fixed: now tracks gold_covered (recall) and pred_matched (precision)
        as separate sets in a single loop, giving true P/R/F1.

FIX [2] — Per-category NFR had P = R = F1 (all three set to the same
        coverage ratio). Fixed: precision and recall are now computed
        separately using category-specific keyword matching.

FIX [3] — _entity_prf() returned PRF(1.0, 1.0, 1.0) when no gold entities
        exist. This inflated macro entity F1 for every scenario without a
        domain_entities list. Fixed: returns P=0,R=1,F1=0 when there are
        predicted entities but no gold; returns (1,1,1) only when both are
        empty (vacuously perfect).

FIX [4] — semantic_sim field for token-matched RequirementMatch objects was
        left as 0.0 with a "will fill in pass 3" comment that was never
        executed. Added an optional post-pass to fill semantic_sim for
        token-matched pairs when the embedder is available.

FIX [5] — EvaluationSuite.aggregate() used n_conflicts = len(results),
        claiming "1 conflict per scenario". This assumption was wrong for
        scenarios with 0 or multiple conflicts.  Changed to track detected
        and resolved counts correctly, with a comment explaining the proxy.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PRF:
    """Precision / Recall / F1 triple."""
    precision: float
    recall: float
    f1: float
    n_predicted: int = 0
    n_gold: int = 0
    n_matched: int = 0

    def __repr__(self):
        return (
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"({self.n_matched}/{self.n_predicted} pred, "
            f"{self.n_matched}/{self.n_gold} gold)"
        )


@dataclass
class RequirementMatch:
    """A matched pair (gold_req → generated_req) with scores."""
    gold_req:    str
    matched_req: str
    exact_match: bool
    token_f1:    float       # token-level F1 (ROUGE-1 style)
    jaccard:     float       # Jaccard similarity of token sets
    semantic_sim: float      # cosine similarity from sentence-BERT
    match_type:  str         # "exact" | "token" | "semantic" | "none"


@dataclass
class MAREEvalResult:
    """Complete evaluation result for one SRS document."""
    domain: str

    # Tier 1 — Requirement Coverage (primary MARE metric)
    req_prf_exact:    PRF
    req_prf_token:    PRF
    req_prf_semantic: PRF

    # Tier 2 — Entity Extraction
    entity_prf: PRF

    # Tier 3 — NFR Coverage
    nfr_prf: PRF
    nfr_by_category: Dict[str, PRF] = field(default_factory=dict)

    # Hidden requirement elicitation
    hidden_req_recall: float = 0.0
    n_hidden: int = 0
    n_hidden_found: int = 0

    # Conflict detection
    conflict_detected: bool = False
    conflict_resolved: bool = False

    # Per-requirement detail
    matches: List[RequirementMatch] = field(default_factory=list)
    hallucinated_reqs: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  MARE-style Evaluation | Domain: {self.domain}",
            f"{'═'*65}",
            f"  Tier 1 — Requirement Coverage F1",
            f"    Exact   : {self.req_prf_exact}",
            f"    Token   : {self.req_prf_token}",
            f"    Semantic: {self.req_prf_semantic}  ← primary metric",
            f"",
            f"  Tier 2 — Entity Extraction F1",
            f"    {self.entity_prf}",
            f"",
            f"  Tier 3 — NFR Coverage F1",
            f"    {self.nfr_prf}",
        ]
        for cat, prf in self.nfr_by_category.items():
            lines.append(f"      {cat:<20}: {prf}")
        lines += [
            f"",
            f"  Hidden Req Recall   : {self.hidden_req_recall:.3f}  "
            f"({self.n_hidden_found}/{self.n_hidden} elicited)",
            f"  Conflict Detected   : {self.conflict_detected}",
            f"  Conflict Resolved   : {self.conflict_resolved}",
            f"  Hallucinated reqs   : {len(self.hallucinated_reqs)}",
            f"{'═'*65}",
        ]
        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """Aggregate results across all evaluation episodes."""
    n_episodes: int
    domains: List[str]

    macro_req_f1_exact:    float
    macro_req_f1_token:    float
    macro_req_f1_semantic: float
    macro_req_precision:   float
    macro_req_recall:      float

    macro_entity_f1:       float
    macro_nfr_f1:          float
    macro_hidden_recall:   float

    micro_req_f1_semantic: float
    micro_req_precision:   float
    micro_req_recall:      float

    std_req_f1_semantic: float
    std_req_f1_token:    float

    conflict_detection_rate:   float
    conflict_resolution_rate:  float

    per_domain:         Dict[str, MAREEvalResult] = field(default_factory=dict)
    delta_vs_baseline:  Dict[str, float]          = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  TOKENISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for",
    "with", "that", "this", "is", "are", "was", "be", "by",
    "on", "at", "as", "from", "it", "its", "shall", "must",
    "will", "should", "system", "user", "application",
}


def _tokenise(text: str) -> List[str]:
    """Lower-case tokenisation; strip punctuation; remove stopwords."""
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _token_f1(a: str, b: str) -> float:
    """ROUGE-1 / token-level F1 between two strings."""
    ta = set(_tokenise(a))
    tb = set(_tokenise(b))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    p = overlap / len(ta)
    r = overlap / len(tb)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity of token sets."""
    ta = set(_tokenise(a))
    tb = set(_tokenise(b))
    if not ta and not tb:
        return 1.0
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return len(ta & tb) / union


def _bleu1(candidate: str, reference: str) -> float:
    """Unigram BLEU (precision of candidate tokens in reference)."""
    cand_tokens = _tokenise(candidate)
    ref_tokens  = set(_tokenise(reference))
    if not cand_tokens:
        return 0.0
    return sum(1 for t in cand_tokens if t in ref_tokens) / len(cand_tokens)


# ─────────────────────────────────────────────────────────────────────────────
#  SRS REQUIREMENT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_requirements_from_srs(srs_text: str) -> List[str]:
    """
    Extract atomic requirement statements from a generated SRS.

    Strategy (priority order):
      1. Lines matching "FR-NNN:" or "REQ-NNN:" or "UC-NNN:" patterns
      2. Lines containing "shall" or "must" (formal requirement language)
      3. Numbered/bulleted list items with ≥ 8 words (fallback)
    """
    reqs: List[str] = []
    seen: set = set()
    lines = [l.strip() for l in srs_text.split("\n") if l.strip()]

    for line in lines:
        if len(line) < 15:
            continue
        if line.startswith("#") or line.startswith("---") or line.startswith("==="):
            continue

        # Strategy 1: Explicit req IDs
        if re.match(r"^(FR|NFR|REQ|UC|STK|BR)[-\s]?\d{1,4}[.:\)]", line, re.IGNORECASE):
            cleaned = re.sub(
                r"^(FR|NFR|REQ|UC|STK|BR)[-\s]?\d{1,4}[.:\)]\s*", "",
                line, flags=re.IGNORECASE,
            ).strip()
            if cleaned and cleaned not in seen:
                reqs.append(cleaned)
                seen.add(cleaned)
            continue

        # Strategy 2: Shall/must statements
        line_lower = line.lower()
        if ("shall" in line_lower or "must" in line_lower) and len(line.split()) >= 5:
            cleaned = re.sub(r"^[\*\-\•\d]+[.\)]\s*", "", line).strip()
            if cleaned and cleaned not in seen and len(cleaned) > 15:
                reqs.append(cleaned)
                seen.add(cleaned)

    # Fallback: long bulleted items if fewer than 3 requirements found
    if len(reqs) < 3:
        for line in lines:
            if re.match(r"^[\*\-\•]\s+", line) and len(line.split()) >= 8:
                cleaned = re.sub(r"^[\*\-\•]\s+", "", line).strip()
                if cleaned and cleaned not in seen:
                    reqs.append(cleaned)
                    seen.add(cleaned)

    return reqs


def extract_entities_from_srs(srs_text: str) -> List[str]:
    """
    Extract domain entities mentioned in the SRS.
    Looks for capitalised noun phrases not in common English vocabulary.
    """
    entities: set = set()

    # Pattern 1: Bold markdown terms **Entity**
    bold = re.findall(r"\*\*([A-Z][a-zA-Z\s]+)\*\*", srs_text)
    entities.update(b.strip() for b in bold if 2 <= len(b.split()) <= 4)

    # Pattern 2: Definition table first-column entries
    table_entries = re.findall(r"\|\s*([A-Z][a-zA-Z\s]+)\s*\|", srs_text)
    entities.update(e.strip() for e in table_entries if 1 <= len(e.split()) <= 3)

    # Pattern 3: "the X" where X is a capitalised compound noun
    noun_phrases = re.findall(
        r"\bthe\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", srs_text
    )
    entities.update(n.strip() for n in noun_phrases if 1 <= len(n.split()) <= 3)

    EXCLUDE = {
        "The", "This", "These", "Those", "Table", "Section", "Figure",
        "System", "User", "Application", "SRS", "Document",
    }
    return [e for e in entities if e not in EXCLUDE and len(e) > 2]


# ─────────────────────────────────────────────────────────────────────────────
#  MARE EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class MAREEvaluator:
    """
    Computes MARE-style P/R/F1 evaluation for a single SRS output.

    Matching hierarchy (same as MARE paper):
      1. Exact string match (case-insensitive, normalised whitespace)
      2. Token F1 ≥ token_threshold  (default 0.50)
      3. Semantic cosine ≥ sem_threshold  (default 0.65)
    """

    def __init__(
        self,
        token_threshold: float = 0.50,
        sem_threshold: float = 0.65,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.tok_thresh  = token_threshold
        self.sem_thresh  = sem_threshold
        self._model_name = model_name
        self._model      = None   # lazy load

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, srs_text: str, scenario) -> MAREEvalResult:
        """
        Full MARE-style evaluation of one SRS against one scenario.

        Args:
            srs_text: raw text of the generated SRS document
            scenario: Scenario dataclass from scenario_gen.py

        Returns:
            MAREEvalResult with all tiers computed
        """
        gold_reqs     = scenario.ground_truth_reqs
        gold_nfrs     = scenario.nfr
        gold_entities = scenario.domain_entities
        hidden_reqs   = scenario.hidden_reqs
        conflicts     = scenario.conflicts

        pred_reqs     = extract_requirements_from_srs(srs_text)
        pred_entities = extract_entities_from_srs(srs_text)

        logger.info(
            f"[{scenario.domain}] Predicted {len(pred_reqs)} reqs, "
            f"gold has {len(gold_reqs)}"
        )

        # ── Tier 1: Requirement matching ──────────────────────────────────────
        matches, hall    = self._match_requirements(pred_reqs, gold_reqs)
        req_prf_exact    = self._compute_prf(matches, "exact",    pred_reqs, gold_reqs)
        req_prf_token    = self._compute_prf(matches, "token",    pred_reqs, gold_reqs)
        req_prf_semantic = self._compute_prf(matches, "semantic", pred_reqs, gold_reqs)

        # ── Tier 2: Entity extraction ─────────────────────────────────────────
        entity_prf = self._entity_prf(pred_entities, gold_entities)

        # ── Tier 3: NFR coverage ──────────────────────────────────────────────
        nfr_prf, nfr_by_cat = self._nfr_prf(srs_text, gold_nfrs)

        # ── Hidden requirement elicitation ────────────────────────────────────
        hidden_found = self._count_hidden_found(srs_text, hidden_reqs)

        # ── Conflict detection / resolution ───────────────────────────────────
        conflict_detected, conflict_resolved = self._score_conflicts(
            srs_text, conflicts
        )

        return MAREEvalResult(
            domain=scenario.domain,
            req_prf_exact=req_prf_exact,
            req_prf_token=req_prf_token,
            req_prf_semantic=req_prf_semantic,
            entity_prf=entity_prf,
            nfr_prf=nfr_prf,
            nfr_by_category=nfr_by_cat,
            hidden_req_recall=hidden_found / max(len(hidden_reqs), 1),
            n_hidden=len(hidden_reqs),
            n_hidden_found=hidden_found,
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            matches=matches,
            hallucinated_reqs=hall,
        )

    # ── Requirement matching ───────────────────────────────────────────────────

    def _match_requirements(
        self,
        pred_reqs: List[str],
        gold_reqs: List[str],
    ) -> Tuple[List[RequirementMatch], List[str]]:
        """
        Greedy one-to-one matching: each gold req can be matched at most once.
        Each predicted req can match at most one gold req.
        Priority: exact > token > semantic.

        Returns: (matches, hallucinated_preds)
        """
        model      = self._get_model()
        gold_used  = [False] * len(gold_reqs)
        pred_used  = [False] * len(pred_reqs)
        matches:   List[RequirementMatch] = []

        # ── Pass 1: Exact matches ─────────────────────────────────────────────
        for pi, pred in enumerate(pred_reqs):
            pred_norm = self._normalise(pred)
            for gi, gold in enumerate(gold_reqs):
                if gold_used[gi]:
                    continue
                if pred_norm == self._normalise(gold):
                    matches.append(RequirementMatch(
                        gold_req=gold, matched_req=pred,
                        exact_match=True,
                        token_f1=1.0, jaccard=1.0, semantic_sim=1.0,
                        match_type="exact",
                    ))
                    gold_used[gi] = True
                    pred_used[pi] = True
                    break

        # ── Pass 2: Token-F1 matches ──────────────────────────────────────────
        token_matched_pairs: List[Tuple[int, int]] = []   # (pi, gi) for FIX [4]

        for pi, pred in enumerate(pred_reqs):
            if pred_used[pi]:
                continue
            best_score, best_gi = 0.0, -1
            for gi, gold in enumerate(gold_reqs):
                if gold_used[gi]:
                    continue
                score = _token_f1(pred, gold)
                if score > best_score:
                    best_score, best_gi = score, gi

            if best_score >= self.tok_thresh and best_gi >= 0:
                gold = gold_reqs[best_gi]
                matches.append(RequirementMatch(
                    gold_req=gold, matched_req=pred,
                    exact_match=False,
                    token_f1=best_score,
                    jaccard=_jaccard(pred, gold),
                    semantic_sim=0.0,   # FIX [4]: filled in post-pass below
                    match_type="token",
                ))
                gold_used[best_gi] = True
                pred_used[pi] = True
                token_matched_pairs.append((pi, best_gi))

        # ── Pass 3: Semantic matches ───────────────────────────────────────────
        unmatched_preds = [p for p, used in zip(pred_reqs, pred_used) if not used]
        unmatched_golds = [g for g, used in zip(gold_reqs, gold_used) if not used]

        if unmatched_preds and unmatched_golds and model is not None:
            from sentence_transformers import util
            pred_emb = model.encode(unmatched_preds, batch_size=32, show_progress_bar=False)
            gold_emb = model.encode(unmatched_golds, batch_size=32, show_progress_bar=False)
            sim_matrix = util.cos_sim(pred_emb, gold_emb).numpy()   # (P, G)

            assigned_p: set = set()
            assigned_g: set = set()
            pairs = sorted(
                [
                    (i, j, float(sim_matrix[i, j]))
                    for i in range(len(unmatched_preds))
                    for j in range(len(unmatched_golds))
                ],
                key=lambda x: -x[2],
            )
            for pi_local, gi_local, sim in pairs:
                if pi_local in assigned_p or gi_local in assigned_g:
                    continue
                if sim >= self.sem_thresh:
                    pred = unmatched_preds[pi_local]
                    gold = unmatched_golds[gi_local]
                    matches.append(RequirementMatch(
                        gold_req=gold, matched_req=pred,
                        exact_match=False,
                        token_f1=_token_f1(pred, gold),
                        jaccard=_jaccard(pred, gold),
                        semantic_sim=sim,
                        match_type="semantic",
                    ))
                    assigned_p.add(pi_local)
                    assigned_g.add(gi_local)

            hallucinated = [
                unmatched_preds[i]
                for i in range(len(unmatched_preds))
                if i not in assigned_p
            ]
        else:
            hallucinated = unmatched_preds

        # FIX [4]: fill semantic_sim for token-matched pairs using a single
        # batch encode so the RequirementMatch objects are complete and accurate
        # when used for detailed inspection or reporting.
        if token_matched_pairs and model is not None:
            from sentence_transformers import util as u
            tok_preds = [pred_reqs[pi] for pi, _ in token_matched_pairs]
            tok_golds = [gold_reqs[gi] for _, gi in token_matched_pairs]
            pe = model.encode(tok_preds, batch_size=32, show_progress_bar=False)
            ge = model.encode(tok_golds, batch_size=32, show_progress_bar=False)
            sims = u.cos_sim(pe, ge).numpy()
            # Find the token matches in the matches list and fill semantic_sim
            token_match_idx = 0
            for m in matches:
                if m.match_type == "token" and token_match_idx < len(token_matched_pairs):
                    m.semantic_sim = float(sims[token_match_idx, token_match_idx])
                    token_match_idx += 1

        return matches, hallucinated

    def _compute_prf(
        self,
        matches: List[RequirementMatch],
        level: str,
        pred_reqs: List[str],
        gold_reqs: List[str],
    ) -> PRF:
        """
        Compute P/R/F1 at a given match level (exact | token | semantic).

        At level 'exact': only exact matches count.
        At level 'token': exact + token matches count (cumulative).
        At level 'semantic': all match types count (most lenient).
        """
        LEVEL_ORDER = ["exact", "token", "semantic"]
        level_idx   = LEVEL_ORDER.index(level)
        level_types = LEVEL_ORDER[:level_idx + 1]
        valid       = [m for m in matches if m.match_type in level_types]

        n_matched = len(valid)
        n_pred    = len(pred_reqs)
        n_gold    = len(gold_reqs)

        precision = n_matched / max(n_pred, 1)
        recall    = n_matched / max(n_gold, 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        return PRF(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            n_predicted=n_pred,
            n_gold=n_gold,
            n_matched=n_matched,
        )

    # ── Entity matching ────────────────────────────────────────────────────────

    def _entity_prf(
        self,
        pred_entities: List[str],
        gold_entities: List[str],
    ) -> PRF:
        """
        Token-F1 based entity matching.
        An entity is matched if its token F1 with any gold entity ≥ 0.5.

        FIX [3]: Old code returned PRF(1.0, 1.0, 1.0) when no gold entities
        exist. This inflated the macro entity F1 average for every scenario
        without a domain_entities list.

        New behaviour:
          - Both empty (no gold AND no predicted): vacuously perfect → (1,1,1)
          - Gold empty but predictions exist: hallucination → P=0, R=1 vacuous
          - Gold non-empty: normal P/R/F1 computation
        """
        if not gold_entities:
            if not pred_entities:
                # Nothing expected, nothing predicted — vacuously perfect.
                return PRF(1.0, 1.0, 1.0, 0, 0, 0)
            else:
                # We predicted entities but none were expected — all hallucinated.
                return PRF(0.0, 1.0, 0.0, len(pred_entities), 0, 0)

        matched_preds = 0
        gold_matched  = [False] * len(gold_entities)

        for pred in pred_entities:
            for gi, gold in enumerate(gold_entities):
                if not gold_matched[gi] and _token_f1(pred, gold) >= 0.5:
                    matched_preds += 1
                    gold_matched[gi] = True
                    break

        n_gold_matched = sum(gold_matched)
        p  = matched_preds    / max(len(pred_entities), 1)
        r  = n_gold_matched   / max(len(gold_entities), 1)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

        return PRF(
            round(p, 4), round(r, 4), round(f1, 4),
            len(pred_entities), len(gold_entities), n_gold_matched,
        )

    # ── NFR scoring ────────────────────────────────────────────────────────────

    NFR_CATEGORIES = {
        "performance":     ["performance", "response time", "latency", "throughput",
                            "speed", "page load", "second", "millisecond"],
        "security":        ["security", "encrypt", "authenticat", "authoriz",
                            "password", "ssl", "tls", "gdpr", "pci", "aes"],
        "reliability":     ["availab", "reliab", "uptime", "99", "failover",
                            "backup", "recover", "fault"],
        "scalability":     ["scalab", "concurrent", "user load", "horizontal",
                            "vertical", "elastic", "cloud"],
        "usability":       ["usab", "accessibility", "wcag", "intuitive",
                            "user-friendly", "onboard", "help"],
        "maintainability": ["maintain", "modular", "document", "log",
                            "monit", "updat", "version"],
    }

    def _nfr_prf(
        self,
        srs_text: str,
        gold_nfrs: List[str],
    ) -> Tuple[PRF, Dict[str, PRF]]:
        """
        Score NFR coverage both overall and per-category.

        FIX [1]: The original precision formula was:
            p = n_covered / len(nfr_sections)
        where n_covered counted GOLD NFRs found — making the numerator a
        recall count and the denominator a prediction count.  This mixed up
        P and R roles.  For certain inputs it could also produce p > 1.0.

        Fixed: now tracks two separate sets:
          - gold_covered: gold NFRs that appear in any SRS section (→ recall)
          - pred_matched: SRS sections that match at least one gold NFR (→ precision)
        """
        if not gold_nfrs:
            return PRF(1.0, 1.0, 1.0, 0, 0, 0), {}

        nfr_sections   = self._extract_nfr_section(srs_text)
        sections_check = nfr_sections if nfr_sections else [srs_text]
        model          = self._get_model()

        # FIX [1]: track gold coverage and prediction matches separately.
        gold_covered: set = set()   # indices into gold_nfrs
        pred_matched: set = set()   # indices into sections_check

        for gi, nfr in enumerate(gold_nfrs):
            found = False
            for pi, section in enumerate(sections_check):
                if _token_f1(nfr, section) >= 0.3:
                    gold_covered.add(gi)
                    pred_matched.add(pi)
                    found = True
                    break

            # Semantic check if not found by token overlap
            if not found and model and nfr_sections:
                from sentence_transformers import util
                nfr_emb = model.encode([nfr], show_progress_bar=False)
                sec_emb = model.encode(nfr_sections, show_progress_bar=False)
                sims    = util.cos_sim(nfr_emb, sec_emb).numpy()[0]
                best_pi = int(np.argmax(sims))
                if float(sims[best_pi]) >= self.sem_thresh - 0.05:
                    gold_covered.add(gi)
                    pred_matched.add(best_pi)

        n_pred     = max(len(sections_check), 1)
        n_gold     = max(len(gold_nfrs), 1)
        n_pred_hit = len(pred_matched)    # SRS sections hitting ≥1 gold NFR
        n_gold_hit = len(gold_covered)    # gold NFRs covered by SRS

        p  = n_pred_hit / n_pred          # FIX [1]: true precision
        r  = n_gold_hit / n_gold          # recall
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

        overall = PRF(
            round(p, 4), round(r, 4), round(f1, 4),
            n_pred, n_gold, n_gold_hit,
        )

        # ── Per-category (FIX [2]: separate P and R) ──────────────────────────
        srs_lower  = srs_text.lower()
        by_cat: Dict[str, PRF] = {}

        for cat, keywords in self.NFR_CATEGORIES.items():
            gold_cat = [n for n in gold_nfrs if any(kw in n.lower() for kw in keywords)]
            if not gold_cat:
                continue

            # Count SRS "predicted" NFR mentions for this category
            srs_cat_mentions = sum(1 for kw in keywords if kw in srs_lower)

            # Count how many gold NFRs in this category are represented in the SRS
            n_gold_covered_cat = sum(
                1 for nfr in gold_cat
                if any(kw in nfr.lower() for kw in keywords)
                and any(kw in srs_lower for kw in keywords)
            )

            # FIX [2]: p and r are now distinct values (not always equal)
            p_c = (
                n_gold_covered_cat / max(srs_cat_mentions, 1)
                if srs_cat_mentions > 0 else 0.0
            )
            r_c  = n_gold_covered_cat / max(len(gold_cat), 1)
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0

            by_cat[cat] = PRF(
                round(p_c, 4), round(r_c, 4), round(f1_c, 4),
                srs_cat_mentions, len(gold_cat), n_gold_covered_cat,
            )

        return overall, by_cat

    def _extract_nfr_section(self, srs_text: str) -> List[str]:
        """Extract sentences from the NFR section of the SRS."""
        lines    = srs_text.split("\n")
        in_nfr   = False
        nfr_lines: List[str] = []
        for line in lines:
            l = line.strip().lower()
            if any(k in l for k in ["non-functional", "nfr", "performance req",
                                     "quality req", "system constraint"]):
                in_nfr = True
            elif in_nfr and line.startswith("#"):
                in_nfr = False   # new section ends NFR block
            if in_nfr and len(line.strip()) > 10:
                nfr_lines.append(line.strip())
        return nfr_lines if nfr_lines else []   # empty = caller falls back to [srs_text]

    # ── Hidden requirement scoring ─────────────────────────────────────────────

    def _count_hidden_found(self, srs_text: str, hidden_reqs: List[str]) -> int:
        """Count how many hidden requirements the agents surfaced."""
        if not hidden_reqs:
            return 0
        model = self._get_model()
        found = 0
        for req in hidden_reqs:
            if _token_f1(req, srs_text) >= 0.35:
                found += 1
                continue
            if model:
                sentences = [l.strip() for l in srs_text.split("\n") if len(l.strip()) > 10]
                if sentences:
                    from sentence_transformers import util
                    req_emb = model.encode([req], show_progress_bar=False)
                    sen_emb = model.encode(sentences[:100], show_progress_bar=False)
                    sims    = util.cos_sim(req_emb, sen_emb).numpy()[0]
                    if float(sims.max()) >= self.sem_thresh:
                        found += 1
        return found

    # ── Conflict scoring ───────────────────────────────────────────────────────

    def _score_conflicts(
        self, srs_text: str, conflicts: List[dict]
    ) -> Tuple[bool, bool]:
        """
        Detect whether conflicts were identified and resolved in the SRS.
        Returns (detected, resolved).
        """
        if not conflicts:
            return True, True   # no conflicts → vacuously true

        srs_lower = srs_text.lower()
        detection_words  = ["conflict", "contradict", "inconsist", "incompatible",
                             "tension", "disagree", "resolution", "negotiat"]
        resolution_words = ["resolved", "agreed", "compromise", "priority",
                            "unless", "except when", "shall be", "override"]

        detected = any(w in srs_lower for w in detection_words)
        resolved = any(w in srs_lower for w in resolution_words)

        for conflict in conflicts:
            req_a_terms = set(_tokenise(conflict.get("req_a", "")))
            req_b_terms = set(_tokenise(conflict.get("req_b", "")))
            context_has_a = any(t in srs_lower for t in list(req_a_terms)[:3])
            context_has_b = any(t in srs_lower for t in list(req_b_terms)[:3])
            if context_has_a and context_has_b and resolved:
                return True, True

        return detected, resolved

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Case-fold, collapse whitespace, strip punctuation for exact match."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _get_model(self):
        """Lazy-load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded sentence-transformers: {self._model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; semantic matching disabled."
                )
        return self._model


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationSuite:
    """
    Runs MARE-style evaluation across all RESimEnv scenarios.

    Usage:
        suite = EvaluationSuite()
        results = suite.evaluate_batch(srs_outputs, scenarios)
        suite.print_comparison_table(remarl_results, baseline_results)
        suite.print_latex_table(remarl_results, baseline_results)
    """

    def __init__(self, token_threshold: float = 0.50, sem_threshold: float = 0.65):
        self.evaluator = MAREEvaluator(
            token_threshold=token_threshold,
            sem_threshold=sem_threshold,
        )

    def evaluate_batch(
        self,
        srs_outputs: List[str],
        scenarios: List,
    ) -> List[MAREEvalResult]:
        """Evaluate a list of SRS outputs against their scenarios."""
        results: List[MAREEvalResult] = []
        for srs, scenario in zip(srs_outputs, scenarios):
            result = self.evaluator.evaluate(srs, scenario)
            results.append(result)
            logger.info(
                f"Evaluated {scenario.domain}: "
                f"F1={result.req_prf_semantic.f1:.3f}"
            )
        return results

    def aggregate(self, results: List[MAREEvalResult]) -> BenchmarkResult:
        """Compute macro and micro averages across all evaluation results."""
        if not results:
            raise ValueError("No results to aggregate")

        f1_exact    = [r.req_prf_exact.f1    for r in results]
        f1_token    = [r.req_prf_token.f1    for r in results]
        f1_semantic = [r.req_prf_semantic.f1 for r in results]
        prec_sem    = [r.req_prf_semantic.precision for r in results]
        recall_sem  = [r.req_prf_semantic.recall    for r in results]
        entity_f1   = [r.entity_prf.f1   for r in results]
        nfr_f1      = [r.nfr_prf.f1      for r in results]
        hidden_rec  = [r.hidden_req_recall for r in results]

        # Micro averages (pool matched/pred/gold counts)
        total_matched = sum(r.req_prf_semantic.n_matched   for r in results)
        total_pred    = sum(r.req_prf_semantic.n_predicted for r in results)
        total_gold    = sum(r.req_prf_semantic.n_gold      for r in results)
        micro_p  = total_matched / max(total_pred, 1)
        micro_r  = total_matched / max(total_gold, 1)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0

        # FIX [5]: count episodes with detected/resolved conflicts rather than
        # assuming 1 conflict per scenario.  conflict_detection_rate is
        # "fraction of episodes where the SRS mentioned conflict language";
        # it is a proxy, not a per-conflict ground-truth match.
        n_detected = sum(1 for r in results if r.conflict_detected)
        n_resolved = sum(1 for r in results if r.conflict_resolved)
        n_episodes = len(results)

        return BenchmarkResult(
            n_episodes=n_episodes,
            domains=[r.domain for r in results],
            macro_req_f1_exact=round(float(np.mean(f1_exact)), 4),
            macro_req_f1_token=round(float(np.mean(f1_token)), 4),
            macro_req_f1_semantic=round(float(np.mean(f1_semantic)), 4),
            macro_req_precision=round(float(np.mean(prec_sem)), 4),
            macro_req_recall=round(float(np.mean(recall_sem)), 4),
            macro_entity_f1=round(float(np.mean(entity_f1)), 4),
            macro_nfr_f1=round(float(np.mean(nfr_f1)), 4),
            macro_hidden_recall=round(float(np.mean(hidden_rec)), 4),
            micro_req_f1_semantic=round(micro_f1, 4),
            micro_req_precision=round(micro_p, 4),
            micro_req_recall=round(micro_r, 4),
            std_req_f1_semantic=round(float(np.std(f1_semantic)), 4),
            std_req_f1_token=round(float(np.std(f1_token)), 4),
            conflict_detection_rate=round(n_detected / max(n_episodes, 1), 4),  # FIX [5]
            conflict_resolution_rate=round(n_resolved / max(n_episodes, 1), 4), # FIX [5]
            per_domain={r.domain: r for r in results},
        )

    def compare_with_significance(
        self,
        remarl_results: List[MAREEvalResult],
        baseline_results: List[MAREEvalResult],
    ) -> dict:
        """
        Paired t-test (with Bonferroni correction) comparing REMARL vs baseline.
        Tests: F1 (primary), Precision, Recall, Hidden Recall — 4 hypotheses.
        Bonferroni alpha = 0.05 / 4 = 0.0125.
        """
        from scipy import stats

        if len(remarl_results) != len(baseline_results):
            raise ValueError(
                f"compare_with_significance: list length mismatch "
                f"({len(remarl_results)} vs {len(baseline_results)}). "
                f"Both lists must be paired (same scenarios)."
            )

        ALPHA      = 0.05
        N_TESTS    = 4
        ALPHA_BONF = ALPHA / N_TESTS

        metrics = {
            "req_f1_semantic": (
                [r.req_prf_semantic.f1        for r in remarl_results],
                [r.req_prf_semantic.f1        for r in baseline_results],
            ),
            "req_precision": (
                [r.req_prf_semantic.precision for r in remarl_results],
                [r.req_prf_semantic.precision for r in baseline_results],
            ),
            "req_recall": (
                [r.req_prf_semantic.recall    for r in remarl_results],
                [r.req_prf_semantic.recall    for r in baseline_results],
            ),
            "hidden_recall": (
                [r.hidden_req_recall          for r in remarl_results],
                [r.hidden_req_recall          for r in baseline_results],
            ),
        }

        out = {"alpha_bonferroni": ALPHA_BONF, "n": len(remarl_results)}
        print(f"\n{'─'*70}")
        print(f"  REMARL vs Baseline — Mathematical F1 Statistical Comparison")
        print(f"  Bonferroni alpha = {ALPHA_BONF}  ({N_TESTS} tests)")
        print(f"{'─'*70}")
        print(
            f"  {'Metric':<22} {'REMARL':>8} {'Baseline':>10} "
            f"{'Delta':>8}  {'p-value':>9}  Sig?"
        )
        print(f"{'─'*70}")

        for metric, (r_vals, b_vals) in metrics.items():
            r_arr = np.array(r_vals)
            b_arr = np.array(b_vals)
            diff  = r_arr - b_arr
            t, p  = stats.ttest_rel(r_arr, b_arr)
            d     = diff.mean() / (diff.std(ddof=1) + 1e-8)
            sig   = "✓" if p < ALPHA_BONF else "✗"
            print(
                f"  {metric:<22} {r_arr.mean():>8.4f} {b_arr.mean():>10.4f} "
                f"{diff.mean():>+8.4f}  {p:>9.4f}  {sig}"
            )
            out[metric] = {
                "remarl_mean":   round(float(r_arr.mean()), 4),
                "baseline_mean": round(float(b_arr.mean()), 4),
                "delta":         round(float(diff.mean()), 4),
                "p_value":       round(float(p), 4),
                "cohen_d":       round(float(d), 3),
                "significant":   bool(p < ALPHA_BONF),
            }
        print(f"{'─'*70}\n")
        return out

    def print_comparison_table(
        self,
        remarl_results: List[MAREEvalResult],
        baseline_results: List[MAREEvalResult],
        label_remarl: str = "REMARL",
        label_baseline: str = "MARE Baseline",
    ):
        r_agg = self.aggregate(remarl_results)
        b_agg = self.aggregate(baseline_results)

        def delta(a, b):
            d   = a - b
            pct = (d / max(abs(b), 1e-6)) * 100
            return f"{d:+.3f} ({pct:+.1f}%)"

        print(f"\n{'═'*80}")
        print(f"  Mathematical Evaluation: {label_remarl} vs {label_baseline}")
        print(f"  Method: MARE-style P/R/F1 | Matching: Exact + Token + Semantic")
        print(f"{'═'*80}")
        print(
            f"  {'Metric':<35} {label_remarl:>10} "
            f"{label_baseline:>14}  {'Delta':>18}"
        )
        print(f"{'─'*80}")

        rows = [
            ("Req F1 (Semantic) [PRIMARY]",
             r_agg.macro_req_f1_semantic,  b_agg.macro_req_f1_semantic),
            ("Req F1 (Token)",
             r_agg.macro_req_f1_token,     b_agg.macro_req_f1_token),
            ("Req F1 (Exact)",
             r_agg.macro_req_f1_exact,     b_agg.macro_req_f1_exact),
            ("Precision (Semantic)",
             r_agg.macro_req_precision,    b_agg.macro_req_precision),
            ("Recall (Semantic)",
             r_agg.macro_req_recall,       b_agg.macro_req_recall),
            ("Entity Extraction F1",
             r_agg.macro_entity_f1,        b_agg.macro_entity_f1),
            ("NFR Coverage F1",
             r_agg.macro_nfr_f1,           b_agg.macro_nfr_f1),
            ("Hidden Req Recall",
             r_agg.macro_hidden_recall,    b_agg.macro_hidden_recall),
            ("Conflict Detection Rate",
             r_agg.conflict_detection_rate,  b_agg.conflict_detection_rate),
            ("Conflict Resolution Rate",
             r_agg.conflict_resolution_rate, b_agg.conflict_resolution_rate),
        ]

        for label, r_val, b_val in rows:
            marker = " ★" if "PRIMARY" in label else ""
            print(
                f"  {label:<35} {r_val:>10.4f} {b_val:>14.4f}  "
                f"{delta(r_val, b_val):>18}{marker}"
            )

        print(f"{'─'*80}")
        print(
            f"  {'Micro F1 (pooled)':<35} {r_agg.micro_req_f1_semantic:>10.4f} "
            f"{b_agg.micro_req_f1_semantic:>14.4f}"
        )
        print(
            f"  {'Std Dev (Semantic F1)':<35} {r_agg.std_req_f1_semantic:>10.4f} "
            f"{b_agg.std_req_f1_semantic:>14.4f}"
        )
        print(f"{'═'*80}\n")

    def print_latex_table(
        self,
        remarl_results: List[MAREEvalResult],
        baseline_results: List[MAREEvalResult],
    ) -> str:
        """Generate LaTeX table code matching MARE's Table 3 format."""
        r_agg = self.aggregate(remarl_results)
        b_agg = self.aggregate(baseline_results)

        def fmt(v):
            return f"{v * 100:.1f}"

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            (r"\caption{Requirement Coverage Evaluation: REMARL vs. MARE Baseline "
             r"(Macro-averaged F1 across 14 domains)}"),
            r"\label{tab:eval_results}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Method & Prec & Rec & F1 & Entity F1 & NFR F1 & Hidden Recall \\",
            r"\midrule",
            (f"MARE Baseline & {fmt(b_agg.macro_req_precision)} & "
             f"{fmt(b_agg.macro_req_recall)} & {fmt(b_agg.macro_req_f1_semantic)} & "
             f"{fmt(b_agg.macro_entity_f1)} & {fmt(b_agg.macro_nfr_f1)} & "
             f"{fmt(b_agg.macro_hidden_recall)} \\\\"),
            (f"\\textbf{{REMARL}} & \\textbf{{{fmt(r_agg.macro_req_precision)}}} & "
             f"\\textbf{{{fmt(r_agg.macro_req_recall)}}} & "
             f"\\textbf{{{fmt(r_agg.macro_req_f1_semantic)}}} & "
             f"\\textbf{{{fmt(r_agg.macro_entity_f1)}}} & "
             f"\\textbf{{{fmt(r_agg.macro_nfr_f1)}}} & "
             f"\\textbf{{{fmt(r_agg.macro_hidden_recall)}}} \\\\"),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        latex = "\n".join(lines)
        print("\n% LaTeX Table (paste directly into paper):\n")
        print(latex)
        return latex

    def print_per_domain_table(
        self, results: List[MAREEvalResult], label: str = "REMARL"
    ):
        print(f"\n{'─'*80}")
        print(f"  Per-Domain Results — {label}")
        print(
            f"  {'Domain':<35} {'Prec':>7} {'Rec':>7} {'F1':>7}  "
            f"{'NFR F1':>8}  {'Hidden R':>9}  {'Conflict':>9}"
        )
        print(f"{'─'*80}")
        for r in sorted(results, key=lambda x: x.req_prf_semantic.f1, reverse=True):
            conflict_str = (
                "D+R" if r.conflict_resolved else
                ("D"  if r.conflict_detected else "—")
            )
            print(
                f"  {r.domain:<35} "
                f"{r.req_prf_semantic.precision:>7.3f} "
                f"{r.req_prf_semantic.recall:>7.3f} "
                f"{r.req_prf_semantic.f1:>7.3f}  "
                f"{r.nfr_prf.f1:>8.3f}  "
                f"{r.hidden_req_recall:>9.3f}  "
                f"{conflict_str:>9}"
            )
        print(f"{'─'*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI / SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    from sim.scenario_gen import ScenarioGenerator

    gen      = ScenarioGenerator("data/scenarios/")
    scenario = gen.sample(domain="e_commerce_marketplace")

    # Simulate a "perfect" SRS using all ground-truth requirements
    perfect_srs = "\n".join(
        ["# Software Requirements Specification", "## Functional Requirements"]
        + [f"- {r}" for r in scenario.ground_truth_reqs]
        + ["## Non-Functional Requirements"]
        + [f"- {n}" for n in scenario.nfr]
    )

    # Simulate a "partial" SRS (only visible reqs)
    partial_srs = "\n".join(
        ["# Software Requirements Specification", "## Functional Requirements"]
        + [f"- {r}" for r in scenario.visible_reqs]
        + ["## Non-Functional Requirements"]
        + [f"- {n}" for n in scenario.nfr[:2]]
    )

    evaluator = MAREEvaluator()

    print("\n=== Perfect SRS (should score ~1.0) ===")
    result_perfect = evaluator.evaluate(perfect_srs, scenario)
    print(result_perfect.summary())

    print("\n=== Partial SRS (visible reqs only, should score ~0.75) ===")
    result_partial = evaluator.evaluate(partial_srs, scenario)
    print(result_partial.summary())

    suite = EvaluationSuite()
    suite.print_comparison_table([result_perfect], [result_partial])
    suite.print_latex_table([result_perfect], [result_partial])