"""Syntactic complexity metric using spacy dependency parsing.

Reimplements the L2 Syntactic Complexity Analyzer (L2SCA) features from
summ-eval's SyntacticMetric, using spacy's dependency parser instead of
Stanford CoreNLP. This avoids the need for Java.

Reference: Xiaofei Lu, "Automatic analysis of syntactic complexity in second
language writing", International Journal of Corpus Linguistics, 2010.

This metric is reference-free and only analyzes the summary text.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)


def _count_structures(doc) -> dict[str, int]:
    """Count syntactic structures using spacy dependency parse.

    Returns raw counts of: words, sentences, verb phrases, clauses, T-units,
    dependent clauses, complex T-units, coordinate phrases, complex nominals.
    """
    words = sum(1 for t in doc if not t.is_punct and not t.is_space)
    sentences = len(list(doc.sents))

    verb_phrases = 0
    clauses = 0
    dependent_clauses = 0
    coordinate_phrases = 0
    complex_nominals = 0
    t_units = 0
    complex_t_units = 0

    for sent in doc.sents:
        # T-unit: independent clause + any dependent clauses attached to it
        # Approximate as each sentence root (main clause)
        t_units += 1

        has_dependent_clause = False

        for token in sent:
            # Verb phrases: verbs that head a phrase (not auxiliaries)
            if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass"):
                verb_phrases += 1

            # Clauses: tokens that are clause heads
            # advcl, ccomp, xcomp, relcl, acl are dependent clauses
            # Root is an independent clause
            if token.dep_ in ("ROOT", "advcl", "ccomp", "xcomp", "relcl", "acl",
                               "csubj", "csubjpass"):
                clauses += 1

            # Dependent clauses: subordinate clause types
            if token.dep_ in ("advcl", "ccomp", "xcomp", "relcl", "acl",
                               "csubj", "csubjpass"):
                dependent_clauses += 1
                has_dependent_clause = True

            # Coordinate phrases: tokens with cc (coordinating conjunction) child
            if any(child.dep_ == "cc" for child in token.children):
                if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "ADV"):
                    coordinate_phrases += 1

            # Complex nominals: noun phrases with modifiers
            # (prepositional phrase, relative clause, or appositive)
            if token.pos_ in ("NOUN", "PROPN") and token.dep_ in (
                "nsubj", "nsubjpass", "dobj", "pobj", "attr", "ROOT",
                "conj", "appos",
            ):
                has_complex_mod = any(
                    child.dep_ in ("relcl", "acl", "prep", "appos",
                                    "amod", "compound")
                    and child.pos_ not in ("DET", "NUM")
                    for child in token.children
                )
                if has_complex_mod:
                    complex_nominals += 1

        # Complex T-unit: a T-unit containing a dependent clause
        if has_dependent_clause:
            complex_t_units += 1

    return {
        "words": words,
        "sentences": sentences,
        "verb_phrases": verb_phrases,
        "clauses": max(clauses, 1),  # at least 1 to avoid division by zero
        "t_units": max(t_units, 1),
        "dependent_clauses": dependent_clauses,
        "complex_t_units": complex_t_units,
        "coordinate_phrases": coordinate_phrases,
        "complex_nominals": complex_nominals,
    }


def _safe_div(x: float, y: float) -> float:
    if y == 0:
        return 0.0
    return x / y


def _compute_complexity(counts: dict[str, int]) -> dict[str, float]:
    """Compute the 14 L2SCA syntactic complexity indices from raw counts."""
    w = counts["words"]
    s = counts["sentences"]
    vp = counts["verb_phrases"]
    c = counts["clauses"]
    t = counts["t_units"]
    dc = counts["dependent_clauses"]
    ct = counts["complex_t_units"]
    cp = counts["coordinate_phrases"]
    cn = counts["complex_nominals"]

    return {
        "syn_words": float(w),
        "syn_sentences": float(s),
        "syn_verb_phrases": float(vp),
        "syn_clauses": float(c),
        "syn_t_units": float(t),
        "syn_dependent_clauses": float(dc),
        "syn_complex_t_units": float(ct),
        "syn_coordinate_phrases": float(cp),
        "syn_complex_nominals": float(cn),
        "syn_words_per_sentence": _safe_div(w, s),
        "syn_words_per_t_unit": _safe_div(w, t),
        "syn_words_per_clause": _safe_div(w, c),
        "syn_clauses_per_sentence": _safe_div(c, s),
        "syn_verb_phrases_per_t_unit": _safe_div(vp, t),
        "syn_clauses_per_t_unit": _safe_div(c, t),
        "syn_dependent_clauses_per_clause": _safe_div(dc, c),
        "syn_dependent_clauses_per_t_unit": _safe_div(dc, t),
        "syn_t_units_per_sentence": _safe_div(t, s),
        "syn_complex_t_units_per_t_unit": _safe_div(ct, t),
        "syn_coordinate_phrases_per_t_unit": _safe_div(cp, t),
        "syn_coordinate_phrases_per_clause": _safe_div(cp, c),
        "syn_complex_nominals_per_t_unit": _safe_div(cn, t),
        "syn_complex_nominals_per_clause": _safe_div(cn, c),
    }


@register_metric("syntactic")
class SyntacticMetric(BaseMetric):
    """L2 Syntactic Complexity Analyzer using spacy dependency parsing.

    Computes 14 syntactic complexity indices from the summary text.
    This is a reference-free metric — only the summary is analyzed.

    Uses spacy instead of Stanford CoreNLP, so no Java installation is needed.
    """

    def __init__(self, **kwargs):
        self._nlp = None

    @property
    def name(self) -> str:
        return "Syntactic"

    def _load(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

    def score(self, summary: str, source: str) -> dict[str, float]:
        self._load()
        doc = self._nlp(summary)
        counts = _count_structures(doc)
        return _compute_complexity(counts)
