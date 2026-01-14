from __future__ import annotations

import json
import threading
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
import logging

from langchain_text_splitters import CharacterTextSplitter

from .logging_utils import setup_logger

try:
    from pyserini.search import SimpleSearcher as LuceneSearcher
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    HAS_PYSERINI_ANALYSIS = True
except ImportError:
    HAS_PYSERINI_ANALYSIS = False

# Module-level logger for consistent logging across all functions
logger = logging.getLogger(__name__)

# Avoid repeated setup_logger() calls & make analyzer init thread-safe
_LOGGER_ONCE = False
_ANALYZER_LOCK = threading.Lock()

def _ensure_logger_configured() -> logging.Logger:
    global _LOGGER_ONCE
    if not _LOGGER_ONCE:
        setup_logger()
        _LOGGER_ONCE = True
    return logging.getLogger(__name__)


# ===== ATOMIC SCORING FUNCTIONS =====

def _calculate_itf_prob(
    term: str,
    q_counts: Counter[str],
    feedback_model: dict[str, float] | Counter[str] | set[str] | None,
    beta: float = 0.5
) -> float:
    """Calculate Interpolated Token Feedback (ITF) probability for a term.
    
    Formula: p_q^ITF(t) = beta * P(t|q) + (1 - beta) * P(t|R_rel)
    
    Args:
        term: The term to compute probability for
        q_counts: Query term counts
        feedback_model: Either:
            - dict/Counter of term -> probability (MLE from feedback docs)
            - set of relevant terms (backward compat: uniform distribution)
            - None (no feedback)
        beta: Interpolation weight (higher = more query, lower = more feedback)
    """
    # P(t|q) component: MLE from query
    q_len = sum(q_counts.values())
    p_q = q_counts[term] / q_len if q_len > 0 else 0.0
    
    # P(t|R_rel) component: MLE from feedback
    p_rel = 0.0
    if feedback_model is not None:
        if isinstance(feedback_model, set):
            # Backward compatibility: uniform distribution over relevant tokens
            if len(feedback_model) > 0:
                p_rel = (1.0 / len(feedback_model)) if term in feedback_model else 0.0
        elif isinstance(feedback_model, (dict, Counter)):
            # Proper MLE: weighted by term frequency in feedback docs
            total = sum(feedback_model.values()) if isinstance(feedback_model, Counter) else 1.0
            if isinstance(feedback_model, Counter) and total > 0:
                p_rel = feedback_model.get(term, 0.0) / total
            else:
                p_rel = feedback_model.get(term, 0.0)
        
    return beta * p_q + (1 - beta) * p_rel


def _calculate_rank_score(rank: int, v: int = 60) -> float:
    """Calculate rank-to-score transformation: 1 / (v + rank)."""
    return 1.0 / (v + rank)


def _normalize_query_model(query_model: dict[str, float] | Counter[str]) -> dict[str, float]:
    """Normalize a query model (counts or weights) to a probability distribution summing to 1."""
    if isinstance(query_model, Counter):
        total = sum(query_model.values())
        return {t: c / total for t, c in query_model.items()} if total > 0 else {}
    # dict case
    total = sum(query_model.values())
    if total <= 0:
        return {}
    return {t: (p / total) for t, p in query_model.items() if p > 0}


def _neg_kl_divergence_score(
    query_model: dict[str, float] | Counter[str],
    doc_counts: Counter[str],
    doc_len: int,
    mu: int,
    p_collection: dict[str, float],
    eps: float = 1e-12,
) -> float:
    """Compute true -KL(P_q || P_d) = sum_t Pq(t) * (log Pd(t) - log Pq(t)).
    
    Higher is better.
    
    This is the TRUE negative KL divergence, not just negative cross-entropy.
    The difference matters when comparing across different query model variants
    (e.g., MLE vs ITF) because it includes the query entropy term.
    
    Pd(t) uses Dirichlet smoothing:
      Pd(t) = (tf(t,d) + mu * Pc(t)) / (|d| + mu)
    """
    q_probs = _normalize_query_model(query_model)

    score = 0.0
    denom = (doc_len + mu) if (doc_len + mu) > 0 else 1.0

    for t, p_q in q_probs.items():
        if p_q <= 0:
            continue

        tf = doc_counts.get(t, 0)
        p_c = max(p_collection.get(t, 0.0), eps)
        p_d = (tf + mu * p_c) / denom
        p_d = max(p_d, eps)

        # -KL = sum Pq * (log Pd - log Pq)
        score += p_q * (math.log(p_d) - math.log(max(p_q, eps)))

    return score


def _neg_cross_entropy_score(
    query_model: dict[str, float] | Counter[str],
    doc_counts: Counter[str],
    doc_len: int,
    mu: int,
    p_collection: dict[str, float]
) -> float:
    """Calculate negative cross-entropy score (Higher is better).
    
    Score = sum_{t in q} P(t|q) * log(P(t|d))
    
    This is -H(P_q, P_d). Use _neg_kl_divergence_score for true -KL when
    comparing across different query model variants.
    
    Higher score = document model P_d is closer to query model P_q.
    """
    q_probs = _normalize_query_model(query_model)

    score = 0.0
    denom = doc_len + mu if (doc_len + mu) > 0 else 1.0
    
    for t, p_q in q_probs.items():
        if p_q <= 0: continue
        
        tf = doc_counts.get(t, 0)
        p_c = max(p_collection.get(t, 0), 1e-9)
        p_d = (tf + mu * p_c) / denom
        
        if p_d > 0:
            score += p_q * math.log(p_d)
            
    return score


def _calculate_bm25_term_score(
    term_freq: int,
    doc_len: int,
    avg_doc_len: float,
    df: int,
    total_docs: int,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """Calculate BM25 score for a single term."""
    if term_freq <= 0 or df <= 0:
        return 0.0
    
    if avg_doc_len <= 0:
        return 0.0
    
    # Term frequency component
    tf_component = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    
    # Inverse document frequency component
    # Use Lucene/Pyserini formula to avoid negative IDF
    idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
    
    return tf_component * idf




def _calculate_jm_score(
    term_freq: int,
    doc_len: int,
    collection_prob: float,
    lambda_param: float
) -> float:
    """Calculate Jelinek-Mercer smoothed log-probability for a single term.
    
    If collection_prob is zero/negative, clamps to epsilon to avoid returning 0
    (which is mathematically incorrect for "term impossible under collection").
    """
    # Clamp collection_prob to epsilon instead of returning 0
    p_c = max(collection_prob, 1e-9)
    
    doc_prob = term_freq / doc_len if doc_len > 0 else 0.0
    smoothed_prob = (1 - lambda_param) * doc_prob + lambda_param * p_c  # Use clamped p_c
    # Standard QL term score is log(P(t|d))
    return math.log(smoothed_prob) if smoothed_prob > 0 else -100.0


def _calculate_two_stage_score(
    term_freq: int,
    doc_len: int,
    collection_prob: float,
    mu: int = 1000,
    jm_lambda: float = 0.1
) -> float:
    """Calculate Two-Stage smoothed log-probability (Zhai & Lafferty '01).
    
    Two-Stage smoothing applies Dirichlet smoothing first, then JM interpolation:
    P(t|d) = (1-)  [(tf + Pc) / (|d| + )] +   Pc
    
    Args:
        term_freq: Term frequency in document
        doc_len: Document length
        collection_prob: P(t|C) collection probability
        mu: Dirichlet smoothing parameter (Stage 1)
        jm_lambda: JM interpolation parameter (Stage 2)
    """
    p_c = max(collection_prob, 1e-12)
    
    # Stage 1: Dirichlet smoothing
    p_dirichlet = (term_freq + mu * p_c) / (doc_len + mu) if (doc_len + mu) > 0 else p_c
    
    # Stage 2: JM interpolation
    p_two_stage = (1 - jm_lambda) * p_dirichlet + jm_lambda * p_c
    
    return math.log(max(p_two_stage, 1e-12))


def _build_weighted_feedback_model(
    feedback_docs: list[str],
    index_reader=None
) -> dict[str, float]:
    """Build weighted MLE feedback model P(t|R_rel) from feedback documents.
    
    This is the proper implementation for ITF per Sheetrit et al.:
    P(t|R) = count(t, R) / _t' count(t', R)
    
    Args:
        feedback_docs: List of document text strings
        index_reader: Optional Pyserini index reader for consistent tokenization
        
    Returns:
        Dictionary of term -> probability (MLE)
    """
    term_counts: Counter = Counter()
    for doc_text in feedback_docs:
        tokens = _tokenize(doc_text, index_reader=index_reader)
        term_counts.update(tokens)
    
    total = sum(term_counts.values())
    if total <= 0:
        return {}
    
    return {t: c / total for t, c in term_counts.items()}


# ===== ITF SCORING FUNCTIONS (Sheetrit et al.) =====

def _titf_kl_score(
    q_term_model: dict[str, float],
    doc_counts: Counter[str],
    doc_len: int,
    p_collection: dict[str, float],
    mu: int = 1000
) -> float:
    """T-ITF: Term-based KL-divergence score.
    
    Uses only term (token) features for document scoring.
    Score = sum_t P(t|q_terms) * log P(t|d)
    """
    if doc_len <= 0:
        return -1e9
    
    score = 0.0
    for term, q_prob in q_term_model.items():
        if q_prob <= 0:
            continue
        tf = doc_counts.get(term, 0)
        p_c = max(p_collection.get(term, 1e-12), 1e-12)
        # Dirichlet smoothed P(t|d)
        p_d = (tf + mu * p_c) / (doc_len + mu)
        score += q_prob * math.log(max(p_d, 1e-12))
    return score


def _eitf_kl_score(
    q_entity_model: dict[str, float],
    doc_entity_counts: Counter[str],
    doc_entity_total: float,
    p_collection_entity: dict[str, float],
    mu: int = 100
) -> float:
    """E-ITF: Entity-based KL-divergence score.
    
    Uses only entity (phrase) features for document scoring.
    Score = sum_e P(e|q_entities) * log P(e|d)
    """
    if doc_entity_total <= 0:
        return -1e9
    
    score = 0.0
    for entity, q_prob in q_entity_model.items():
        if q_prob <= 0:
            continue
        ef = doc_entity_counts.get(entity, 0)
        p_c = max(p_collection_entity.get(entity, 1e-12), 1e-12)
        # Dirichlet smoothed P(e|d)
        p_d = (ef + mu * p_c) / (doc_entity_total + mu)
        score += q_prob * math.log(max(p_d, 1e-12))
    return score


def _fitf_score(titf_score: float, eitf_score: float, alpha: float = 0.5) -> float:
    """F-ITF: Linear fusion of T-ITF and E-ITF (Sheetrit et al.).
    
    Final =  * T-ITF + (1-) * E-ITF
    
    Args:
        titf_score: Term-based ITF KL-divergence score
        eitf_score: Entity-based ITF KL-divergence score
        alpha: Weight for term component (default 0.5)
    """
    return alpha * titf_score + (1 - alpha) * eitf_score


# ===== ATOMIC ENTITY BOOST FUNCTIONS =====

def _phrase_in_query(phrase: str, query_lower: str) -> bool:
    """Check if phrase appears in query with word boundaries (not as substring).
    
    Uses regex word boundaries to avoid matching 'car' in 'scar'.
    """
    if not phrase:
        return False
    pattern = r'\b' + re.escape(phrase) + r'\b'
    return bool(re.search(pattern, query_lower))


def _calculate_entity_boost_score(phrases: set[str], query_lower: str, weight: float) -> float:
    """Calculate simple entity boost score based on phrase matches."""
    if weight <= 0:
        return 0.0
    
    match_count = sum(1 for p in phrases if p and _phrase_in_query(p, query_lower))
    return weight * match_count


def _calculate_entity_df_boost_score(
    phrases: set[str], 
    query_lower: str, 
    phrase_df: Counter[str], 
    weight: float
) -> float:
    """Calculate entity DF (Document Frequency) boost score."""
    if weight <= 0:
        return 0.0
    
    bonus = 0.0
    for p in phrases:
        if _phrase_in_query(p, query_lower):
            df = phrase_df.get(p, 0)
            if df > 1:
                bonus += weight * (df - 1)
    return bonus


def _calculate_entity_idf_boost_score(
    phrases: set[str], 
    query_lower: str, 
    phrase_idf: dict[str, float], 
    weight: float
) -> float:
    """Calculate entity IDF (Inverse Document Frequency) boost score."""
    if weight <= 0:
        return 0.0
    
    bonus = 0.0
    for p in phrases:
        if _phrase_in_query(p, query_lower):
            idf = phrase_idf.get(p, 0.0)
            bonus += weight * idf
    return bonus


def _calculate_booster_filter_score(
    phrases: set[str], 
    query_lower: str, 
    booster_phrases: set[str], 
    enabled: bool
) -> float:
    """Calculate booster filter score (penalties for poison pills, boosts for consensus)."""
    if not enabled:
        return 0.0
    
    bonus = 0.0
    for p in phrases:
        if _phrase_in_query(p, query_lower):
            if p not in booster_phrases:
                bonus -= 0.05  # Penalty for non-consensus phrases
            else:
                bonus += 0.1   # Boost for consensus phrases
    return bonus


# ===== ATOMIC MMR FUNCTIONS =====

def _calculate_mmr_similarity_score(
    candidate_vec: dict[str, float], 
    selected_vecs: list[dict[str, float]], 
    norms: list[float], 
    candidate_norm: float
) -> float:
    """Calculate maximum similarity between a candidate and already selected items."""
    if not selected_vecs:
        return 0.0
    
    max_sim = 0.0
    for sel_vec, sel_norm in zip(selected_vecs, norms):
        sim = _cosine_sim(candidate_vec, sel_vec, candidate_norm, sel_norm)
        max_sim = max(max_sim, sim)
    return max_sim


def _calculate_mmr_score(relevance: float, max_similarity: float, lambda_diversity: float) -> float:
    """Calculate MMR score combining relevance and diversity."""
    return lambda_diversity * relevance - (1 - lambda_diversity) * max_similarity


# ===== ATOMIC RM3 FUNCTIONS =====

def _build_feedback_term_counters(feedback_hits: list[Hit]) -> tuple[list[Counter], dict[str, int]]:
    """Build term counters and lengths for feedback documents."""
    feedback_term_counters: list[Counter] = []
    feedback_lengths: dict[str, int] = {}
    
    for hit in feedback_hits:
        contents = hit.contents or ""
        toks = _tokenize(contents)
        counter = Counter(toks)
        feedback_term_counters.append(counter)
        feedback_lengths[hit.docid] = len(toks)
    
    return feedback_term_counters, feedback_lengths


def _build_relevance_model(
    feedback_hits: list[Hit], 
    feedback_term_counters: list[Counter], 
    feedback_lengths: dict[str, int],
    q_counts: Counter,
    mu: int,
    p_collection: dict[str, float]
) -> dict[str, float]:
    """Build RM1 relevance model from feedback documents."""
    rm_probs: dict[str, float] = {}
    
    # 1. Compute log-likelihoods P(q|d) for all feedback docs
    log_weights = []
    for i, hit in enumerate(feedback_hits):
        doc_counts = feedback_term_counters[i]
        doc_len = feedback_lengths.get(hit.docid, 0)
        ll = _dirichlet_query_loglikelihood(q_counts, doc_counts, doc_len, mu, p_collection)
        log_weights.append(ll)
    
    # 2. Numerical stability: subtract max before exp (Log-Sum-Exp)
    max_ll = max(log_weights) if log_weights else 0.0
    weights = [math.exp(ll - max_ll) for ll in log_weights]
    sum_weights = sum(weights)
    
    # Normalize document weights P(d|q)
    doc_weights = [w / sum_weights for w in weights] if sum_weights > 0 else [0.0] * len(weights)
    
    # 3. Accumulate term probabilities: sum_d P(d|q) * P(t|d)
    # Using Dirichlet smoothed P(t|d) instead of MLE
    for i, weight in enumerate(doc_weights):
        if weight <= 0: continue
            
        doc_counts = feedback_term_counters[i]
        doc_len = feedback_lengths.get(feedback_hits[i].docid, 0)
        denom = doc_len + mu if (doc_len + mu) > 0 else 1.0
        
        for term, count in doc_counts.items():
            if term not in rm_probs:
                rm_probs[term] = 0.0
            
            p_c = p_collection.get(term, 1e-9)
            p_t_d = (count + mu * p_c) / denom
            rm_probs[term] += weight * p_t_d
            
    # 4. Normalize resulting distribution
    total_prob = sum(rm_probs.values())
    if total_prob > 0:
        for term in rm_probs:
            rm_probs[term] /= total_prob
            
    return rm_probs


def _expand_query_with_rm3(
    q_counts: Counter, 
    rm_probs: dict[str, float], 
    rm3_fb_terms: int, 
    rm3_original_query_weight: float
) -> Counter:
    """Create expanded query by interpolating original query with relevance model."""
    expanded_query_counts = Counter()
    rm_weight = 1.0 - rm3_original_query_weight
    
    # Normalize original query counts to probabilities for consistent interpolation
    q_total = sum(q_counts.values())
    q_probs = {t: c / q_total for t, c in q_counts.items()} if q_total > 0 else {}
    
    # Original query terms
    for term, prob in q_probs.items():
        expanded_query_counts[term] += rm3_original_query_weight * prob
    
    # Relevance model terms (top rm3_fb_terms)
    sorted_rm_terms = sorted(rm_probs.items(), key=lambda x: x[1], reverse=True)[:rm3_fb_terms]
    
    for term, prob in sorted_rm_terms:
        expanded_query_counts[term] += rm_weight * prob
    
    return expanded_query_counts


def _filter_poison_pills(
    expansion_terms: dict[str, float],
    feedback_hits: list,
    min_docs: int = 2,
    index_reader=None
) -> dict[str, float]:
    """Filter expansion terms that appear in fewer than min_docs feedback documents.
    
    This is the CORRECT way to handle Poison Pills (Sheetrit et al.):
    Filter terms during EXPANSION, not penalize documents during SCORING.
    
    A term with no "consensus" (appears in <min_docs) is likely noise.
    """
    if not feedback_hits or not expansion_terms:
        return expansion_terms
    
    term_doc_counts: Counter = Counter()
    for hit in feedback_hits:
        contents = hit.contents if hasattr(hit, 'contents') else ""
        terms = set(_tokenize(contents or "", index_reader=index_reader))
        term_doc_counts.update(terms)
    
    return {t: w for t, w in expansion_terms.items() 
            if term_doc_counts.get(t, 0) >= min_docs}


def _aggregate_max_p(passage_hits: list) -> list:
    """MaxP aggregation: Group passages by docid, keep highest scoring per doc.
    
    This converts passage-level ranking to document-level ranking,
    useful when we want unique documents ranked by their best passage.
    """
    if not passage_hits:
        return passage_hits
    
    best_by_doc = {}
    for hit in passage_hits:
        docid = hit.docid.split("_chunk_")[0]  # Handle chunked docids
        if docid not in best_by_doc or hit.score > best_by_doc[docid].score:
            best_by_doc[docid] = hit
    
    return sorted(best_by_doc.values(), key=lambda h: h.score, reverse=True)


def _apply_rm3_query_expansion(
    initial_hits: list,
    query: str,
    searcher,
    rm3_fb_terms: int,
    rm3_fb_docs: int,
    rm3_original_query_weight: float,
    mu: int,
    top_k: int,
    p_collection: dict[str, float],
    index_reader=None
) -> list:
    """Apply RM3 as query expansion with second retrieval (per Lavrenko & Croft).
    
    FIXED: This performs a NEW search with the expanded query instead of just
    reranking the initial hits. This improves recall by finding documents
    containing expansion terms that weren't in the original query.
    
    Args:
        initial_hits: Initial retrieval results (used for feedback)
        query: Original query string
        searcher: Pyserini searcher for second retrieval
        rm3_fb_terms: Number of expansion terms to add
        rm3_fb_docs: Number of feedback documents to use
        rm3_original_query_weight: Weight for original query (1-weight for expansion)
        mu: Dirichlet smoothing parameter
        top_k: Number of results to return
        p_collection: Collection probability estimates
        index_reader: Optional index reader for tokenization
        
    Returns:
        New list of hits from expanded query retrieval
    """
    if not initial_hits:
        return initial_hits
    
    # Take top feedback docs
    feedback_hits = initial_hits[:rm3_fb_docs]
    
    # Build term counters for feedback docs
    feedback_term_counters, feedback_lengths = _build_feedback_term_counters(feedback_hits)
    
    # Tokenize query
    query_tokens = _tokenize(query, index_reader=index_reader)
    q_counts = Counter(query_tokens)
    
    # Build relevance model (RM1)
    rm_probs = _build_relevance_model(
        feedback_hits, feedback_term_counters, feedback_lengths,
        q_counts, mu, p_collection
    )
    
    # Expand query (RM3 = interpolation of original query with RM1)
    expanded_query = _expand_query_with_rm3(
        q_counts, rm_probs, rm3_fb_terms, rm3_original_query_weight
    )
    
    # Build weighted query string for Pyserini
    # Format: "term1^weight1 term2^weight2 ..."
    total_weight = sum(expanded_query.values())
    if total_weight <= 0:
        return initial_hits
    
    query_parts = []
    for term, weight in sorted(expanded_query.items(), key=lambda x: x[1], reverse=True):
        if weight > 0:
            # Normalize and boost
            normalized_weight = weight / total_weight
            query_parts.append(f"{term}^{normalized_weight:.4f}")
    
    expanded_query_str = " ".join(query_parts[:rm3_fb_terms + len(q_counts)])
    
    # Perform second retrieval with expanded query
    raw_hits = searcher.search(expanded_query_str, top_k * 2)
    
    # Convert to Hit objects
    from raglab.retrieval import Hit
    result_hits = []
    for hit in raw_hits[:top_k]:
        contents = hit.raw.get("contents", "") if hit.raw else ""
        result_hits.append(Hit(
            docid=hit.docid,
            score=hit.score,
            contents=contents,
            raw=hit.raw,
            rm3_score=hit.score
        ))
    
    return result_hits

def _segment_passages(text: str, window_size: int = 300, stride: int = 150) -> list[tuple[int, int, str]]:
    """Segment text into sliding-window passages.
    
    Args:
        text: Source text to segment
        window_size: Target passage length in words (approx)
        stride: Overlap stride in words (approx)
    
    Returns:
        List of (start_char, end_char, passage_text) tuples
    """
    if not text or not text.strip():
        return []
    
    # Issue 14 Fix: Use regex to find actual word spans for correct offsets.
    # text.split() loses whitespace info. 
    # We want word-based windows but character-based offsets.
    
    # Find all words with their (start, end) character offsets
    word_matches = list(re.finditer(r'\S+', text))
    if not word_matches:
        return []
    
    passages = []
    start_idx = 0
    num_matches = len(word_matches)
    
    while start_idx < num_matches:
        # Define window in terms of words
        end_idx = min(start_idx + window_size, num_matches)
        
        # Get start/end offsets from the first and last word in the window
        start_char = word_matches[start_idx].start()
        end_char = word_matches[end_idx - 1].end()
        
        # Extract the exact substring from original text (preserves original spacing)
        passage_text = text[start_char:end_char]
        
        passages.append((start_char, end_char, passage_text))
        
        if end_idx >= num_matches:
            break
            
        start_idx += stride
    
    return passages


# Global cache for analyzer to avoid recreation overhead
_GLOBAL_ANALYZER = None

def _get_start_analyzer(index_reader=None):
    """Get or create a Pyserini Analyzer."""
    global _GLOBAL_ANALYZER
    if _GLOBAL_ANALYZER:
        return _GLOBAL_ANALYZER
        
    if HAS_PYSERINI_ANALYSIS:
        try:
            # If we had a way to get analyzer from index_reader, we should use it.
            # But SimpleSearcher doesn't easily expose the Analyzer object itself in python bindings versions < 0.22 sometimes.
            # Ideally: return Analyzer(index_reader.object.getAnalyzer())
            # For now, default to standard Lucene analyzer which Pyserini uses by default.
            _GLOBAL_ANALYZER = Analyzer(get_lucene_analyzer())
            return _GLOBAL_ANALYZER
        except Exception as e:
            logger.warning(f"Failed to initialize Pyserini Analyzer: {e}")
            return None
    return None


def _get_global_collection_stats(index_reader) -> tuple[int, float]:
    """Fetch rigorous global stats: Total Docs (N) and Avg Doc Len (avgdl)."""
    if not index_reader:
        raise ValueError("Strict Math Mode: index_reader is required for BM25/LM statistics.")
    
    stats = index_reader.stats()
    N = stats.get('documents', 0)
    total_terms = stats.get('total_terms', 0)
    
    if N <= 0:
        raise ValueError("Index is empty or stats unavailable.")
        
    avgdl = total_terms / N if N > 0 else 0.0
    return N, avgdl


def _get_global_term_stats(term: str, index_reader) -> tuple[int, int]:
    """Fetch rigorous global stats for a term: DF and CF (Collection Freq)."""
    if not index_reader:
        raise ValueError("Strict Math Mode: index_reader is required for BM25/LM statistics.")
    try:
        # Input 'term' is assumed to be tokenized/stemmed matching the index.
        # Pass analyzer=None to treat it as a raw term lookup.
        df, cf = index_reader.get_term_counts(term, analyzer=None)
        return df, cf
    except Exception:
        return 0, 0


def _get_pyserini_collection_probs(query_terms: list[str], index_reader) -> dict[str, float]:
    """Get P(t|C) using Pyserini's global statistics strictly.
    
    P(t|C) = cf(t) / TotalTermsInCollection
    """
    if not index_reader:
        raise ValueError("Strict Math Mode: index_reader is required for BM25/LM statistics.")

    try:
        stats = index_reader.stats()
        total_terms = stats.get('total_terms', 0)
        
        if total_terms <= 0:
            logger.warning(" Index stats returned 0 total_terms. Is the index valid?")
            return {}
            
        probs = {}
        for t in query_terms:
            df, cf = _get_global_term_stats(t, index_reader)
            if cf > 0:
                probs[t] = cf / total_terms
            else:
                # OOV Term in global context.
                # Strictly 0, but usually we want a tiny probability for smoothing stability (epsilon).
                # Zhai & Lafferty: P(t|C) approx 1/CollectionSize
                probs[t] = 1.0 / total_terms
            
        return probs
    except Exception as e:
        logger.warning(f"Failed to get collection probs: {e}")
        return {}



def _tokenize(text: str, index_reader=None) -> list[str]:
    """Tokenize text using Pyserini analyzer if available, else regex.
    
    FIXED: Uses regex instead of split() to properly handle punctuation.
    "apple." and "apple" are now the same token.
    """
    analyzer = _get_start_analyzer(index_reader)
    if analyzer:
        try:
            # analyzer.analyze(text) returns list of strings (stemmed)
            return analyzer.analyze(text)
        except Exception:
            # Fallback to regex
            pass
    
    # Regex fallback: extract word characters only (handles "apple." -> "apple")
    return re.findall(r'\b\w+\b', (text or "").lower())


_CAP_PHRASE_RE = re.compile(r"(?:[A-Z][A-Za-z0-9_-]*)(?:\s+(?:[A-Z][A-Za-z0-9_-]*)){0,3}")

# Stopwords to filter from entity extraction (sentence starters, common words)
_ENTITY_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "if", "it", "its", "this", "that", "these",
    "those", "he", "she", "they", "we", "you", "i", "my", "your", "his",
    "her", "their", "our", "what", "which", "who", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "but", "and", "or", "yet", "because", "although"
})


def _extract_capitalized_phrases(text: str, max_words: int = 4) -> list[str]:
    """Lightweight extractor for capitalized multi-word phrases (entity-like).

    Keeps 1-4 token spans starting with uppercase, intended to approximate entities
    without external NER. Returns raw surface strings.
    
    FIXED: Filters stopwords to avoid "The", "A", "If" etc. as entities.
    """
    if not text:
        return []
    phrases: list[str] = []
    for m in _CAP_PHRASE_RE.finditer(text):
        phrase = m.group(0).strip()
        if phrase and len(phrase.split()) <= max_words:
            # Filter single-word stopwords
            if phrase.lower() not in _ENTITY_STOPWORDS:
                phrases.append(phrase)
    return phrases


# ===== ATOMIC TF-IDF FUNCTIONS =====

def _calculate_tf(term_freq: int, doc_len: int) -> float:
    """Calculate term frequency component."""
    return term_freq / doc_len if doc_len > 0 else 0.0


def _calculate_idf(df: int, total_docs: int) -> float:
    """Calculate inverse document frequency."""
    if df <= 0:
        return 0.0
    return math.log((total_docs + 1) / (df + 1)) + 1.0


def _calculate_tfidf_vector(tokenized_doc: list[str], df: Counter[str], total_docs: int) -> tuple[dict[str, float], float]:
    """Calculate TF-IDF vector for a single document."""
    tf = Counter(tokenized_doc)
    vec = {}
    for term, freq in tf.items():
        tf_score = _calculate_tf(freq, len(tokenized_doc))
        idf_score = _calculate_idf(df.get(term, 0), total_docs)
        vec[term] = tf_score * idf_score
    
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1e-9
    return vec, norm


def _tfidf_vectors(passages: list[str], global_df: dict[str, float] = None, total_docs: int = 0, index_reader=None) -> tuple[list[dict[str, float]], list[float]]:
    """Compute lightweight TF-IDF vectors for passages.
    
    Issue 30: Use global DF if provided.
    Refactor: Use index_reader for tokenization if provided.
    """
    tokenized = [_tokenize(p, index_reader=index_reader) for p in passages]
    
    local_df: Counter[str] = Counter()
    for toks in tokenized:
        local_df.update(set(toks))
        
    vectors: list[dict[str, float]] = []
    norms: list[float] = []
    
    N = total_docs if total_docs > 0 else len(passages)
    
    for toks in tokenized:
        vec = {}
        counts = Counter(toks)
        for term, freq in counts.items():
            tf = _calculate_tf(freq, len(toks))
            
            # Get IDF
            if global_df and term in global_df:
                # Assuming global_df maps term -> doc_count
                doc_freq = global_df[term]
                idf = _calculate_idf(doc_freq, N)
            else:
                idf = _calculate_idf(local_df[term], len(passages))
                
            vec[term] = tf * idf
            
        vectors.append(vec)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1e-9
        norms.append(norm)
        
    return vectors, norms


def _cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float], norm_a: float, norm_b: float) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = vec_a.keys() & vec_b.keys()
    if not shared:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    denom = norm_a * norm_b
    return dot / denom if denom else 0.0


def _zscore(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
    std = math.sqrt(var) or 1e-9
    return {k: (v - mean) / std for k, v in scores.items()}


# ===== ATOMIC COLLECTION PROBABILITY FUNCTIONS =====

def _calculate_collection_probability(term: str, doc_term_counters: list[Counter]) -> float:
    """Calculate collection probability for a term across all documents."""
    total_term_count = sum(counter.get(term, 0) for counter in doc_term_counters)
    total_tokens = sum(sum(counter.values()) for counter in doc_term_counters)
    return total_term_count / total_tokens if total_tokens > 0 else 0.0





# ===== ATOMIC QUERY LIKELIHOOD FUNCTIONS =====

def _calculate_dirichlet_term_score(
    term_freq: int, 
    doc_len: int, 
    collection_prob: float, 
    mu: int
) -> float:
    """Calculate Dirichlet-smoothed log-likelihood for a single term.
    
    If collection_prob is zero/negative, clamps to epsilon (1e-9) to ensure
    valid smoothing even for OOV terms.
    """
    # Clamp collection_prob to epsilon instead of returning 0 (Issue 9 fix)
    p_c = max(collection_prob, 1e-9)
    
    denom = doc_len + mu if (doc_len + mu) > 0 else 1
    smoothed_prob = (term_freq + mu * p_c) / denom
    return math.log(max(smoothed_prob, 1e-12))


def _dirichlet_query_loglikelihood(
    query_counts: Counter,
    doc_counts: Counter,
    doc_len: int,
    mu: int,
    p_collection: dict[str, float],
) -> float:
    """Compute log P(q|d) under a Dirichlet-smoothed unigram LM.

    log P(q|d) = sum_t q_tf(t) * log( (tf(t,d) + mu * P(t|C)) / (|d| + mu) )
    """
    score = 0.0
    for t, q_tf in query_counts.items():
        p_c = p_collection.get(t, 1e-9)
        term_score = _calculate_dirichlet_term_score(doc_counts.get(t, 0), doc_len, p_c, mu)
        score += q_tf * term_score
    return score


def _mmr_rerank(
    candidates: list[tuple[float, str, str, float, float, float, float, float, float]],
    *,
    top_k: int,
    lambda_diversity: float = 0.7,
    max_candidates: int | None = None,
) -> list[tuple[float, str, str, float, float, float, float, float, float]]:
    """Apply Maximal Marginal Relevance (MMR) to diversify passages."""
    if not candidates:
        return []

    # Limit pool size to keep computation light
    if max_candidates is None:
        max_candidates = min(len(candidates), top_k * 4)
    pool = candidates[:max_candidates]

    # Precompute TF-IDF vectors for cosine similarity
    passages = [p[2] for p in pool]
    vecs, norms = _tfidf_vectors(passages)

    # Start with the highest relevance item
    selected: list[int] = []
    remaining = list(range(len(pool)))
    remaining.sort(key=lambda i: pool[i][0], reverse=True)  # Sort by total_score

    if not remaining:
        return []
    selected.append(remaining.pop(0))

    while len(selected) < top_k and remaining:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            rel = pool[idx][0]  # total_score
            max_sim = _calculate_mmr_similarity_score(vecs[idx], [vecs[i] for i in selected], [norms[i] for i in selected], norms[idx])
            mmr_score = _calculate_mmr_score(rel, max_sim, lambda_diversity)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [pool[i] for i in selected]


def _apply_rm3_reranking(hits: list[Hit], query: str, searcher, rm3_fb_terms: int = 20, rm3_fb_docs: int = 10, rm3_original_query_weight: float = 0.5, mu: int = 1000, index_reader=None) -> list[Hit]:
    """Apply RM3 reranking as post-processing to any retrieval results.
    
    RM3 expands the query using terms from top feedback documents, then reranks.
    """
    if not hits or len(hits) < rm3_fb_docs:
        return hits
    
    # Use top rm3_fb_docs documents for feedback
    feedback_hits = hits[:rm3_fb_docs]
    
    # Build term counters for feedback documents
    # Refactor: use index_reader for tokenization
    query_terms = _tokenize(query, index_reader=index_reader)
    q_counts = Counter(query_terms)
    feedback_term_counters, feedback_lengths = _build_feedback_term_counters(feedback_hits)
    
    # Estimate collection probabilities
    # Issue 29 Fix: Use global stats if available. If not, use the FULL retrieved set (hits) 
    # to estimate P(t|C), not just the top feedback documents. Using only feedback docs 
    # for background model causes "self-clustering" / topic drift.
    all_feedback_terms = set()
    for counter in feedback_term_counters:
        all_feedback_terms.update(counter.keys())
        
    p_collection = {}
    if index_reader:
        p_collection = _get_pyserini_collection_probs(list(all_feedback_terms), index_reader)
    else:
        logger.warning("Skipping RM3: index_reader required for strict collection statistics.")
        return hits
    
    # Build relevance model (RM1)
    rm_probs = _build_relevance_model(feedback_hits, feedback_term_counters, feedback_lengths, q_counts, mu, p_collection)
    
    # Create expanded query
    expanded_query_counts = _expand_query_with_rm3(q_counts, rm_probs, rm3_fb_terms, rm3_original_query_weight)
    
    # Rerank all hits using expanded query
    reranked_hits = []
    all_doc_term_counters: list[Counter] = []
    all_doc_lengths: dict[str, int] = {}
    
    for hit in hits:
        contents = hit.contents or ""
        toks = _tokenize(contents)
        c = Counter(toks)
        all_doc_term_counters.append(c)
        all_doc_lengths[hit.docid] = len(toks)
    
    p_collection_all = {}
    if index_reader:
        p_collection_all = _get_pyserini_collection_probs(list(expanded_query_counts.keys()), index_reader)
    else:
        # Should have returned earlier
        return hits
    
    for i, hit in enumerate(hits):
        doc_counts = all_doc_term_counters[i]
        doc_len = all_doc_lengths.get(hit.docid, 0)
        rm3_score = _dirichlet_query_loglikelihood(expanded_query_counts, doc_counts, doc_len, mu, p_collection_all)
        hit.rm3_score = rm3_score
        # NOTE: We overwrite hit.score here because RM3 is typically a re-ranking step 
        # that defines the final order. ideally we should keep original score too.
        # hit.original_score = hit.score # If we had an original_score field
        hit.score = rm3_score  # Update the hit score to the RM3 score for display
        reranked_hits.append((rm3_score, hit))
    
    # Sort by RM3 score (higher is better) and return hits
    reranked_hits.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in reranked_hits]


def _apply_kl_reranking(hits: list[Hit], query: str, searcher, mu: int, index_reader=None) -> list[Hit]:
    """Apply KL-divergence reranking as post-processing to any retrieval results."""
    if not hits:
        return hits
    
    # Build term counters for docs and query
    query_terms = _tokenize(query, index_reader=index_reader)
    q_counts = Counter(query_terms)
    doc_term_counters: list[Counter] = []
    doc_lengths: dict[str, int] = {}
    
    for hit in hits:
        contents = hit.contents or ""
        toks = _tokenize(contents, index_reader=index_reader)
        c = Counter(toks)
        doc_term_counters.append(c)
        doc_lengths[hit.docid] = len(toks)

    if not index_reader:
        raise ValueError("Strict Math Mode: index_reader is required for BM25/LM statistics.")
        
    p_collection = _get_pyserini_collection_probs(query_terms, index_reader)

    # Compute true -KL divergence scores and rerank
    reranked_hits = []
    for i, hit in enumerate(hits):
        c = doc_term_counters[i]
        dlen = doc_lengths.get(hit.docid, 0)
        kl_score = _neg_kl_divergence_score(q_counts, c, dlen, mu, p_collection)
        hit.score = kl_score
        hit.kl_divergence_score = kl_score
        reranked_hits.append((kl_score, hit))
    
    # Sort by -KL score (higher is better) and return hits
    reranked_hits.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in reranked_hits]


# ===== STLM (Structured Term Language Model) =====
# Lecture-accurate implementation: pseudo-counts mixing terms and entities

def _stlm_pseudo_counts(
    text: str,
    stlm_lambda: float,
    index_reader=None,
    entity_confidence_fn=None,
) -> tuple[Counter, float]:
    """Compute STLM pseudo-counts over T = terms  entities.
    
    Args:
        text: Document or query text
        stlm_lambda: Weight for terms (1-stlm_lambda for entities)
        index_reader: Optional Pyserini index reader for consistent tokenization
        entity_confidence_fn: Optional function returning confidence for a mention
        
    Returns:
        pc: Counter[token] pseudo-counts (keys prefixed with TERM:: or ENT::)
        pl: pseudo-length = sum of pseudo-counts
    """
    # Default confidence heuristic based on phrase length (Sheetrit et al. paper):
    # Longer phrases are more likely to be specific entities
    # 3+ words  =1.0, 2 words  =0.7, 1 word  =0.4
    if entity_confidence_fn is None:
        def _length_based_confidence(mention: str) -> float:
            words = mention.split()
            if len(words) >= 3:
                return 1.0
            elif len(words) == 2:
                return 0.7
            else:
                return 0.4
        entity_confidence_fn = _length_based_confidence
    
    pc: Counter = Counter()
    
    # Term pseudo-counts: pc(t, x) =   tf(t, x)
    tokens = _tokenize(text, index_reader=index_reader)
    tf = Counter(tokens)
    for t, c in tf.items():
        pc[f"TERM::{t}"] += stlm_lambda * c
    
    # Entity pseudo-counts: pc(t, x) = (1 - )   (m)
    # Using capitalized phrase extractor as weak entity linker
    # (m) = entity confidence score based on length heuristic
    mentions = _extract_capitalized_phrases(text)
    for m in mentions:
        ent = m.strip().lower()
        if not ent:
            continue
        confidence = float(entity_confidence_fn(m))
        pc[f"ENT::{ent}"] += (1.0 - stlm_lambda) * confidence
    
    pl = sum(pc.values())
    return pc, max(pl, 1e-9)  # Avoid division by zero





def _stlm_dirichlet_score(
    q_pc: Counter,
    doc_pc: Counter,
    doc_pl: float,
    p_coll: dict[str, float],
    mu: float = 1000.0
) -> float:
    """Compute STLM score using Dirichlet smoothing over pseudo-counts.
    
    score(d, q) = _{t  T_q} pc(t, q)  log( (pc(t,d) + P(t|C)) / (pl(d) + ) )
    """
    score = 0.0
    denom = doc_pl + mu
    if denom <= 0:
        return -1e9
    
    for t, q_w in q_pc.items():
        if q_w <= 0:
            continue
        p_c = max(p_coll.get(t, 0.0), 1e-12)  # Epsilon fallback
        num = doc_pc.get(t, 0.0) + mu * p_c
        if num <= 0:
            continue
        score += q_w * math.log(num / denom)
    return score


def _apply_stlm_reranking(
    hits: list,
    query: str,
    stlm_lambda: float = 0.8,
    mu: int = 1000,
    index_reader=None
) -> list:
    """Apply STLM reranking to a list of hits.
    
    STLM (Structured Term Language Model) uses pseudo-counts that mix:
    - Terms: pc(t, x) =   tf(t, x)
    - Entities: pc(t, x) = (1 - )   (m)
    
    Args:
        hits: List of Hit objects with contents
        query: Original query string
        stlm_lambda: Weight for terms (0.8-0.9 recommended)
        mu: Dirichlet smoothing parameter
        index_reader: Optional Pyserini index reader
        
    Returns:
        Reranked list of hits with updated scores
    """
    if not hits:
        return hits
    
    # Compute query pseudo-counts
    q_pc, _ = _stlm_pseudo_counts(query, stlm_lambda=stlm_lambda, index_reader=index_reader)
    
    # Compute document pseudo-counts
    doc_pcs: list[tuple[Counter, float]] = []
    for hit in hits:
        text = hit.contents or ""
        pc, pl = _stlm_pseudo_counts(text, stlm_lambda=stlm_lambda, index_reader=index_reader)
        doc_pcs.append((pc, pl))
    
    # Estimate collection probabilities STRICTLY
    # Identify all TERMS and ENTITIES in the candidate pool
    raw_terms = set()
    entities = set()
    
    for pc, _ in doc_pcs:
        for k in pc.keys():
            if k.startswith("TERM::"):
                 raw_terms.add(k.replace("TERM::", ""))
            elif k.startswith("ENT::"):
                 entities.add(k.replace("ENT::", ""))
                 
    # 1. Get stats for direct terms
    term_probs = _get_pyserini_collection_probs(list(raw_terms), index_reader)
    
    # 2. Get stats for entity surface forms (Approximation via Geometric Mean)
    # We map Entity -> Tokens -> GeoMean(P(Tokens|C))
    entity_tokens = set()
    for ent in entities:
        entity_tokens.update(_tokenize(ent, index_reader))
        
    entity_token_probs = _get_pyserini_collection_probs(list(entity_tokens), index_reader)
    
    p_coll = {}
    
    # Fill TERM:: probs (STRICT GLOBAL ONLY)
    for t in raw_terms:
         p = term_probs.get(t, 0.0)
         if p > 0:
             p_coll[f"TERM::{t}"] = p
         else:
             p_coll[f"TERM::{t}"] = 1e-12 # Static prior for unknown terms

    # Fill ENT:: probs
    for ent in entities:
        toks = _tokenize(ent, index_reader)
        if not toks:
            p_coll[f"ENT::{ent}"] = 1e-12
        else:
            log_sum = 0.0
            for t in toks:
                p = entity_token_probs.get(t, 0.0) 
                if p <= 0: p = 1e-12
                log_sum += math.log(p)
            
            geo_mean = math.exp(log_sum / len(toks))
            p_coll[f"ENT::{ent}"] = geo_mean
    
    # Score and rerank
    scored_hits = []
    for hit, (pc, pl) in zip(hits, doc_pcs):
        stlm_score = _stlm_dirichlet_score(q_pc, pc, pl, p_coll, mu=float(mu))
        hit.score = stlm_score
        scored_hits.append((stlm_score, hit))
    
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in scored_hits]


@dataclass
class Hit:
    docid: str
    score: float
    contents: str | None
    raw: dict[str, Any] | None
    query_likelihood_score: float | None = None
    kl_divergence_score: float | None = None  # True -KL(Pq || Pd) (higher is better)
    rm3_score: float | None = None  # RM3 relevance score after query expansion
    jm_score: float | None = None
    tfidf_score: float | None = None
    bm25_score: float | None = None
    exact_match_count: int | None = None
    term_overlap_score: float | None = None
    titf_score: float | None = None
    fitf_score: float | None = None
    rrf_score: float | None = None
    entity_boost_score: float | None = None  # Flat entity boost score (match count * weight)
    entity_df_boost_score: float | None = None  # Entity DF boost score (based on document frequency)
    entity_idf_boost_score: float | None = None  # Entity IDF boost score (based on inverse document frequency)
    booster_filter_score: float | None = None  # Booster filter score (penalties for poison pills, boosts for consensus)
    mmr_score: float | None = None  # MMR diversification score
    fusion_score: float | None = None  # BM25 + LM fusion score
    passage_smoothing_score: float | None = None  # Hierarchical passage smoothing score
    entity_score: float | None = None  # Entity-based relevance score
    two_stage_score: float | None = None  # Dirichlet+JM interpolation score (not true two-stage)
    cosine_similarity_score: float | None = None  # TF-IDF Cosine similarity
    jaccard_score: float | None = None  # Jaccard similarity
    dice_score: float | None = None  # Dice coefficient
    
    # Ranks for various scores
    query_likelihood_rank: int | None = None
    kl_divergence_rank: int | None = None
    rm3_rank: int | None = None
    jm_rank: int | None = None
    tfidf_rank: int | None = None
    bm25_rank: int | None = None
    exact_match_rank: int | None = None
    term_overlap_rank: int | None = None
    titf_rank: int | None = None
    fitf_rank: int | None = None
    rrf_rank: int | None = None
    fusion_rank: int | None = None
    two_stage_rank: int | None = None
    cosine_similarity_rank: int | None = None
    jaccard_rank: int | None = None
    dice_rank: int | None = None
    entity_boost_rank: int | None = None
    entity_df_boost_rank: int | None = None
    entity_idf_boost_rank: int | None = None
    booster_filter_rank: int | None = None


@dataclass
class RetrievalMetadata:
    """Metadata about retrieval method and configuration."""
    method: str  # BM25, QLD, JM, etc.
    parameters: dict[str, Any] = field(default_factory=dict)  # Method-specific params
    query_original: str = ""  # Original query before modifications
    query_modified: str = ""  # Query after any modifications
    query_modifications: dict[str, Any] = field(default_factory=dict)  # Added terms, boosts, etc.
    index_name: str = ""  # Which index was used
    top_k: int = 10  # Number of results requested
    final_k: int | None = None  # Final cutoff applied to results
    method_label: str = ""  # Human-readable method label
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "parameters": self.parameters,
            "query_original": self.query_original,
            "query_modified": self.query_modified,
            "query_modifications": self.query_modifications,
            "index_name": self.index_name,
            "top_k": self.top_k,
            "final_k": self.final_k,
            "method_label": self.method_label,
        }


def _load_searcher(prebuilt_name: str | None = None, local_path: str | None = None):
    # Use LuceneSearcher (Pyserini 0.36) to avoid optional FAISS imports
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        from pyserini.search import SimpleSearcher as LuceneSearcher
    if local_path:
        s = LuceneSearcher(local_path)
    else:
        s = LuceneSearcher.from_prebuilt_index(prebuilt_name or "wikipedia-kilt-doc")
    logger = setup_logger()
    logger.info(f" Searcher ready  index='{prebuilt_name or local_path or 'wikipedia-kilt-doc'}'")
    return s


def _canonical_method(method: str) -> str:
    """Normalize user-provided method labels."""
    m = method.upper()
    if m in {"QLD", "DIRICHLET", "DIRICHLET LM", "DIRICHLET_LM"}:
        return "DIRICHLET"
    if m in {"JM", "JELINEK-MERCER", "JELINEK_MERCER"}:
        return "JM"
    if m in {"TWO_STAGE", "TWO-STAGE", "TWOSTAGE"}:
        return "TWO_STAGE"
    if m in {"KL", "KL_DIVERGENCE", "KL-DIVERGENCE"}:
        return "KL_DIVERGENCE"
    if m in {"RM3"}:
        return "RM3"
    if m in {"FUSION", "BM25_LM_FUSION", "FUSION_BM25_LM", "BM25+LM", "BM25_LM"}:
        return "FUSION_BM25_LM"
    if m in {"MMR_PASSAGES"}:
        return "MMR_PASSAGES"
    if m in {"T-ITF", "TITF", "TERM_ITF"}:
        return "T-ITF"
    if m in {"E-ITF", "EITF", "ENTITY_ITF"}:
        return "E-ITF"
    if m in {"F-ITF", "FITF", "FUSION_ITF"}:
        return "F-ITF"
    return "BM25" if m == "BM25" else m


def _set_method(
    searcher,
    method: str,
    *,
    k1=0.9,
    b=0.4,
    mu=1000,
    jm_lambda=0.2,
    rm3_fb_terms: int = 20,
    rm3_fb_docs: int = 10,
    rm3_original_query_weight: float = 0.5,
):
    m = _canonical_method(method)
    if m == "BM25":
        searcher.set_bm25(k1=k1, b=b)
        params = {"k1": k1, "b": b}
    elif m == "DIRICHLET":
        searcher.set_qld(mu=mu)
        params = {"mu": mu}
    elif m == "JM":
        # Method #2 (Jelinek & Mercer, Lecture 9)
        # Use Pyserini's native JM smoothing if available
        try:
            searcher.set_lmjm(jm_lambda)
            params = {"lambda": jm_lambda}
        except AttributeError:
            # Fallback for older Pyserini versions that might miss the binding
            searcher.set_qld(mu=mu)
            params = {"lambda": jm_lambda, "mu": mu, "fallback": "QLD"}
    elif m == "TWO_STAGE":
        # Method #3 (Zhai & Lafferty, 2002): Two-stage smoothing
        # Stage 1 (Dirichlet) is used for candidate generation.
        # Stage 2 (Interpolation) is applied in the post-retrieval scoring loop.
        searcher.set_qld(mu=mu)
        params = {"mu": mu, "lambda": jm_lambda, "note": "Stage 2 applied in rescoring"}
    elif m == "KL_DIVERGENCE":
        # Method #4 (Lafferty & Zhai, 2001): KL-divergence ranking
        # QLD is the mathematically correct initial retrieval for KL with MLE query model.
        # Advanced KL (with query expansion) is handled via _apply_kl_reranking post-process.
        searcher.set_qld(mu=mu)
        params = {"mu": mu, "idf_boost": jm_lambda, "note": "KL rescoring applied post-retrieval"}
    elif m == "RM3":
        # Method #6 (RM3 PRF): build relevance model from top feedback docs, then interpolate with original query
        # Base scorer: Dirichlet LM to stay lecture-aligned; RM3 uses top fb_docs docs/ fb_terms terms
        # NOTE: We do NOT call searcher.set_rm3() here because it requires document vectors in the index.
        # Instead, we set the base scorer (QLD) and rely on manual client-side RM3 reranking in search().
        searcher.set_qld(mu=mu)
        params = {
            "mu": mu,
            "fb_terms": rm3_fb_terms,
            "fb_docs": rm3_fb_docs,
            "original_query_weight": rm3_original_query_weight,
            "note": "Client-side RM3 reranking"
        }
    elif m == "FUSION_BM25_LM":
        # Method #8 handled outside; here we set a sensible base for completeness
        searcher.set_qld(mu=mu)
        params = {"mu": mu, "fusion_weight": 0.6}
    elif m == "MMR_PASSAGES":
        # Method #7 (MMR passage diversification): use Dirichlet scoring, diversify later via TF-IDF cosine
        searcher.set_qld(mu=mu)
        params = {"mu": mu, "mmr": True}
    else:
        searcher.set_bm25(k1=k1, b=b)
        params = {"k1": k1, "b": b}
    logger = setup_logger()
    logger.info(f" Retrieval method: {m} | params={params}")


@lru_cache(maxsize=4)
def get_searcher_cached(
    prebuilt_name: str | None,
    local_path: str | None,
    method: str,
    k1: float,
    b: float,
    mu: int,
    jm_lambda: float,
    rm3_fb_terms: int,
    rm3_fb_docs: int,
    rm3_original_query_weight: float,
):
    canonical_method = _canonical_method(method)
    s = _load_searcher(prebuilt_name, local_path)
    _set_method(
        s,
        canonical_method,
        k1=k1,
        b=b,
        mu=mu,
        jm_lambda=jm_lambda,
        rm3_fb_terms=rm3_fb_terms,
        rm3_fb_docs=rm3_fb_docs,
        rm3_original_query_weight=rm3_original_query_weight,
    )
    return s


@lru_cache(maxsize=4)
def get_index_reader_cached(prebuilt_name: str | None, local_path: str | None):
    try:
        from pyserini.index.lucene import LuceneIndexReader
    except ImportError:
        from pyserini.index import IndexReader as LuceneIndexReader
    if local_path:
        return LuceneIndexReader(local_path)
    return LuceneIndexReader.from_prebuilt_index(prebuilt_name or "wikipedia-kilt-doc")





def _process_document_hits(
    hits: list,
    searcher,
    entity_boost_weight: float,
    query_lower: str
) -> list[tuple[float, str, str | None, dict[str, Any] | None, set[str], float]]:
    """Process initial document hits and extract basic information with entity boost scores."""
    doc_results = []
    for h in hits:
        doc = searcher.doc(h.docid)
        contents = None
        # Issue 27: Prefer existing content (e.g. from chunks) before expensive/failing doc fetch
        contents = getattr(h, 'contents', None)
        raw = getattr(h, 'raw', None)

        if not contents:
            doc = searcher.doc(h.docid)
            if doc is not None:
                try:
                    raw_json = doc.raw()
                    raw = json.loads(raw_json)
                    contents = raw.get("contents")
                except Exception:
                    contents = doc.raw() if doc is not None else None

        base_score = h.score
        entity_boost_score = 0.0
        phrases: set[str] = set()

        if isinstance(contents, str):
            phrases = {p.lower() for p in _extract_capitalized_phrases(contents)}
            if entity_boost_weight > 0:
                match_count = sum(1 for p in phrases if p and _phrase_in_query(p, query_lower))
                if match_count:
                    entity_boost_score = entity_boost_weight * match_count

        doc_results.append((base_score, h.docid, contents, raw, phrases, entity_boost_score))
    return doc_results


def _compute_entity_phrase_stats(
    doc_results: list,
    entity_df_boost_weight: float,
    entity_idf_boost_weight: float,
    booster_filter_enabled: bool,
    booster_min_consensus_docs: int
) -> tuple[Counter[str], dict[str, float], set[str]]:
    """Compute document frequency, IDF, and booster phrases for entity boosts."""
    phrase_df: Counter[str] = Counter()
    if entity_df_boost_weight > 0 or entity_idf_boost_weight > 0 or booster_filter_enabled:
        for _, _, _, _, phrases, _ in doc_results:
            phrase_df.update(phrases)

    # IDF computation for rare entities
    phrase_idf: dict[str, float] = {}
    if entity_idf_boost_weight > 0 and len(doc_results) > 0:
        total_docs = len(doc_results)
        for phrase in phrase_df.keys():
            df = phrase_df.get(phrase, 0) + 1
            phrase_idf[phrase] = math.log(total_docs / df)

    # Booster filtering: identify consensus (boosters) vs poison-pill entities
    booster_phrases: set[str] = set()
    if booster_filter_enabled and phrase_df:
        booster_phrases = {p for p in phrase_df.keys() if phrase_df[p] >= booster_min_consensus_docs}

    return phrase_df, phrase_idf, booster_phrases


def _apply_entity_boosts_to_documents(
    doc_results: list[tuple[float, str, str | None, dict[str, Any] | None, set[str], float]],
    phrase_df: Counter[str],
    phrase_idf: dict[str, float],
    booster_phrases: set[str],
    query_lower: str,
    entity_boost_weight: float,
    entity_df_boost_weight: float,
    entity_idf_boost_weight: float,
    booster_filter_enabled: bool
) -> list[tuple[float, str, str | None, dict[str, Any] | None, float, float, float, float]]:
    """Apply all entity boosts to documents and return boosted document tuples."""
    boosted_docs = []
    for base_score, docid, contents, raw, phrases, entity_boost_score in doc_results:
        entity_df_boost_score = 0.0
        entity_idf_boost_score = 0.0
        booster_filter_score = 0.0

        if (entity_df_boost_weight > 0 or entity_idf_boost_weight > 0 or booster_filter_enabled) and phrases:
            # Separate accumulators
            bonus_filter = 0.0
            bonus_df = 0.0
            bonus_idf = 0.0
            
            for p in phrases:

                # p is already lowercased from _process_document_hits
                # Check for boundary-aware match
                if _phrase_in_query(p, query_lower):
                    # Apply booster filtering: penalize poison pills, boost consensus phrases
                    if booster_filter_enabled:
                        if p not in booster_phrases:
                            # Poison pill: penalize (or optionally filter)
                            bonus_filter -= 0.05  # light penalty for isolated phrases
                        else:
                            # Booster: consensus phrase appearing in >=booster_min_consensus_docs
                            bonus_filter += 0.1
                    
                    if entity_df_boost_weight > 0:
                        df = phrase_df.get(p, 0)
                        if df > 1:
                            bonus_df += entity_df_boost_weight * (df - 1)
                    
                    if entity_idf_boost_weight > 0:
                        idf = phrase_idf.get(p, 0.0)
                        bonus_idf += entity_idf_boost_weight * idf

            entity_df_boost_score = bonus_df
            entity_idf_boost_score = bonus_idf
            booster_filter_score = bonus_filter

        total_score = base_score + entity_boost_score + entity_df_boost_score + entity_idf_boost_score + booster_filter_score
        boosted_docs.append((total_score, docid, contents, raw, entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score))

    boosted_docs.sort(key=lambda x: x[0], reverse=True)
    return boosted_docs


def _create_hit_from_boosted_doc(
    total_score: float,
    docid: str,
    contents: str | None,
    raw: dict[str, Any] | None,
    entity_boost_score: float,
    entity_df_boost_score: float,
    entity_idf_boost_score: float,
    booster_filter_score: float
) -> Hit:
    """Create a Hit object from boosted document data with score tracking."""
    return Hit(
        docid=docid,
        score=total_score,
        contents=contents,
        raw=raw,
        entity_boost_score=entity_boost_score,
        entity_df_boost_score=entity_df_boost_score,
        entity_idf_boost_score=entity_idf_boost_score,
        booster_filter_score=booster_filter_score
    )


def _process_passage_hits(
    hits: list,
    searcher,
    query: str,
    q_counts: Counter,
    mu: int,
    passage_mu: int,  # Issue 5 fix: separate mu for passages
    p_collection: dict,
    passage_window_size: int,
    passage_stride: int,
    passage_smoothing_alpha: float,
    entity_boost_weight: float,
    entity_df_boost_weight: float,
    entity_idf_boost_weight: float,
    booster_filter_enabled: bool,
    query_lower: str,
    booster_min_consensus_docs: int = 2,
    index_reader=None
) -> tuple[list[tuple[float, str, str, float, float, float, float, float, float, int, int]], Counter[str], dict[str, float], set[str], dict[str, list[str]]]:
    """Process passage-level retrieval with entity boosts."""
    passage_hits: list[tuple[float, str, str, float, float, float, float, float, float]] = []

    # Build doc counters first so we can estimate P(t|C) from the local candidate pool.
    doc_counts_by_id: dict[str, Counter] = {}
    doc_len_by_id: dict[str, int] = {}
    doc_contents_by_id: dict[str, str] = {}
    doc_term_counters: list[Counter] = []
    doc_phrase_sets: dict[str, set[str]] = {}

    for h in hits:
        doc = searcher.doc(h.docid)
        if doc is None:
            doc_counts = Counter()
            doc_contents = ""
        else:
            try:
                raw_json = doc.raw()
                raw = json.loads(raw_json)
                doc_contents = raw.get("contents", "")
            except Exception:
                doc_contents = doc.raw() if doc is not None else ""

            if not isinstance(doc_contents, str):
                doc_contents = ""

            doc_counts = Counter(_tokenize(doc_contents, index_reader=index_reader))

        doc_counts_by_id[h.docid] = doc_counts
        doc_len_by_id[h.docid] = sum(doc_counts.values())
        doc_contents_by_id[h.docid] = doc_contents
        doc_term_counters.append(doc_counts)
        if entity_df_boost_weight > 0 or entity_idf_boost_weight > 0 or booster_filter_enabled:
            doc_phrase_sets[h.docid] = {p.lower() for p in _extract_capitalized_phrases(doc_contents)}

    phrase_df: Counter[str] = Counter()
    phrase_idf: dict[str, float] = {}
    booster_phrases: set[str] = set()
    if entity_df_boost_weight > 0 or entity_idf_boost_weight > 0 or booster_filter_enabled:
        for phrases in doc_phrase_sets.values():
            phrase_df.update(phrases)
        if entity_idf_boost_weight > 0 and len(doc_phrase_sets) > 0:
            total_docs = len(doc_phrase_sets)
            for phrase in phrase_df.keys():
                df = phrase_df.get(phrase, 0) + 1
                phrase_idf[phrase] = math.log(total_docs / df)
        if booster_filter_enabled and phrase_df:
            booster_phrases = {p for p in phrase_df.keys() if phrase_df[p] >= booster_min_consensus_docs}

    if not p_collection and not index_reader:
        raise ValueError("Strict Math Mode: index_reader is required for BM25/LM statistics.")
        # p_collection = _estimate_collection_probs(list(q_counts.keys()), doc_term_counters)

    doc_passages_map: dict[str, list[str]] = {}

    for h in hits:
        doc_contents = doc_contents_by_id.get(h.docid, "")
        if not doc_contents:
            continue

        doc_counts = doc_counts_by_id.get(h.docid, Counter())
        doc_len = doc_len_by_id.get(h.docid, 0)
        doc_ll = _dirichlet_query_loglikelihood(q_counts, doc_counts, doc_len, mu, p_collection)

        # Segment into passages
        passages = _segment_passages(doc_contents, window_size=passage_window_size, stride=passage_stride)
        doc_passages_map[h.docid] = [p[2] for p in passages]
        total_passages = len(passages)
        
        for p_idx, (_, _, passage_text) in enumerate(passages):
            passage_scores = _calculate_passage_scores(
                passage_text, q_counts, doc_counts, doc_len, passage_mu, mu, p_collection, passage_smoothing_alpha, doc_ll, index_reader
            )
            entity_scores = _calculate_passage_entity_scores(
                passage_text, query_lower, phrase_df, phrase_idf, booster_phrases,
                entity_boost_weight, entity_df_boost_weight, entity_idf_boost_weight, booster_filter_enabled
            )

            total_score = passage_scores['total'] + sum(entity_scores.values())
            passage_hits.append((
                total_score, h.docid, passage_text,
                passage_scores['base'], entity_scores['boost'], entity_scores['df'], entity_scores['idf'], entity_scores['filter'],
                passage_scores['smoothing'],
                p_idx, total_passages
            ))

    return passage_hits, phrase_df, phrase_idf, booster_phrases, doc_passages_map


def _calculate_passage_scores(
    passage_text: str,
    q_counts: Counter,
    doc_counts: Counter,
    doc_len: int,
    passage_mu: int,
    doc_mu: int,
    p_collection: dict,
    passage_smoothing_alpha: float,
    doc_ll: float,
    index_reader=None  # Issue 6 fix: pass for consistent tokenization
) -> dict[str, float]:
    """Calculate base scores for a passage including hierarchical smoothing.
    
    Issue 5 fix: Uses separate passage_mu for passage smoothing (shorter texts
    need smaller mu to avoid collapsing toward collection model).
    """
    p_counts = Counter(_tokenize(passage_text, index_reader=index_reader))
    p_len = sum(p_counts.values())
    
    # Calculate hierarchical smoothing score: P(t|mix) = (1-alpha)*P(t|p) + alpha*P(t|d)
    score = 0.0
    for t, q_tf in q_counts.items():
        p_c = p_collection.get(t, 1e-9)
        
        # P(t|p) - Dirichlet smoothed passage probability (using passage_mu)
        p_tf = p_counts.get(t, 0)
        denom_p = p_len + passage_mu if (p_len + passage_mu) > 0 else 1.0
        prob_p = (p_tf + passage_mu * p_c) / denom_p
        
        # P(t|d) - Dirichlet smoothed document probability (using doc_mu)
        d_tf = doc_counts.get(t, 0)
        denom_d = doc_len + doc_mu if (doc_len + doc_mu) > 0 else 1.0
        prob_d = (d_tf + doc_mu * p_c) / denom_d
        
        # Mixture
        prob_mix = (1 - passage_smoothing_alpha) * prob_p + passage_smoothing_alpha * prob_d
        if prob_mix > 0:
            score += q_tf * math.log(prob_mix)

    return {
        'base': score,
        'total': score,
        'smoothing': score
    }


def _calculate_passage_entity_scores(
    passage_text: str,
    query_lower: str,
    phrase_df: Counter[str],
    phrase_idf: dict[str, float],
    booster_phrases: set[str],
    entity_boost_weight: float,
    entity_df_boost_weight: float,
    entity_idf_boost_weight: float,
    booster_filter_enabled: bool
) -> dict[str, float]:
    """Calculate entity boost scores for a passage."""
    phrases = {p.lower() for p in _extract_capitalized_phrases(passage_text)}

    entity_boost_score = _calculate_entity_boost_score(phrases, query_lower, entity_boost_weight)
    entity_df_boost_score = _calculate_entity_df_boost_score(phrases, query_lower, phrase_df, entity_df_boost_weight)
    entity_idf_boost_score = _calculate_entity_idf_boost_score(phrases, query_lower, phrase_idf, entity_idf_boost_weight)
    booster_filter_score = _calculate_booster_filter_score(phrases, query_lower, booster_phrases, booster_filter_enabled)

    return {
        'boost': entity_boost_score,
        'df': entity_df_boost_score,
        'idf': entity_idf_boost_score,
        'filter': booster_filter_score
    }


def search(
    query: str,
    *,
    top_k: int = 10,
    final_k: int | None = None,
    prebuilt_name: str | None = None,
    local_path: str | None = None,
    method: str = "BM25",
    k1: float = 1.2,
    b: float = 0.75,
    mu: int = 1000,
    jm_lambda: float = 0.1,
    use_passages: bool = False,
    passage_window_size: int = 150,
    passage_stride: int = 75,
    passage_smoothing_alpha: float = 0.2,
    passage_mu: int | None = None,  # Separate mu for passages (Issue 5); defaults to mu // 10
    passage_expand_neighbors: int = 0,
    passage_candidate_multiplier: int = 5,  # Callan '94: expand initial doc pool for recall
    passage_candidate_depth: int | None = None,  # Explicit depth (overrides multiplier if set)
    passage_strategy: str = "PASSAGE",  # PASSAGE (snippets) or MAX_P (doc ranking)
    # --- Query Expansion (happens BEFORE final retrieval) ---
    expand_query: bool = False,  # Enable query expansion
    expansion_method: str = "RM3",  # RM3, KL
    filter_poison_pills: bool = True,  # Filter low-consensus expansion terms
    apply_mmr: bool = False,
    mmr_lambda: float = 0.7,
    mmr_max_candidates: int | None = None,
    fusion_weight: float = 0.5,
    rm3_fb_terms: int = 50,
    rm3_fb_docs: int = 10,
    rm3_original_query_weight: float = 0.7,
    entity_boost_weight: float = 0.0,
    entity_df_boost_weight: float = 0.0,
    entity_idf_boost_weight: float = 0.0,
    booster_filter_enabled: bool = False,  # Enable entity consensus filtering
    booster_min_consensus_docs: int = 2,  # Used by filter_poison_pills
    rrf_enabled: bool = False,
    rrf_method1: str = "DIRICHLET",
    rrf_method2: str = "KL_DIVERGENCE",
    rrf_v: int = 60,
    itf_beta: float = 0.5,
    fitf_alpha: float = 0.5,  # F-ITF: weight for T-ITF vs E-ITF (Sheetrit et al.)
    bm25_extract_passages: bool = False,  # Extract relevant passages from top BM25 documents
    bm25_passage_extract_chars: int = 100000,  # Character limit for extracted passages from top docs
    apply_kl_reranker: bool = False,  # Apply KL-divergence reranking as post-processing
    apply_stlm_reranker: bool = False,  # Apply STLM (Structured Term LM) reranking
    stlm_lambda: float = 0.8,  # STLM: weight for terms (0.8-0.9 recommended; entities get 1-lambda)
    apply_char_splitter: bool = False,  # Apply character-based text splitting
    char_splitter_chunk_size: int = 1000,  # Character chunk size for splitting
    char_splitter_chunk_overlap: int = 200,  # Character overlap between chunks
    max_chars: int | None = None,  # Maximum characters per hit contents (truncation after processing)
) -> tuple[list[Hit], RetrievalMetadata]:
    """Run a search and return parsed hits with metadata.
    
    Supports both document-level and passage-level retrieval with hierarchical smoothing.

    Note: JM, TWO_STAGE, and KL_DIVERGENCE currently fall back to Dirichlet at document level;
    passage mode applies hierarchical smoothing (passage  document  collection).
    
    Args:
        top_k: Number of documents/passages to retrieve initially
        final_k: If specified, limit final results to this many items (applied after all processing)
        use_passages: If True, segment top documents into passages and re-rank by passage score
        passage_window_size: Target passage length in words (approx)
        passage_stride: Overlap stride in words (approx); smaller = more overlap
        passage_smoothing_alpha: Hierarchical smoothing weight; (1-alpha)*passage_score + alpha*doc_score
        apply_mmr: If True, apply MMR diversification over passages (Method #7)
        mmr_lambda: Trade-off between relevance (higher) and diversity (lower)
        mmr_max_candidates: Max passages to consider for MMR (limits cost); defaults to 4x top_k
        fusion_weight: Weight for LM vs BM25 in score fusion (Method #8), higher = prioritize LM
        bm25_extract_passages: If True, extract query-relevant passages from top docs (up to X chars)
        bm25_passage_extract_chars: Character limit for passage extraction from each top doc
        apply_kl_reranker: If True, apply KL-divergence reranking as post-processing on any base method
        apply_char_splitter: If True, split retrieved documents into character-based chunks
        char_splitter_chunk_size: Size of each character chunk
        char_splitter_chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        tuple[list[Hit], RetrievalMetadata]: List of hits (passages if use_passages=True) and metadata.
    """
    logger = setup_logger()
    logger.info(f"Query: '{query}' | top_k={top_k} | passages={use_passages}")
    canonical_method = _canonical_method(method)
    requested_k = int(top_k)
    query_lower = query.lower()

    # 1. Setup Defaults & Maps
    if canonical_method == "RM3":
        expand_query = True
        expansion_method = "RM3"
    
    # Initialize index_reader
    index_reader = get_index_reader_cached(prebuilt_name, local_path)

    # Tokenize initial query
    query_terms = _tokenize(query, index_reader=index_reader)
    q_counts = Counter(query_terms)
    
    # Initialize p_collection (Global)
    p_collection = {}
    if index_reader:
         p_collection = _get_pyserini_collection_probs(query_terms, index_reader)
    
    # 2. Query Expansion (The "Good" RM3)
    # This runs BEFORE retrieval to modify the query
    if expand_query:
        msg = f"Query Expansion: {expansion_method} (fb_docs={rm3_fb_docs}, fb_terms={rm3_fb_terms})"
        logger.info(msg.encode('ascii', 'ignore').decode('ascii'))
        
        # Get initial feedback documents
        expansion_searcher = get_searcher_cached(
            prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda,
            rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight
        )
        feedback_hits_raw = expansion_searcher.search(query, rm3_fb_docs * 2)
        
        # Convert to Hit objects for feedback
        feedback_hits = []
        for hit in feedback_hits_raw[:rm3_fb_docs]:
            doc = expansion_searcher.doc(hit.docid)
            if doc is None:
                contents = ""
                raw = None
            else:
                try:
                    raw_json = doc.raw()
                    raw = json.loads(raw_json)
                    contents = raw.get("contents", "")
                except Exception:
                    contents = doc.raw() if doc is not None else ""
                    raw = None
            feedback_hits.append(Hit(
                docid=hit.docid, score=hit.score, contents=contents, raw=raw
            ))
        
        # Build feedback model
        feedback_term_counters, feedback_lengths = _build_feedback_term_counters(feedback_hits)
        rm_probs = _build_relevance_model(
            feedback_hits, feedback_term_counters, feedback_lengths,
            q_counts, mu, p_collection
        )
        
        # Apply poison pills filter if enabled
        if filter_poison_pills:
            original_count = len(rm_probs)
            rm_probs = _filter_poison_pills(rm_probs, feedback_hits, booster_min_consensus_docs, index_reader)
            logger.info(f"   Poison pills: filtered {original_count - len(rm_probs)} low-consensus terms")
        
        # Build expanded query string
        expanded_terms = _expand_query_with_rm3(q_counts, rm_probs, rm3_fb_terms, rm3_original_query_weight)
        total_weight = sum(expanded_terms.values())
        if total_weight > 0:
            query_parts = []
            for term, weight in sorted(expanded_terms.items(), key=lambda x: x[1], reverse=True):
                if weight > 0:
                    normalized = weight / total_weight
                    query_parts.append(f"{term}^{normalized:.4f}")
            expanded_query = " ".join(query_parts[:rm3_fb_terms + len(q_counts)])
            logger.info(f"   Expanded query: {expanded_query[:100]}...")
            
            # Update query for subsequent retrieval steps
            query = expanded_query
    
    # Use effective query (now updated in 'query' variable)
    effective_query = query
    
    hits = []
    
    # Calculate retrieval depth (doc_pool_k) ONCE for all methods
    # Start with requested_k
    # If using passages, we multiply to get enough docs to cover passage segments (Callan '94)
    doc_pool_k = (requested_k * passage_candidate_multiplier) if use_passages else requested_k
    
    # Allow explicit override
    if passage_candidate_depth:
        doc_pool_k = passage_candidate_depth
        
    # Cap doc_pool_k to something reasonable to prevent OOM
    if doc_pool_k > 10000:
        logger.warning(f"doc_pool_k={doc_pool_k} is very large. Capping at 10000.")
        doc_pool_k = 10000

    # 3. Retrieval Strategy Fork
    
    # Handle F-ITF (Fusion of T-ITF and E-ITF)
    if canonical_method == "F-ITF":
        logger.info(f" F-ITF requested: Linear fusion with ={fitf_alpha}")
        
        # Get initial documents using Dirichlet
        s = get_searcher_cached(
            prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda,
            rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight
        )
        # Use doc_pool_k for depth
        initial_k = max(doc_pool_k * 3, 100)
        initial_hits = s.search(effective_query, initial_k)
        
        if initial_hits:
            # Build query models
            q_term_model = _normalize_query_model(q_counts)
            q_entities = _extract_capitalized_phrases(query)
            q_entity_model = _normalize_query_model(Counter([e.lower() for e in q_entities]))
            
            # Prepare Document Models & Collection Stats
            doc_data = []
            all_entity_counts = Counter()
            
            for hit in initial_hits:
                doc = s.doc(hit.docid)
                if doc is None:
                    contents = ""
                else:
                    try:
                        raw_json = doc.raw()
                        raw = json.loads(raw_json)
                        contents = raw.get("contents", "")
                    except Exception:
                        contents = doc.raw() if doc is not None else ""
                tokens = _tokenize(contents, index_reader=index_reader)
                ents = _extract_capitalized_phrases(contents)
                ent_counts = Counter([e.lower() for e in ents])
                all_entity_counts.update(ent_counts)
                
                doc_data.append((hit, Counter(tokens), len(tokens), ent_counts, sum(ent_counts.values()), contents))
                
            # Entity Collection Probabilities (Local Estimate from Candidate Set)
            total_ents = sum(all_entity_counts.values())
            p_coll_ent = {e: c/total_ents for e,c in all_entity_counts.items()} if total_ents > 0 else {}
            
            scored_hits = []
            for hit, term_c, term_l, ent_c, ent_l, contents in doc_data:
                titf = _titf_kl_score(q_term_model, term_c, term_l, p_collection, mu)
                eitf = _eitf_kl_score(q_entity_model, ent_c, ent_l, p_coll_ent, mu=100)
                fitf = _fitf_score(titf, eitf, fitf_alpha)
                
                new_hit = Hit(
                    docid=hit.docid, 
                    score=fitf, 
                    contents=contents, 
                    raw=None,
                    titf_score=titf,
                    fitf_score=fitf
                )
                scored_hits.append(new_hit)
            
            scored_hits.sort(key=lambda h: h.score, reverse=True)
            hits = scored_hits[:doc_pool_k]

    # Reciprocal Rank Fusion
    elif rrf_enabled:
        logger.info(f" RRF enabled: {rrf_method1} + {rrf_method2} (v={rrf_v})")
        # Run both retrieval methods
        hits1, meta1 = search(
            query,
            top_k=doc_pool_k,
            prebuilt_name=prebuilt_name,
            local_path=local_path,
            method=rrf_method1,
            k1=k1, b=b, mu=mu, jm_lambda=jm_lambda,
            use_passages=False,  # RRF fix: force document-level fusion
            apply_mmr=False,  # RRF fix: disable MMR in sub-searches
            apply_char_splitter=False,  # RRF fix: disable chunking
            entity_boost_weight=entity_boost_weight,
            entity_df_boost_weight=entity_df_boost_weight,
            entity_idf_boost_weight=entity_idf_boost_weight,
            booster_filter_enabled=booster_filter_enabled,
            booster_min_consensus_docs=booster_min_consensus_docs,
            rrf_enabled=False,  # Prevent infinite recursion
        )
        hits2, meta2 = search(
            query,
            top_k=doc_pool_k,
            prebuilt_name=prebuilt_name,
            local_path=local_path,
            method=rrf_method2,
            k1=k1, b=b, mu=mu, jm_lambda=jm_lambda,
            use_passages=False,  # RRF fix: force document-level fusion
            apply_mmr=False,  # RRF fix: disable MMR in sub-searches
            apply_char_splitter=False,  # RRF fix: disable chunking
            entity_boost_weight=entity_boost_weight,
            entity_df_boost_weight=entity_df_boost_weight,
            entity_idf_boost_weight=entity_idf_boost_weight,
            booster_filter_enabled=booster_filter_enabled,
            booster_min_consensus_docs=booster_min_consensus_docs,
            rrf_enabled=False,  # Prevent infinite recursion
        )
        
        # Compute RRF scores: Score(d) = sum(1 / (v + rank(d)))
        rrf_scores: dict[str, float] = {}
        for rank, hit in enumerate(hits1, start=1):
            if hit.docid not in rrf_scores:
                rrf_scores[hit.docid] = 0.0
            rrf_scores[hit.docid] += 1.0 / (rrf_v + rank)
        
        for rank, hit in enumerate(hits2, start=1):
            if hit.docid not in rrf_scores:
                rrf_scores[hit.docid] = 0.0
            rrf_scores[hit.docid] += 1.0 / (rrf_v + rank)
        
        # Sort by RRF score and return top_k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:doc_pool_k]
        
        # Hydrate hits with content from original results
        content_map = {}
        raw_map = {}
        for h in hits1 + hits2:
            content_map[h.docid] = h.contents
            raw_map[h.docid] = h.raw

        hits = []
        for docid, score in sorted_docs:
            hit = Hit(
                docid=docid, 
                score=score, 
                raw=raw_map.get(docid), 
                contents=content_map.get(docid, ""), 
                rrf_score=score
            )
            hits.append(hit)
            
        # Ensure searcher 's' is available for later use (fallback to standard)
        s = get_searcher_cached(prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)
    
    # 4. Standard Retrieval (Fallback)
    if not hits:
        # doc_pool_k is already calculated at top of function
        if canonical_method == "FUSION_BM25_LM":
            logger.info(f" Running Fusion BM25 + LM (weight={fusion_weight})")
            s_bm25 = get_searcher_cached(prebuilt_name, local_path, "BM25", k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)
            s_lm = get_searcher_cached(prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)

            bm25_hits = s_bm25.search(effective_query, doc_pool_k * 2)
            lm_hits = s_lm.search(effective_query, doc_pool_k * 2)

            bm25_scores = {h.docid: h.score for h in bm25_hits}
            lm_scores = {h.docid: h.score for h in lm_hits}
            bm25_z = _zscore(bm25_scores)
            lm_z = _zscore(lm_scores)
            min_bm = (min(bm25_z.values()) if bm25_z else 0.0) - 1.0
            min_lm = (min(lm_z.values()) if lm_z else 0.0) - 1.0

            fused: list[tuple[float, str]] = []
            for docid in set(bm25_scores.keys()) | set(lm_scores.keys()):
                bm = bm25_z.get(docid, min_bm)
                lm = lm_z.get(docid, min_lm)
                fused_score = fusion_weight * lm + (1 - fusion_weight) * bm
                fused.append((fused_score, docid))

            fused.sort(key=lambda x: x[0], reverse=True)
            
            # Hydrate hits
            hits = []
            s = s_lm 
            for fs, docid in fused[:doc_pool_k]:
                 hits.append(Hit(docid=docid, score=fs, contents=None, raw=None, fusion_score=fs))
        elif canonical_method in ("T-ITF", "E-ITF"):
             # Legacy ITF block - fallback to standard if F-ITF not used
             logger.info(f"Running Legacy {canonical_method} logic (Delegating to Standard Dirichlet)")
             s_final = get_searcher_cached(prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)
             hits_raw = s_final.search(effective_query, doc_pool_k)
             hits = []
             for h in hits_raw:
                 doc = s_final.doc(h.docid)
                 if doc is None:
                     hits.append(Hit(docid=h.docid, score=h.score, contents="", raw=None))
                     continue
                 try:
                     raw_json = doc.raw()
                     raw = json.loads(raw_json)
                     contents = raw.get("contents", "")
                 except Exception:
                     contents = doc.raw() if doc is not None else ""
                     raw = None
                 hits.append(Hit(docid=h.docid, score=h.score, contents=contents, raw=raw))
             s = s_final

        elif canonical_method == "KL_DIVERGENCE":
            # Retrieve a larger candidate set with Dirichlet, then rerank via KL(M_q || M_d)
            s = get_searcher_cached(
                prebuilt_name,
                local_path,
                "DIRICHLET",
                k1,
                b,
                mu,
                jm_lambda,
                rm3_fb_terms,
                rm3_fb_docs,
                rm3_original_query_weight,
            )
            candidate_k = max(100, doc_pool_k * 10)
            base_hits = s.search(effective_query, candidate_k)

            # Build term counters for docs and query
            q_counts_kl = Counter(_tokenize(effective_query, index_reader=index_reader))
            doc_term_counters: list[Counter] = []
            doc_lengths: dict[str, int] = {}
            enriched_hits: list[tuple[str, str]] = []  # (docid, contents) for content hydration
            for h in base_hits:
                doc = s.doc(h.docid)
                if doc is None:
                    doc_term_counters.append(Counter())
                    doc_lengths[h.docid] = 0
                    enriched_hits.append((h.docid, ""))
                    continue
                try:
                    raw_json = doc.raw()
                    raw = json.loads(raw_json)
                    contents = raw.get("contents", "")
                except Exception:
                    contents = doc.raw() if doc is not None else ""
                toks = _tokenize(contents, index_reader=index_reader)
                c = Counter(toks)
                doc_term_counters.append(c)
                doc_lengths[h.docid] = sum(c.values())
                enriched_hits.append((h.docid, contents))

            if not p_collection:
                p_collection = _estimate_collection_probs(list(q_counts_kl.keys()), doc_term_counters)

            # Compute KL scores and sort
            scored: list[tuple[float, str]] = []
            for i, h in enumerate(base_hits):
                c = doc_term_counters[i]
                dlen = doc_lengths.get(h.docid, 0)
                kl = _neg_kl_divergence_score(q_counts_kl, c, dlen, mu, p_collection)
                scored.append((kl, h.docid))
            scored.sort(key=lambda x: x[0], reverse=True)

            # Build map from enriched hits (collected during tokenization loop)
            doc_data_map = {docid: contents for docid, contents in enriched_hits}
            hits = []
            for score, docid in scored[:doc_pool_k]:
                # Use doc_data_map (populated during tokenization) instead of re-fetching
                contents = doc_data_map.get(docid, "")
                hits.append(Hit(
                    docid=docid,
                    score=score,
                    contents=contents,
                    raw=None,  # Raw not cached, can be None
                    kl_divergence_score=score
                ))
        elif canonical_method in ("T-ITF", "E-ITF"):
             # Legacy ITF block - fallback to standard to ensure stability
             logger.info(f"Running Legacy {canonical_method} logic (Delegating to Standard Dirichlet)")
             s_final = get_searcher_cached(prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)
             hits_raw = s_final.search(effective_query, doc_pool_k)
             hits = []
             for h in hits_raw:
                 doc = s_final.doc(h.docid)
                 if doc is None:
                     hits.append(Hit(docid=h.docid, score=h.score, contents="", raw=None))
                     continue
                 try:
                     raw_json = doc.raw()
                     raw = json.loads(raw_json)
                     contents = raw.get("contents", "")
                 except Exception:
                     contents = doc.raw() if doc is not None else ""
                     raw = None
                 hits.append(Hit(docid=h.docid, score=h.score, contents=contents, raw=raw))
             s = s_final

        elif canonical_method == "KL_DIVERGENCE":
            logger.info("Running KL Divergence Retrieval")
            s = get_searcher_cached(prebuilt_name, local_path, "DIRICHLET", k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)
            base_hits = s.search(effective_query, max(100, doc_pool_k * 10))
            
            if not index_reader:
                raise ValueError("Strict Math Mode: index_reader is required for BM25/LM statistics.")
                #  hits = []
                #  for h in base_hits[:doc_pool_k]:
                #      doc = s.doc(h.docid)
                #      if doc is None:
                #          hits.append(Hit(h.docid, h.score, "", None))
                #          continue
                #      try:
                #          raw_json = doc.raw()
                #          raw = json.loads(raw_json)
                #          contents = raw.get("contents", "")
                #      except Exception:
                #          contents = doc.raw() if doc is not None else ""
                #          raw = None
                #      hits.append(Hit(h.docid, h.score, contents, raw))
            else:
                 scored = []
                 for h in base_hits:
                     doc = s.doc(h.docid)
                     if doc is None:
                         ct = ""
                     else:
                         raw_json = doc.raw()
                         raw = json.loads(raw_json)
                         ct = raw.get("contents", "")
                     c = Counter(_tokenize(ct, index_reader))
                     # Neg KL uses q_counts (global)
                     q_c = q_counts if 'q_counts' in locals() else Counter(_tokenize(effective_query, index_reader))
                     kl = _neg_kl_divergence_score(q_c, c, sum(c.values()), mu, p_collection)
                     scored.append((kl, h, ct))
                 scored.sort(key=lambda x: x[0], reverse=True)
                 hits = []
                 for sc, h, ct in scored[:doc_pool_k]:
                     hits.append(Hit(docid=h.docid, score=sc, contents=ct, raw=None, kl_divergence_score=sc))
        else:
            # Standard Retrieval
            s_final = get_searcher_cached(prebuilt_name, local_path, canonical_method, k1, b, mu, jm_lambda, rm3_fb_terms, rm3_fb_docs, rm3_original_query_weight)
            hits_raw = s_final.search(effective_query, doc_pool_k)
            hits = []
            for h in hits_raw:
                doc = s_final.doc(h.docid)
                if doc is None:
                    hits.append(Hit(docid=h.docid, score=h.score, contents="", raw=None))
                    continue
                raw_json = doc.raw()
                raw = json.loads(raw_json)
                contents = raw.get("contents", "")
                hits.append(Hit(docid=h.docid, score=h.score, contents=contents, raw=raw))
            s = s_final

    
    # Capture extra scores from hits before they are processed (ITF, RRF, etc.)
    extra_scores = {}
    for h in hits:
        extra_scores[h.docid] = {
            'titf': getattr(h, 'titf_score', None),
            'fitf': getattr(h, 'fitf_score', None),
            'rrf': getattr(h, 'rrf_score', None)
        }

    results: list[Hit] = []
    
    # 5. Document/Passage Processing
    if not use_passages:
        # Standard document retrieval
        doc_results = _process_document_hits(hits, s, entity_boost_weight, query_lower)
        phrase_df, phrase_idf, booster_phrases = _compute_entity_phrase_stats(
            doc_results, entity_df_boost_weight, entity_idf_boost_weight, booster_filter_enabled, booster_min_consensus_docs
        )
        boosted_docs = _apply_entity_boosts_to_documents(
            doc_results, phrase_df, phrase_idf, booster_phrases, query_lower,
            entity_boost_weight, entity_df_boost_weight, entity_idf_boost_weight, booster_filter_enabled
        )

        # BM25 passage extraction
        if bm25_extract_passages and boosted_docs:
            logger.info(f" Extracting passages (max {bm25_passage_extract_chars} chars) from top docs")
            # Re-tokenize query for overlap check
            query_terms_set = set(query_lower.split())
            
            for total_score, docid, contents, raw, entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score in boosted_docs:
                if not contents:
                    results.append(_create_hit_from_boosted_doc(
                        total_score, docid, contents, raw,
                        entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score
                    ))
                    continue
                
                # Split document into sentences (simple heuristic)
                sentences = [s.strip() for s in contents.replace('!', '.').replace('?', '.').split('.') if s.strip()]
                
                # Score sentences by query term overlap
                sentence_scores = []
                for sent in sentences:
                    sent_lower = sent.lower()
                    sent_words = set(sent_lower.split())
                    overlap = len(query_terms_set & sent_words)
                    sentence_scores.append((overlap, sent))
                
                # Sort by relevance and extract top sentences
                sentence_scores.sort(key=lambda x: x[0], reverse=True)
                extracted_passage = ""
                for _, sent in sentence_scores:
                    if len(extracted_passage) + len(sent) + 2 <= bm25_passage_extract_chars:
                        if extracted_passage:
                            extracted_passage += " " + sent
                        else:
                            extracted_passage = sent
                    else:
                        break
                
                if not extracted_passage and contents:
                    extracted_passage = contents[:bm25_passage_extract_chars]
                
                results.append(_create_hit_from_boosted_doc(
                    total_score, docid, extracted_passage, raw,
                    entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score
                ))
        else:
            for total_score, docid, contents, raw, entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score in boosted_docs:
                if max_chars is not None and contents and len(contents) > max_chars:
                    contents = contents[:max_chars]
                results.append(_create_hit_from_boosted_doc(
                    total_score, docid, contents, raw,
                    entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score
                ))
    else:
        # Passage-based retrieval
        effective_passage_mu = passage_mu if passage_mu is not None else max(mu // 10, 100)
        passage_hits, phrase_df, phrase_idf, booster_phrases, doc_passages_map = _process_passage_hits(
            hits, s, effective_query, q_counts, mu, effective_passage_mu, p_collection, passage_window_size, passage_stride,
            passage_smoothing_alpha, entity_boost_weight, entity_df_boost_weight, entity_idf_boost_weight,
            booster_filter_enabled, query_lower, booster_min_consensus_docs, index_reader
        )
        
        passage_hits.sort(key=lambda x: x[0], reverse=True)
        if apply_mmr:
            reranked = _mmr_rerank(
                passage_hits,
                top_k=requested_k,
                lambda_diversity=mmr_lambda,
                max_candidates=mmr_max_candidates,
            )
            for total_score, docid, passage_text, base_score, entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score, passage_smoothing_score, p_idx, total_passages in reranked:
                if passage_expand_neighbors > 0:
                     start_idx = max(0, p_idx - passage_expand_neighbors)
                     end_idx = min(total_passages, p_idx + passage_expand_neighbors + 1)
                     neighbor_passages = doc_passages_map[docid][start_idx:end_idx]
                     passage_text = "\n".join(neighbor_passages)

                mmr_score = total_score
                if max_chars is not None and len(passage_text) > max_chars:
                    passage_text = passage_text[:max_chars]
                results.append(Hit(
                    docid=docid, 
                    score=mmr_score, 
                    contents=passage_text, 
                    raw=None,
                    entity_boost_score=entity_boost_score,
                    entity_df_boost_score=entity_df_boost_score,
                    entity_idf_boost_score=entity_idf_boost_score,
                    booster_filter_score=booster_filter_score,
                    passage_smoothing_score=passage_smoothing_score,
                    mmr_score=mmr_score
                ))
        else:
            for total_score, docid, passage_text, base_score, entity_boost_score, entity_df_boost_score, entity_idf_boost_score, booster_filter_score, passage_smoothing_score, p_idx, total_passages in passage_hits[:requested_k]:
                if passage_expand_neighbors > 0:
                     start_idx = max(0, p_idx - passage_expand_neighbors)
                     end_idx = min(total_passages, p_idx + passage_expand_neighbors + 1)
                     neighbor_passages = doc_passages_map[docid][start_idx:end_idx]
                     passage_text = "\n".join(neighbor_passages)
                
                if max_chars is not None and len(passage_text) > max_chars:
                    passage_text = passage_text[:max_chars]
                
                results.append(Hit(
                    docid=docid, 
                    score=total_score, 
                    contents=passage_text, 
                    raw=None,
                    entity_boost_score=entity_boost_score,
                    entity_df_boost_score=entity_df_boost_score,
                    entity_idf_boost_score=entity_idf_boost_score,
                    booster_filter_score=booster_filter_score,
                    passage_smoothing_score=passage_smoothing_score
                ))
        
        # MaxP Aggregation if requested
        if passage_strategy == "MAX_P":
             logger.info(" Aggregating passages via MaxP")
             results = _aggregate_max_p(results)

    # 6. Global Probability Hydration (Fix Leaky Probs)
    if index_reader:
        all_terms = set(q_counts.keys())
        for h in results:
             text = h.contents or ""
             all_terms.update(_tokenize(text, index_reader=index_reader))
        # Update p_collection with global stats for all encountered terms
        new_probs = _get_pyserini_collection_probs(list(all_terms), index_reader)
        p_collection.update(new_probs)

    # 7. Reranking Chains
    if apply_stlm_reranker:
        logger.info(f" Applying STLM Reranking (={stlm_lambda})")
        results = _apply_stlm_reranking(results, effective_query, stlm_lambda, mu, index_reader)
        
    if apply_kl_reranker:
        logger.info(" Applying KL Reranking")
        results = _apply_kl_reranking(results, effective_query, s, mu, index_reader)
    
    # Attach extra scores (ITF, RRF) if available
    if extra_scores:
        for hit in results:
            if hit.docid in extra_scores:
                scores = extra_scores[hit.docid]
                if scores['titf'] is not None: hit.titf_score = scores['titf']
                if scores['fitf'] is not None: hit.fitf_score = scores['fitf']
                if scores['rrf'] is not None: hit.rrf_score = scores['rrf']

    # Build metadata
    method_upper = canonical_method
    parameters: dict[str, Any] = {}
    
    if method_upper == "BM25":
        parameters = {"k1": k1, "b": b}
    elif method_upper == "DIRICHLET":
        parameters = {"mu": mu}
    elif method_upper == "JM":
        parameters = {"lambda": jm_lambda, "mu": mu}
    elif method_upper == "TWO_STAGE":
        parameters = {"mu": mu, "lambda": jm_lambda}  # mu used as fallback
    elif method_upper == "KL_DIVERGENCE":
        parameters = {"mu": mu, "idf_boost": jm_lambda, "candidate_pool": max(top_k * 3, top_k), "collection_estimation": "top_docs"}
    elif method_upper == "RM3":
        parameters = {
            "mu": mu,
            "fb_terms": rm3_fb_terms,
            "fb_docs": rm3_fb_docs,
            "original_query_weight": rm3_original_query_weight,
        }
    elif method_upper == "MMR_PASSAGES":
        parameters = {"mu": mu, "mmr_lambda": mmr_lambda, "mmr_max_candidates": mmr_max_candidates}
    elif method_upper == "FUSION_BM25_LM":
        parameters = {"mu": mu, "k1": k1, "b": b, "fusion_weight": fusion_weight}
    
    if use_passages:
        parameters["passage_window_size"] = passage_window_size
        parameters["passage_stride"] = passage_stride
        parameters["passage_smoothing_alpha"] = passage_smoothing_alpha
        if apply_mmr:
            parameters["mmr_lambda"] = mmr_lambda
            parameters["mmr_max_candidates"] = mmr_max_candidates
    if entity_boost_weight > 0:
        parameters["entity_boost_weight"] = entity_boost_weight
    if entity_df_boost_weight > 0:
        parameters["entity_df_boost_weight"] = entity_df_boost_weight
    if entity_idf_boost_weight > 0:
        parameters["entity_idf_boost_weight"] = entity_idf_boost_weight
    if booster_filter_enabled:
        parameters["booster_filter_enabled"] = booster_filter_enabled
        parameters["booster_min_consensus_docs"] = booster_min_consensus_docs
    if bm25_extract_passages:
        parameters["bm25_extract_passages"] = bm25_extract_passages
        parameters["bm25_passage_extract_chars"] = bm25_passage_extract_chars
    if apply_kl_reranker:
        parameters["apply_kl_reranker"] = apply_kl_reranker
    if apply_char_splitter:
        parameters["apply_char_splitter"] = apply_char_splitter
        parameters["char_splitter_chunk_size"] = char_splitter_chunk_size
        parameters["char_splitter_chunk_overlap"] = char_splitter_chunk_overlap
    
    metadata = RetrievalMetadata(
        method=method_upper,
        parameters=parameters,
        query_original=query,
        query_modified=query,  # No modifications in basic search
        query_modifications={},
        index_name=prebuilt_name or local_path or "wikipedia-kilt-doc",
        top_k=requested_k,
        final_k=final_k,
    )
    
    # Apply character-based splitting if enabled (before reranking)
    if apply_char_splitter and results:
        logger.info(f" Splitting documents into character chunks (size={char_splitter_chunk_size}, overlap={char_splitter_chunk_overlap})")
        split_results = []
        splitter = CharacterTextSplitter(
            chunk_size=char_splitter_chunk_size,
            chunk_overlap=char_splitter_chunk_overlap,
            separator=""
        )
        for hit in results:
            if hit.contents and len(hit.contents.strip()) > char_splitter_chunk_size:
                chunks = splitter.split_text(hit.contents)
                for i, chunk in enumerate(chunks):
                    # Issue 27: Docid mutation. Note that downstream logic using searcher.doc(docid) 
                    # will fail for these chunks. We must ensure h.contents is used.
                    chunk_hit = Hit(
                        docid=f"{hit.docid}_chunk_{i}",
                        score=hit.score,  # Keep original retrieval score
                        contents=chunk,
                        raw=hit.raw,
                        query_likelihood_score=0.0, # Initialize to avoid None type errors later
                        kl_divergence_score=0.0
                    )
                    split_results.append(chunk_hit)
            else:
                split_results.append(hit)
        results = split_results
    

    
    # Compute additional scores for each hit
    if results:
        logger.info(" Computing query likelihood and KL divergence scores")
        query_lower = query.lower()
        query_terms = _tokenize(query, index_reader=index_reader)
        query_counts = Counter(query_terms)
        
        # Build term counters for all retrieved docs/chunks using their contents
        doc_term_counters: list[Counter] = []
        doc_lengths: dict[str, int] = {}
        
        for hit in results:
            contents = hit.contents or ""
            toks = _tokenize(contents)
            c = Counter(toks)
            doc_term_counters.append(c)
            doc_lengths[hit.docid] = len(toks)
        
        if index_reader:
            # Merge existing p_collection with any missing query terms
            global_probs = _get_pyserini_collection_probs(query_terms, index_reader)
            p_collection.update(global_probs)
        
        if not p_collection:
            logger.warning(" Index reader missing & p_collection empty: Scoring will use uniform defaults.")
            # p_collection remains empty, scoring functions must handle missing keys gracefully
        
        # Pre-calculate stats for new scores
        # Default to local stats (fallback)
        total_docs = len(results)
        doc_freqs = Counter()
        for c in doc_term_counters:
            doc_freqs.update(c.keys())
        avg_doc_len = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 0
        
        # Try to get global stats if available (PROPER WAY)
        if index_reader:
            try:
                stats = index_reader.stats()
                # Use global N and avg_dl
                if 'documents' in stats and stats['documents'] > 0:
                    total_docs = stats['documents']
                    if 'total_terms' in stats:
                        avg_doc_len = stats['total_terms'] / stats['documents']
                
                # Update doc_freqs with global DF for query terms only (Optimization: Issue 7 & 10)
                # Fetching global DF for ALL terms in all docs is O(D*L) JNI calls which is too slow.
                # We prioritize query terms for BM25/JM/KL accuracy.
                # Non-query terms will fall back to local DF (computed above in doc_freqs via update) for Cosine normalization.
                batch_terms = set(query_counts.keys())
                
                # Use relevant terms if available (for expansion methods)

                    
                # If we really need better Cosine norms, we'd need global DF for all terms,
                # but we trade that off for runtime speed here.
                
                for t in batch_terms:
                    # get_term_counts returns (df, cf)
                    # We only need df here
                    try:
                        df, _ = index_reader.get_term_counts(t, analyzer=None)
                        if df > 0:
                            doc_freqs[t] = df
                        else:
                            # Term not in index? Keep local or set to 0?
                            # If not in global index, it shouldn't contribute to global score.
                            doc_freqs[t] = 0
                    except Exception:
                        pass # Keep local or 0
                        
            except Exception as e:
                logger.warning(f"Failed to get global stats from index_reader: {e}")

        for i, hit in enumerate(results):
            try:
                doc_counts = doc_term_counters[i]
                doc_len = doc_lengths.get(hit.docid, 0)
                
                # Compute query likelihood score (log P(q|d))
                hit.query_likelihood_score = _dirichlet_query_loglikelihood(
                    query_counts, doc_counts, doc_len, mu, p_collection
                )
                
                # Compute true -KL divergence score
                hit.kl_divergence_score = _neg_kl_divergence_score(
                    query_counts, doc_counts, doc_len, mu, p_collection
                )

                # Compute JM Score
                jm_s = 0.0
                for t, q_tf in query_counts.items():
                    p_c = p_collection.get(t, 1e-9)
                    # P_JM(t|d) = (1-lambda) * P(t|d) + lambda * P(t|C)
                    doc_prob = doc_counts.get(t, 0) / doc_len if doc_len > 0 else 0.0
                    smoothed_prob = (1 - jm_lambda) * doc_prob + jm_lambda * p_c
                    if smoothed_prob > 0:
                        jm_s += q_tf * math.log(smoothed_prob)
                hit.jm_score = jm_s

                # Compute Local/Global BM25 Score
                bm25_s = 0.0
                for t, q_tf in query_counts.items():
                    df = doc_freqs.get(t, 0)
                    bm25_s += q_tf * _calculate_bm25_term_score(
                        doc_counts.get(t, 0), doc_len, avg_doc_len, df, total_docs, k1, b
                    )
                hit.bm25_score = bm25_s

                # Compute TF-IDF Score and Cosine Similarity
                tfidf_s = 0.0
                doc_vec_norm_sq = 0.0
                query_vec_norm_sq = 0.0
                
                for t, q_tf in query_counts.items():
                    # Query TF-IDF (simplified, just TF here or TF*IDF)
                    # Let's use TF*IDF for query too for proper Cosine
                    q_idf = _calculate_idf(doc_freqs.get(t, 0), total_docs)
                    q_val = q_tf * q_idf
                    query_vec_norm_sq += q_val * q_val
                    
                    # Doc TF-IDF
                    tf_val = _calculate_tf(doc_counts.get(t, 0), doc_len)
                    idf_val = _calculate_idf(doc_freqs.get(t, 0), total_docs)
                    d_val = tf_val * idf_val
                    
                    tfidf_s += q_val * d_val # Dot product
                    
                # For doc norm, we need to iterate over all doc terms, not just query terms
                # But that's expensive. Let's approximate or use what we have.
                # Actually, let's just use the query terms for the dot product (tfidf_s)
                # and for Cosine, we need full norms.
                # Calculating full doc norm is O(|D|).
                d_norm_sq = 0.0
                for t, count in doc_counts.items():
                     tf_v = _calculate_tf(count, doc_len)
                     idf_v = _calculate_idf(doc_freqs.get(t, 0), total_docs)
                     d_val = tf_v * idf_v
                     d_norm_sq += d_val * d_val
                
                # 'tfidf_score' here is the dot product of query and doc vectors (standard for ranking if normalized)
                hit.tfidf_score = tfidf_s
                
                if query_vec_norm_sq > 0 and d_norm_sq > 0:
                    hit.cosine_similarity_score = tfidf_s / (math.sqrt(query_vec_norm_sq) * math.sqrt(d_norm_sq))
                else:
                    hit.cosine_similarity_score = 0.0

                # Dirichlet+JM Interpolation Score (Issue 4: renamed from "Two-Stage")
                # NOTE: This is NOT canonical Zhai-Lafferty two-stage. It applies Dirichlet
                # smoothing first, then JM interpolation with collection, which double-injects
                # the background model. Kept for experimental comparison.
                # Formula: P_mix(t|d) = (1-lambda) * P_dir(t|d) + lambda * P(t|C)
                ts_s = 0.0
                for t, q_tf in query_counts.items():
                    p_c = p_collection.get(t, 1e-9)
                    # Dirichlet part
                    denom = doc_len + mu if (doc_len + mu) > 0 else 1.0
                    p_dir = (doc_counts.get(t, 0) + mu * p_c) / denom
                    
                    # Interpolate (double background injection)
                    p_mix = (1 - jm_lambda) * p_dir + jm_lambda * p_c
                    
                    if p_mix > 0:
                        ts_s += q_tf * math.log(p_mix)
                hit.two_stage_score = ts_s  # Kept field name for backwards compatibility

                # Jaccard & Dice
                # Sets of terms
                q_set = set(query_counts.keys())
                d_set = set(doc_counts.keys())
                intersection = len(q_set & d_set)
                union = len(q_set | d_set)
                hit.jaccard_score = intersection / union if union > 0 else 0.0
                hit.dice_score = (2 * intersection) / (len(q_set) + len(d_set)) if (len(q_set) + len(d_set)) > 0 else 0.0

                # Exact Match Count
                # Issue 20: Use token-based check or stricter phrase count instead of raw substring count
                # which can match inside words.
                # Here we'll stick to contents.lower().count() but simpler:
                # If we want "exact match of terms", we should use Jaccard/Dice.
                # If we want "exact phrase count", substring is okay IF we have boundaries.
                # Let's count approximate phrase hits by joining query terms and checking substring,
                # but let's allow "sloppy" matching if possible? No, exact match usually means strict.
                # Better: exact match of the *full query* as a phrase.
                # To be safer, we could use regex with boundary \b.
                # But for now, let's just leave a comment and maybe verify boundaries.
                # contents.lower().count(query_lower) matches "car" in "scar".
                # Fix: using regex for bounds.
                # Fix: using regex for bounds with flexible whitespace
                query_regex = r'\b' + re.escape(query_lower).replace(r'\ ', r'\s+') + r'\b'
                hit_contents = hit.contents or ""
                try:
                     hit.exact_match_count = len(re.findall(query_regex, hit_contents.lower()))
                except Exception:
                     # Fallback if regex fails
                     hit.exact_match_count = hit_contents.lower().count(query_lower)

                # Term Overlap Score
                # Issue 21: Should weight by query TF.
                overlap_weighted = 0
                for t, q_c in query_counts.items():
                    if doc_counts.get(t, 0) > 0:
                        overlap_weighted += q_c
                q_total_tokens = sum(query_counts.values())
                hit.term_overlap_score = overlap_weighted / q_total_tokens if q_total_tokens > 0 else 0.0

            except Exception as e:
                logger.warning(f"Failed to compute scores for doc {hit.docid}: {e}")

        # Compute ranks for all metrics
        def _assign_ranks(hits_list, score_attr, rank_attr, reverse=True):
            # Filter hits that have the score
            valid_hits = [h for h in hits_list if getattr(h, score_attr) is not None]
            # Sort
            valid_hits.sort(key=lambda h: getattr(h, score_attr), reverse=reverse)
            # Assign rank
            for rank, h in enumerate(valid_hits, 1):
                setattr(h, rank_attr, rank)

        _assign_ranks(results, 'query_likelihood_score', 'query_likelihood_rank', reverse=True)
        _assign_ranks(results, 'kl_divergence_score', 'kl_divergence_rank', reverse=True) # Higher is better for -KL
        _assign_ranks(results, 'rm3_score', 'rm3_rank', reverse=True)
        _assign_ranks(results, 'jm_score', 'jm_rank', reverse=True)
        _assign_ranks(results, 'tfidf_score', 'tfidf_rank', reverse=True)
        _assign_ranks(results, 'bm25_score', 'bm25_rank', reverse=True)
        _assign_ranks(results, 'exact_match_count', 'exact_match_rank', reverse=True)
        _assign_ranks(results, 'term_overlap_score', 'term_overlap_rank', reverse=True)
        _assign_ranks(results, 'titf_score', 'titf_rank', reverse=True)
        _assign_ranks(results, 'fitf_score', 'fitf_rank', reverse=True)
        _assign_ranks(results, 'rrf_score', 'rrf_rank', reverse=True)
        _assign_ranks(results, 'fusion_score', 'fusion_rank', reverse=True)
        _assign_ranks(results, 'two_stage_score', 'two_stage_rank', reverse=True)
        _assign_ranks(results, 'cosine_similarity_score', 'cosine_similarity_rank', reverse=True)
        _assign_ranks(results, 'jaccard_score', 'jaccard_rank', reverse=True)
        _assign_ranks(results, 'dice_score', 'dice_rank', reverse=True)
        _assign_ranks(results, 'entity_boost_score', 'entity_boost_rank', reverse=True)
        _assign_ranks(results, 'entity_df_boost_score', 'entity_df_boost_rank', reverse=True)
        _assign_ranks(results, 'entity_idf_boost_score', 'entity_idf_boost_rank', reverse=True)
        _assign_ranks(results, 'booster_filter_score', 'booster_filter_rank', reverse=True)
    
    # Apply final_k cutoff if specified
    if final_k is not None and final_k < len(results):
        results = results[:final_k]
        logger.info(f"Cut results to final_k={final_k}")
    
    logger.info(f"Retrieved {len(results)} hits")
    return results, metadata