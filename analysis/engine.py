"""
analysis/keyword_extractor.py — TF-IDF keyword extraction from job descriptions.
analysis/similarity_scorer.py — Cosine similarity scoring between resume and JD.

Uses scikit-learn for NLP analysis. No external API calls.
"""

import re
import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORD EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

# Known stop words to filter from keyword lists
TECH_STOPWORDS = {
    "experience", "required", "preferred", "ability", "strong", "excellent",
    "working", "knowledge", "skills", "team", "work", "must", "will",
    "including", "related", "using", "years", "year", "demonstrated",
    "proven", "great", "good", "plus", "etc", "responsible", "responsibilities",
    "duties", "role", "position", "candidate", "applicant", "company",
    "organization", "environment", "solutions", "systems", "tools", "support",
}

class KeywordExtractor:
    """
    Extracts significant technical keywords and phrases from job descriptions
    using TF-IDF scoring with n-gram support.
    """

    def __init__(self, max_keywords: int = 40, ngram_range: tuple = (1, 3)):
        self.max_keywords = max_keywords
        self.ngram_range = ngram_range

    def extract_from_jd(self, job_description: str) -> list[str]:
        """
        Extract the most significant keywords from a job description.

        Returns a list of keywords sorted by relevance score (highest first).
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
        except ImportError:
            raise ImportError(
                "scikit-learn is required for keyword extraction. "
                "Install with: pip install scikit-learn"
            )

        # Clean and normalize text
        text = self._preprocess(job_description)
        if not text:
            return []

        # Build TF-IDF vectorizer
        # We use multiple "documents" trick to get meaningful IDF:
        # split JD into sentences so IDF has variance
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if len(sentences) < 2:
            sentences = [text]

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            stop_words="english",
            max_features=500,
            lowercase=True,
            token_pattern=r"[a-zA-Z][a-zA-Z0-9+#\-\.]{1,}",  # Allow C++, .NET, etc.
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            logger.warning("TF-IDF failed on short input, falling back to frequency method")
            return self._frequency_fallback(job_description)

        # Sum TF-IDF scores across sentences and sort
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()
        keyword_scores = sorted(
            zip(feature_names, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter stopwords and return top keywords
        keywords = [
            kw for kw, score in keyword_scores
            if kw.lower() not in TECH_STOPWORDS and score > 0
        ]
        return keywords[:self.max_keywords]

    def find_missing_keywords(
        self,
        jd_keywords: list[str],
        resume_text: str,
    ) -> list[str]:
        """
        Identify keywords from the JD that are not present in the resume.
        Uses case-insensitive substring matching.
        """
        resume_lower = resume_text.lower()
        missing = []
        for kw in jd_keywords:
            # Check for the keyword as a whole word
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if not re.search(pattern, resume_lower):
                missing.append(kw)
        return missing

    def _preprocess(self, text: str) -> str:
        """Basic text cleaning before vectorization."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _frequency_fallback(self, text: str) -> list[str]:
        """Simple word frequency fallback when TF-IDF fails (very short text)."""
        from collections import Counter
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#\-\.]{2,}\b", text.lower())
        words = [w for w in words if w not in TECH_STOPWORDS]
        return [w for w, _ in Counter(words).most_common(self.max_keywords)]


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityScorer:
    """
    Computes cosine similarity between resume text and job description.
    Score range: 0.0 (no match) to 1.0 (perfect match).
    """

    def score(self, resume_text: str, job_description: str) -> float:
        """
        Compute TF-IDF cosine similarity between resume and job description.

        Returns:
            Float between 0 and 1. Values above 0.6 indicate strong alignment.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        if not resume_text.strip() or not job_description.strip():
            logger.warning("Empty text passed to similarity scorer")
            return 0.0

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = float(similarity[0][0])
            logger.debug("Similarity score: %.4f", score)
            return round(score, 4)
        except Exception as e:
            logger.error("Similarity scoring failed: %s", str(e))
            return 0.0

    def score_keywords(
        self,
        jd_keywords: list[str],
        resume_text: str,
    ) -> float:
        """
        Simpler keyword-based match score.
        Returns the fraction of JD keywords present in the resume.
        Complementary to cosine similarity.
        """
        if not jd_keywords:
            return 0.0

        resume_lower = resume_text.lower()
        matched = sum(
            1 for kw in jd_keywords
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", resume_lower)
        )
        return round(matched / len(jd_keywords), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FACADE
# ═══════════════════════════════════════════════════════════════════════════════

class ResumeAnalyzer:
    """
    Combines keyword extraction and similarity scoring into a single analysis.
    This is what the resume module calls.
    """

    def __init__(self):
        self.extractor = KeywordExtractor()
        self.scorer = SimilarityScorer()

    def analyze(self, resume_text: str, job_description: str) -> dict:
        """
        Full analysis of resume vs job description.

        Returns:
            {
                "similarity_score": 0.72,
                "keyword_match_score": 0.65,
                "jd_keywords": [...],
                "missing_keywords": [...],
                "matched_keywords": [...],
            }
        """
        logger.info("Running full resume analysis")

        tfidf_start = time.time()
        jd_keywords = self.extractor.extract_from_jd(job_description)
        tfidf_time_ms = round((time.time() - tfidf_start) * 1000, 2)

        missing_keywords = self.extractor.find_missing_keywords(jd_keywords, resume_text)
        matched_keywords = [kw for kw in jd_keywords if kw not in missing_keywords]

        similarity_start = time.time()
        similarity = self.scorer.score(resume_text, job_description)
        similarity_time_ms = round((time.time() - similarity_start) * 1000, 2)

        keyword_score = self.scorer.score_keywords(jd_keywords, resume_text)

        logger.debug(
            "span_analysis_engine | tfidf_time_ms=%.2f similarity_time_ms=%.2f jd_keywords=%d",
            tfidf_time_ms,
            similarity_time_ms,
            len(jd_keywords),
        )

        logger.info(
            "span_analysis_aggregate | tfidf_time_ms=%.2f similarity_time_ms=%.2f",
            tfidf_time_ms,
            similarity_time_ms,
        )

        return {
            "similarity_score": similarity,
            "keyword_match_score": keyword_score,
            "jd_keywords": jd_keywords,
            "missing_keywords": missing_keywords,
            "matched_keywords": matched_keywords,
            "total_jd_keywords": len(jd_keywords),
            "matched_count": len(matched_keywords),
            "missing_count": len(missing_keywords),
        }


# ─── Singletons ───────────────────────────────────────────────────────────────
keyword_extractor = KeywordExtractor()
similarity_scorer = SimilarityScorer()
resume_analyzer = ResumeAnalyzer()
