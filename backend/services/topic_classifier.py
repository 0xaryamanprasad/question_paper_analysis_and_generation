"""
Topic and Unit Classification Service (Module 6).

Classifies exam questions into syllabus units and topics using a multi-tier hybrid approach:
1. Exact & Lemmatized Keyword Matching: Fast, deterministic matching of domain terms
   and acronyms against syllabus keywords.
2. TF-IDF + Cosine Similarity: Lexical relevance scoring across synthesized topic profiles.
3. Semantic Vector Similarity: Dense sentence embeddings (via all-MiniLM-L6-v2) to capture
   synonymous phrasing and conceptual descriptions.

Outputs structured TopicClassificationResult schemas suitable for persistence and downstream analytics.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.api.schemas.classification import (
    SyllabusDefinition,
    TopicCandidate,
    TopicClassificationResult,
    UnitDefinition,
)
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.services.similarity_engine import SimilarityEngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYLLABUS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "syllabus" / "sample_syllabus.json"
)


def resolve_syllabus_path(subject: Optional[str] = None) -> Path:
    """Resolve the appropriate syllabus file based on subject name."""
    base = Path(__file__).resolve().parent.parent / "data" / "syllabus"
    if subject:
        sub_lower = subject.lower()
        if any(kw in sub_lower for kw in ["os", "operating system"]):
            os_path = base / "os_syllabus.json"
            if os_path.exists():
                return os_path
        elif any(kw in sub_lower for kw in ["coa", "computer organization", "architecture", "microprocessor", "pipelining", "hardware"]):
            coa_path = base / "coa_syllabus.json"
            if coa_path.exists():
                return coa_path
        elif any(kw in sub_lower for kw in ["math", "calculus", "differential", "algebra", "integral"]):
            math_path = base / "mathematics_syllabus.json"
            if math_path.exists():
                return math_path
        elif any(kw in sub_lower for kw in ["dbms", "database", "sql"]):
            dbms_path = base / "dbms_syllabus.json"
            if dbms_path.exists():
                return dbms_path
        elif any(kw in sub_lower for kw in ["dsa", "data structure", "algorithm"]):
            return DEFAULT_SYLLABUS_PATH
    return DEFAULT_SYLLABUS_PATH


def detect_subject_domain(question_text: str) -> Optional[str]:
    """Detect academic subject domain from question phrasing."""
    if not question_text:
        return None
    q_lower = question_text.lower()

    scores = {
        "Operating Systems": 0,
        "Computer Organization and Architecture": 0,
        "Mathematics - II": 0,
        "Database Management Systems": 0,
        "Data Structures and Algorithms": 0,
    }

    os_markers = [
        "deadlock", "banker", "semaphore", "mutex", "paging", "page fault", "page table",
        "page replacement", "virtual memory", "thrashing", "belady", "cpu scheduling",
        "round robin", "fcfs", "sjf", "srtf", "pcb", "context switch", "critical section",
        "dining philosophers", "readers-writers", "multithreading", "disk scheduling",
        "sstf", "scan scheduling", "tlb", "translation lookaside", "operating system",
        "segmentation", "working set", "turnaround time", "process state", "ready queue"
    ]
    coa_markers = [
        "pipelining", "pipeline", "gigabyte", "megabyte", "kilobyte", "terabyte",
        "data hazard", "structural hazard", "control hazard", "pipeline stall",
        "data forwarding", "booth", "booth's algorithm", "cache memory", "direct mapping",
        "set associative", "associative mapping", "instruction cycle", "addressing mode",
        "dma", "direct memory access", "interrupt", "bus arbitration", "control unit",
        "microprogrammed", "datapath", "alu", "superscalar", "risc", "cisc",
        "von neumann", "ieee 754", "hit ratio", "miss penalty", "memory hierarchy",
        "little endian", "big endian", "word size", "microinstruction", "hardwired control",
        "byte", "bit", "data measurement", "meaning of a gigabyte", "what is pipelining"
    ]
    math_markers = [
        "differential equation", "ode", "integrating factor", "particular integral",
        "wronskian", "variation of parameter", "undetermined coefficients",
        "power series solution", "legendre", "green's theorem", "stokes",
        "divergence theorem", "vector field", "directional derivative", "curl",
        "double integral", "line integral", "analytic function", "cauchy",
        "cauchy-riemann", "residue theorem", "laurent series", "contour integral",
        "surface integral", "exactness", "exact differential"
    ]
    dbms_markers = [
        "relational algebra", "bcnf", "3nf", "2nf", "1nf", "normalization",
        "functional dependency", "acid", "serializability", "two-phase locking",
        "2pl", "er model", "entity relationship", "sql", "lossless join",
        "b+ tree", "write-ahead log", "dbms"
    ]
    dsa_markers = [
        "quick sort", "merge sort", "avl", "binary search tree", "bst",
        "heap sort", "breadth first", "depth first", "bfs", "dfs", "dijkstra",
        "kruskal", "prim's", "linked list", "stack", "circular queue",
        "time complexity", "asymptotic", "recurrence relation", "knapsack",
        "dynamic programming", "binary tree", "data structure"
    ]

    for m in os_markers:
        if m in q_lower:
            scores["Operating Systems"] += 2 if len(m) > 6 else 1
    for m in coa_markers:
        if m in q_lower:
            scores["Computer Organization and Architecture"] += 2 if len(m) > 6 else 1
    for m in math_markers:
        if m in q_lower:
            scores["Mathematics - II"] += 2 if len(m) > 6 else 1
    for m in dbms_markers:
        if m in q_lower:
            scores["Database Management Systems"] += 2 if len(m) > 6 else 1
    for m in dsa_markers:
        if m in q_lower:
            scores["Data Structures and Algorithms"] += 2 if len(m) > 6 else 1

    best_subject, best_score = max(scores.items(), key=lambda item: item[1])
    return best_subject if best_score > 0 else None


def classify_domain_and_topic_with_gemini(question_text: str) -> Optional[Dict[str, Any]]:
    """
    Leverages Google Gemini API to dynamically classify an arbitrary examination question
    into its authentic academic subject domain, course code, unit number, unit name, and topic.
    """
    from backend.config import get_settings
    settings = get_settings()
    api_key = getattr(settings, "gemini_api_key", "")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model = getattr(settings, "gemini_model", "gemini-2.5-flash") or "gemini-2.5-flash"

        prompt = f"""You are an expert university curriculum evaluator and professor across computer science and engineering disciplines.
Analyze this examination question prompt:
Question: "{question_text}"

Classify it into:
1. Academic Subject (e.g. "Computer Organization and Architecture", "Operating Systems", "Mathematics - II", "Database Management Systems", "Data Structures and Algorithms", "Computer Networks", "Software Engineering")
2. Course Code (e.g. CS204, CS202, 23BS1004, CS203, CS201, CS205)
3. Unit Number (integer 1 to 5)
4. Unit Name (e.g. "Processor Organization & Instruction Pipelining", "Data Representation & Storage Systems", "Process Synchronization & Deadlocks", "Memory Management")
5. Specific Topic (e.g. "Instruction Pipelining and Pipeline Hazards", "Storage Hierarchy and Data Measurement Units (Bits, Bytes, Words, Megabytes, Gigabytes)", "Deadlock Prevention, Avoidance, and Banker's Algorithm", "Lossless-Join Decomposition")
6. Confidence (float 0.85 to 0.99)
7. Matched Keywords (list of strings representing key concepts identified)
8. Rationale (brief sentence explaining the concept)

Return ONLY a valid JSON object with keys:
"subject", "course_code", "unit_number", "unit_name", "topic", "confidence", "matched_keywords", "rationale".
"""
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            }
        )
        if response and response.text:
            import json
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if "subject" in data and "topic" in data:
                return data
    except Exception as e:
        logger.warning(
            "gemini_topic_classification_failed_fallback_to_local",
            error=str(e),
            note="Falling back to local heuristic/semantic classifier"
        )
    return None


class TopicClassifier:
    """
    Multi-tier topic and unit classifier for academic exam questions.

    Attributes:
        syllabus: SyllabusDefinition holding units, topics, and keywords.
        nlp_preprocessor: NLPPreprocessor instance for tokenization and cleaning.
        similarity_engine: Optional SimilarityEngine for semantic vector scoring.
    """

    def __init__(
        self,
        syllabus: Optional[Union[SyllabusDefinition, Dict[str, Any], Path, str]] = None,
        nlp_preprocessor: Optional[NLPPreprocessor] = None,
        similarity_engine: Optional[SimilarityEngine] = None,
        enable_semantic: bool = True,
    ):
        """
        Initialize the TopicClassifier with syllabus and NLP components.

        Args:
            syllabus: SyllabusDefinition, dict, or file path. If None, loads default sample syllabus.
            nlp_preprocessor: Injected NLPPreprocessor. If None, instantiates a new one.
            similarity_engine: Injected SimilarityEngine. If None and enable_semantic is True,
                               instantiates one lazily.
            enable_semantic: Whether to compute dense semantic vector similarities.
        """
        self.nlp = nlp_preprocessor or NLPPreprocessor()
        self.enable_semantic = enable_semantic
        self._similarity_engine = similarity_engine

        # Load and parse syllabus
        self.syllabus = self._load_syllabus(syllabus)

        # Build topic profiles and indices
        self.topic_profiles: List[Dict[str, Any]] = []
        self._build_topic_profiles()

        # Build TF-IDF vectorizer
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._fit_tfidf()

        # Precompute semantic embeddings for topic profiles if enabled
        self._topic_embeddings: Optional[np.ndarray] = None
        if self.enable_semantic:
            self._precompute_topic_embeddings()

        logger.info(
            "topic_classifier_initialized",
            subject=self.syllabus.subject,
            unit_count=len(self.syllabus.units),
            topic_count=len(self.topic_profiles),
            semantic_enabled=self.enable_semantic,
        )

    @property
    def similarity_engine(self) -> SimilarityEngine:
        """Lazy-load similarity engine if needed."""
        if self._similarity_engine is None:
            self._similarity_engine = SimilarityEngine()
        return self._similarity_engine

    def _load_syllabus(
        self, syllabus: Optional[Union[SyllabusDefinition, Dict[str, Any], Path, str]]
    ) -> SyllabusDefinition:
        """Load syllabus definition from model, dict, or file."""
        if syllabus is None:
            syllabus = DEFAULT_SYLLABUS_PATH

        if isinstance(syllabus, (str, Path)):
            path = Path(syllabus)
            if not path.exists():
                raise FileNotFoundError(f"Syllabus file not found at: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SyllabusDefinition(**data)

        if isinstance(syllabus, dict):
            return SyllabusDefinition(**syllabus)

        if isinstance(syllabus, SyllabusDefinition):
            return syllabus

        raise TypeError(f"Unsupported syllabus type: {type(syllabus)}")

    def _build_topic_profiles(self) -> None:
        """
        Build rich descriptive text representations for each topic in the syllabus.
        Profiles combine Unit name, Topic name, and relevant Keywords for matching.
        """
        self.topic_profiles = []
        for unit in self.syllabus.units:
            unit_keywords_lower = [k.lower() for k in unit.keywords]

            for topic in unit.topics:
                # Find keywords that appear in or associate closely with this topic
                topic_words = set(topic.lower().split())
                specific_keywords = [
                    k for k in unit.keywords if any(tw in k.lower() for tw in topic_words)
                ]
                if not specific_keywords:
                    specific_keywords = unit.keywords[:5]  # Fallback to top unit keywords

                # Create synthesized descriptive document for TF-IDF & dense embeddings
                profile_text = (
                    f"{unit.unit_name}. Topic: {topic}. "
                    f"Keywords: {', '.join(unit.keywords)}. "
                    f"Domain concepts: {' '.join(unit_keywords_lower)} {topic}"
                )

                self.topic_profiles.append(
                    {
                        "unit_number": unit.unit_number,
                        "unit_name": unit.unit_name,
                        "topic": topic,
                        "keywords": unit.keywords,
                        "unit_keywords_lower": unit_keywords_lower,
                        "profile_text": profile_text,
                    }
                )

    def _fit_tfidf(self) -> None:
        """Fit a TF-IDF vectorizer over the topic profile corpus."""
        corpus = [p["profile_text"] for p in self.topic_profiles]
        self._tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def _precompute_topic_embeddings(self) -> None:
        """Precompute dense semantic embeddings for all topic profiles."""
        try:
            texts = [
                f"{p['unit_name']}: {p['topic']}. Concepts: {', '.join(p['keywords'][:8])}"
                for p in self.topic_profiles
            ]
            self._topic_embeddings = self.similarity_engine.encode(texts)
            logger.info("topic_embeddings_precomputed", count=len(texts))
        except Exception as e:
            logger.warning(
                "topic_embeddings_precompute_failed",
                error=str(e),
                note="Falling back to keyword and TF-IDF matching",
            )
            self._topic_embeddings = None
            self.enable_semantic = False

    def _score_keywords(
        self, question_text: str, cleaned_text: str, lemmatized_tokens: List[str]
    ) -> List[Tuple[float, List[str]]]:
        """
        Tier 1: Score each topic profile based on exact and lemmatized keyword matches.
        """
        q_lower = f" {question_text.lower()} "
        cleaned_lower = f" {cleaned_text.lower()} "
        token_set = set(lemmatized_tokens)

        results = []
        for profile in self.topic_profiles:
            matched = []
            score = 0.0

            # 1. Exact topic name or constituent lemmatized words in question
            topic_lower = profile["topic"].lower()
            topic_words = [
                w for w in re.findall(r"\b[a-z]{3,}\b", topic_lower)
                if w not in {"and", "the", "with", "for"}
            ]

            if topic_lower in q_lower or topic_lower in cleaned_lower:
                matched.append(profile["topic"])
                score += 0.65
            elif topic_words and all(
                tw in token_set or tw in cleaned_lower or tw.rstrip('s') in token_set
                for tw in topic_words
            ):
                matched.append(profile["topic"])
                score += 0.55
            elif any(
                tw in token_set or f" {tw} " in cleaned_lower or tw.rstrip('s') in token_set
                for tw in topic_words if len(tw) > 3
            ):
                matched.append(profile["topic"])
                score += 0.35

            # 2. Check each keyword associated with the unit
            for kw in profile["keywords"]:
                kw_lower = kw.lower()
                # Multi-word phrase check
                if " " in kw_lower:
                    if kw_lower in q_lower or kw_lower in cleaned_lower:
                        matched.append(kw)
                        score += 0.35
                    else:
                        k_parts = [p for p in kw_lower.split() if p not in {"and", "of", "the"}]
                        if len(k_parts) > 1 and all(p in token_set or p in cleaned_lower for p in k_parts):
                            matched.append(kw)
                            score += 0.30
                else:
                    # Single word / acronym check
                    if f" {kw_lower} " in q_lower or kw_lower in token_set:
                        matched.append(kw)
                        score += 0.25

            # Cap keyword score at 1.0
            score = min(score, 1.0)
            results.append((score, list(set(matched))))

        return results

    def _score_tfidf(self, cleaned_text: str) -> np.ndarray:
        """
        Tier 2: Compute TF-IDF cosine similarity scores against all topic profiles.
        """
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return np.zeros(len(self.topic_profiles))

        q_vec = self._tfidf_vectorizer.transform([cleaned_text])
        sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        return np.nan_to_num(sims, nan=0.0)

    def _score_semantic(self, cleaned_text: str) -> np.ndarray:
        """
        Tier 3: Compute dense sentence embedding cosine similarity against topic profiles.
        """
        if not self.enable_semantic or self._topic_embeddings is None:
            return np.zeros(len(self.topic_profiles))

        try:
            q_emb = self.similarity_engine.encode([cleaned_text])
            sims = cosine_similarity(q_emb, self._topic_embeddings)[0]
            return np.nan_to_num(sims, nan=0.0)
        except Exception as e:
            logger.warning("semantic_scoring_failed", error=str(e))
            return np.zeros(len(self.topic_profiles))

    def classify_question(
        self, question_text: str, question_id: Optional[str] = None
    ) -> TopicClassificationResult:
        """
        Classify a single question text into its most likely syllabus unit and topic.

        Args:
            question_text: Raw or processed text of the question.
            question_id: Optional UUID or identifier of the question.

        Returns:
            TopicClassificationResult with best match, confidence, and candidate breakdown.
        """
        if not question_text or not question_text.strip():
            return TopicClassificationResult(
                question_id=question_id,
                question_text=question_text,
                unit_number=None,
                unit_name=None,
                topic=None,
                confidence=0.0,
                matched_keywords=[],
                classification_method="empty",
                top_candidates=[],
            )

        # Preprocess question text: expand acronyms and tokenize
        nlp_doc = self.nlp.preprocess(question_text)
        cleaned_text = self.nlp.clean_for_embedding(nlp_doc.normalized_text)
        lemmatized_tokens = nlp_doc.cleaned_tokens

        # 1. Tier 1: Keywords
        kw_scores_and_matches = self._score_keywords(
            question_text=question_text,
            cleaned_text=cleaned_text,
            lemmatized_tokens=lemmatized_tokens,
        )

        # 2. Tier 2: TF-IDF
        tfidf_scores = self._score_tfidf(cleaned_text)

        # 3. Tier 3: Semantic Embeddings
        sem_scores = self._score_semantic(cleaned_text)

        # Hybrid Score Combination
        candidates: List[TopicCandidate] = []
        for i, profile in enumerate(self.topic_profiles):
            kw_score, matched_kws = kw_scores_and_matches[i]
            tf_score = float(tfidf_scores[i])
            sem_score = float(sem_scores[i])

            if self.enable_semantic and self._topic_embeddings is not None:
                # Weights: Semantic (0.40) + Keyword (0.40) + TF-IDF (0.20)
                combined = 0.40 * sem_score + 0.40 * kw_score + 0.20 * tf_score
                method = "hybrid_semantic"
                # If strong keyword match exists (e.g. Dijkstra, BST, Quick Sort), boost
                if kw_score >= 0.50:
                    combined = max(combined, 0.70 * kw_score + 0.30 * sem_score)
            else:
                # Keyword (0.65) + TF-IDF (0.35)
                combined = 0.65 * kw_score + 0.35 * tf_score
                method = "keyword_tfidf"
                if kw_score >= 0.50:
                    combined = max(combined, kw_score)

            confidence = round(float(np.clip(combined, 0.0, 1.0)), 4)

            candidates.append(
                TopicCandidate(
                    unit_number=profile["unit_number"],
                    unit_name=profile["unit_name"],
                    topic=profile["topic"],
                    confidence=confidence,
                    matched_keywords=matched_kws,
                    method=method,
                )
            )

        # Sort candidates descending by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        best = candidates[0] if candidates else None

        all_matched_keywords = []
        if best:
            all_matched_keywords = best.matched_keywords
            # Also aggregate any keywords from top 3
            for c in candidates[:3]:
                for kw in c.matched_keywords:
                    if kw not in all_matched_keywords:
                        all_matched_keywords.append(kw)

        return TopicClassificationResult(
            question_id=question_id,
            question_text=question_text,
            subject=self.syllabus.subject,
            course_code=self.syllabus.course_code,
            unit_number=best.unit_number if best else None,
            unit_name=best.unit_name if best else None,
            topic=best.topic if best else None,
            confidence=best.confidence if best else 0.0,
            matched_keywords=all_matched_keywords,
            classification_method=best.method if best else "none",
            top_candidates=candidates[:3],
        )

    def classify_batch(
        self,
        questions: List[Dict[str, Any]],
        text_key: str = "text",
        id_key: str = "id",
    ) -> List[TopicClassificationResult]:
        """
        Classify a batch of questions.

        Args:
            questions: List of dicts containing question text and optional ID.
            text_key: Key for question text in dicts.
            id_key: Key for question ID in dicts.

        Returns:
            List of TopicClassificationResult objects.
        """
        results = []
        for q in questions:
            q_text = q.get(text_key, "")
            q_id = str(q.get(id_key, "")) if q.get(id_key) is not None else None
            res = self.classify_question(question_text=q_text, question_id=q_id)
            results.append(res)
        return results
