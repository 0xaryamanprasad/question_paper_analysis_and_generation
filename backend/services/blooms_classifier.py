"""
Bloom's Taxonomy Classification Service (Module 7).

Classifies academic questions into the 6 cognitive levels of Revised Bloom's Taxonomy:
- L1: Remember (retrieve knowledge)
- L2: Understand (construct meaning, explain concepts)
- L3: Apply (use procedure in given situation, calculate/solve)
- L4: Analyze (break down, compare, derive, verify)
- L5: Evaluate (make judgments, justify, critique)
- L6: Create (design, synthesize, propose, formulate)

Combines directive verb positioning, domain keyword indicators, sentence structure heuristics,
and calibrated confidence scoring.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.api.schemas.classification import BloomsClassificationResult, BloomsLevelEnum
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BLOOMS_KEYWORDS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "blooms_keywords.json"
)

LEVEL_CODES = {
    BloomsLevelEnum.REMEMBER: "L1",
    BloomsLevelEnum.UNDERSTAND: "L2",
    BloomsLevelEnum.APPLY: "L3",
    BloomsLevelEnum.ANALYZE: "L4",
    BloomsLevelEnum.EVALUATE: "L5",
    BloomsLevelEnum.CREATE: "L6",
}

# Specific high-priority multi-word or leading directive phrases
DIRECTIVE_PATTERNS: List[Tuple[re.Pattern, BloomsLevelEnum, float]] = [
    # Create (L6)
    (re.compile(r"\b(design|formulate|devise|synthesize|invent)\b", re.I), BloomsLevelEnum.CREATE, 0.90),
    (re.compile(r"\b(write a program|propose a solution|develop an algorithm)\b", re.I), BloomsLevelEnum.CREATE, 0.85),
    # Evaluate (L5)
    (re.compile(r"\b(evaluate|critique|appraise|judge|which is better|justify why)\b", re.I), BloomsLevelEnum.EVALUATE, 0.90),
    (re.compile(r"\b(what are the advantages and disadvantages|justify your choice)\b", re.I), BloomsLevelEnum.EVALUATE, 0.85),
    # Analyze (L4)
    (re.compile(r"\b(compare and contrast|distinguish between|differentiate between)\b", re.I), BloomsLevelEnum.ANALYZE, 0.85),
    (re.compile(r"\b(analyze the|derive the|prove that|deduce the)\b", re.I), BloomsLevelEnum.ANALYZE, 0.85),
    (re.compile(r"\b(worst-case running time|time complexity of|space complexity of)\b", re.I), BloomsLevelEnum.ANALYZE, 0.75),
    # Apply (L3)
    (re.compile(r"\b(solve the following|calculate the|compute the|find the shortest path)\b", re.I), BloomsLevelEnum.APPLY, 0.85),
    (re.compile(r"\b(construct an avl|construct a b-tree|insert the elements|trace the execution)\b", re.I), BloomsLevelEnum.APPLY, 0.85),
    (re.compile(r"\b(implement|demonstrate with|apply)\b", re.I), BloomsLevelEnum.APPLY, 0.80),
    # Understand (L2)
    (re.compile(r"\b(explain the|explain how|discuss the|describe how|illustrate with)\b", re.I), BloomsLevelEnum.UNDERSTAND, 0.80),
    (re.compile(r"\b(give examples? of|summarize|paraphrase)\b", re.I), BloomsLevelEnum.UNDERSTAND, 0.80),
    # Remember (L1)
    (re.compile(r"\b(what is (an?|the)|define the term|define|state the|list the|name the)\b", re.I), BloomsLevelEnum.REMEMBER, 0.85),
    (re.compile(r"\b(enumerate|recall|identify)\b", re.I), BloomsLevelEnum.REMEMBER, 0.80),
]


class BloomsClassifier:
    """
    Classifier for mapping question prompts to Bloom's Revised Taxonomy levels.
    """

    def __init__(
        self,
        keywords_path: Optional[Union[Path, str]] = None,
        nlp_preprocessor: Optional[NLPPreprocessor] = None,
    ):
        """
        Initialize the Bloom's Taxonomy Classifier.

        Args:
            keywords_path: Optional path to blooms_keywords.json. Defaults to standard project data.
            nlp_preprocessor: Optional NLPPreprocessor instance for linguistic analysis.
        """
        self.nlp = nlp_preprocessor or NLPPreprocessor()
        self.keywords_data = self._load_keywords(keywords_path or DEFAULT_BLOOMS_KEYWORDS_PATH)
        self.level_verbs: Dict[BloomsLevelEnum, List[str]] = self._build_level_verb_map()

        logger.info(
            "blooms_classifier_initialized",
            levels_configured=len(self.level_verbs),
        )

    def _load_keywords(self, path: Union[Path, str]) -> Dict[str, Any]:
        """Load Bloom's taxonomy keyword dictionary from file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Bloom's keywords file not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_level_verb_map(self) -> Dict[BloomsLevelEnum, List[str]]:
        """Map enum levels to lists of canonical action verbs from keywords data."""
        verb_map: Dict[BloomsLevelEnum, List[str]] = {level: [] for level in BloomsLevelEnum}

        levels_dict = self.keywords_data.get("levels", {})
        for level_str, data in levels_dict.items():
            try:
                enum_level = BloomsLevelEnum(level_str)
                verbs = [v.lower().strip() for v in data.get("action_verbs", [])]
                verb_map[enum_level] = verbs
            except ValueError:
                logger.warning("unrecognized_blooms_level_in_json", level=level_str)

        return verb_map

    def _score_question(
        self,
        question_text: str,
        action_verbs: List[str],
        cleaned_tokens: List[str],
    ) -> Tuple[Dict[BloomsLevelEnum, float], List[str], Optional[str]]:
        """
        Compute weighted scores for each Bloom's level based on verbs, patterns, and position.

        Returns:
            Tuple of (scores_dict, matched_keywords, top_rationale).
        """
        scores: Dict[BloomsLevelEnum, float] = {level: 0.0 for level in BloomsLevelEnum}
        matched_indicators: List[str] = []
        rationales: List[str] = []

        q_lower = question_text.lower().strip()

        # 1. High-priority Regex Directive Patterns
        phrase_matched_words: set[str] = set()
        for pattern, level, weight in DIRECTIVE_PATTERNS:
            match = pattern.search(q_lower)
            if match:
                matched_str = match.group(0)
                matched_indicators.append(matched_str)
                scores[level] += weight * 1.4
                rationales.append(f"Directive phrase '{matched_str}' aligns with {level.value}")
                for w in matched_str.split():
                    phrase_matched_words.add(w.lower())

        # 2. Action Verbs from NLP Processing
        # Earlier action verbs in the sentence receive higher positional weight
        for idx, verb in enumerate(action_verbs):
            verb_clean = verb.lower().strip()
            # If verb was part of an explicit directive pattern, skip conflicting single-token scoring
            if verb_clean in phrase_matched_words:
                continue

            # Positional factor: verbs in the first 3 tokens are primary directives
            pos_weight = 1.0 if idx == 0 else (0.75 if idx == 1 else 0.50)

            for level, verb_list in self.level_verbs.items():
                if verb_clean in verb_list:
                    matched_indicators.append(verb_clean)
                    verb_score = 0.60 * pos_weight
                    scores[level] += verb_score
                    rationales.append(f"Action verb '{verb_clean}' indicates {level.value}")

        # 3. Leading word direct check (e.g. starts with "Define", "Explain", "Prove")
        first_word_match = re.match(r"^([a-zA-Z]+)", q_lower)
        if first_word_match:
            first_word = first_word_match.group(1)
            if first_word not in phrase_matched_words:
                for level, verb_list in self.level_verbs.items():
                    if first_word in verb_list:
                        scores[level] += 0.40
                        if first_word not in matched_indicators:
                            matched_indicators.append(first_word)
                        rationales.append(f"Leading directive '{first_word}' strongly indicates {level.value}")

        # 4. Marks Heuristic Context (if present in text)
        marks_match = re.search(r"\[(\d+)\s*(?:marks?|m)\]", q_lower)
        if marks_match:
            marks = int(marks_match.group(1))
            if marks <= 2:
                # 1-2 marks questions are overwhelmingly Remember or Understand
                scores[BloomsLevelEnum.REMEMBER] += 0.20
                scores[BloomsLevelEnum.UNDERSTAND] += 0.15
            elif marks >= 10:
                # 10+ marks questions lean toward Analyze, Evaluate, Create
                scores[BloomsLevelEnum.ANALYZE] += 0.15
                scores[BloomsLevelEnum.CREATE] += 0.15

        top_rationale = rationales[0] if rationales else None
        return scores, list(set(matched_indicators)), top_rationale

    def classify_question(
        self, question_text: str, question_id: Optional[str] = None
    ) -> BloomsClassificationResult:
        """
        Classify a single question into Bloom's Taxonomy cognitive level.

        Args:
            question_text: Raw question prompt text.
            question_id: Optional question identifier.

        Returns:
            BloomsClassificationResult with assigned level, code, confidence, and rationale.
        """
        if not question_text or not question_text.strip():
            return BloomsClassificationResult(
                question_id=question_id,
                question_text=question_text,
                blooms_level=BloomsLevelEnum.UNDERSTAND,
                level_code="L2",
                confidence=0.0,
                action_verbs=[],
                matched_keywords=[],
                rationale="Empty question text",
            )

        # Preprocess text to extract lemmas and action verbs
        nlp_doc = self.nlp.preprocess(question_text)
        action_verbs = nlp_doc.action_verbs
        cleaned_tokens = nlp_doc.cleaned_tokens

        # Score across all 6 levels
        scores, matched_indicators, top_rationale = self._score_question(
            question_text=question_text,
            action_verbs=action_verbs,
            cleaned_tokens=cleaned_tokens,
        )

        # Determine winner
        best_level = max(scores, key=lambda l: scores[l])
        max_score = scores[best_level]

        # Calculate confidence
        total_score = sum(scores.values())
        if max_score > 0 and total_score > 0:
            # Ratio of winner score to total, smoothed and bounded
            ratio = max_score / total_score
            # Confidence scales between 0.50 (ambiguous) and 0.95 (definite)
            confidence = round(float(min(0.50 + 0.45 * ratio, 0.95)), 4)
        else:
            # Default fallback when no specific verbs matched:
            # Academic questions without explicit verbs default to Understand (L2)
            best_level = BloomsLevelEnum.UNDERSTAND
            confidence = 0.40
            top_rationale = "No explicit directive verbs detected; default conceptual question (Understand)"

        level_code = LEVEL_CODES.get(best_level, "L2")

        return BloomsClassificationResult(
            question_id=question_id,
            question_text=question_text,
            blooms_level=best_level,
            level_code=level_code,
            confidence=confidence,
            action_verbs=action_verbs,
            matched_keywords=matched_indicators,
            rationale=top_rationale or f"Classified as {best_level.value} based on structural cues",
        )

    def classify_batch(
        self,
        questions: List[Dict[str, Any]],
        text_key: str = "text",
        id_key: str = "id",
    ) -> List[BloomsClassificationResult]:
        """
        Classify a batch of questions into Bloom's taxonomy.

        Args:
            questions: List of question dicts.
            text_key: Dict key containing question text.
            id_key: Dict key containing question id.

        Returns:
            List of BloomsClassificationResult objects.
        """
        results = []
        for q in questions:
            q_text = q.get(text_key, "")
            q_id = str(q.get(id_key, "")) if q.get(id_key) is not None else None
            results.append(self.classify_question(q_text, question_id=q_id))
        return results
