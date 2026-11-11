"""
Module 4: NLP Preprocessing Service.

Provides linguistic tokenization, lemmatization, POS tagging, noun chunk extraction,
academic abbreviation expansion, and operational action-verb extraction using spaCy.
"""

import re
import json
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import Doc

from backend.api.schemas.nlp import NLPProcessedQuestion, NLPToken
from backend.config import get_settings
from backend.utils.logger import get_logger
from backend.utils.text_utils import normalize_question_text

logger = get_logger(__name__)


# Academic and Computer Science Acronyms / Abbreviations Dictionary
DOMAIN_ABBREVIATIONS: dict[str, str] = {
    "ADT": "Abstract Data Type",
    "BST": "Binary Search Tree",
    "AVL": "Adelson-Velsky and Landis Tree",
    "BFS": "Breadth First Search",
    "DFS": "Depth First Search",
    "MST": "Minimum Spanning Tree",
    "DP": "Dynamic Programming",
    "OOP": "Object Oriented Programming",
    "DBMS": "Database Management System",
    "RDBMS": "Relational Database Management System",
    "OS": "Operating System",
    "CPU": "Central Processing Unit",
    "RAM": "Random Access Memory",
    "ROM": "Read Only Memory",
    "SQL": "Structured Query Language",
    "FIFO": "First In First Out",
    "LIFO": "Last In First Out",
    "DAG": "Directed Acyclic Graph",
    "API": "Application Programming Interface",
    "HTTP": "Hypertext Transfer Protocol",
    "TCP": "Transmission Control Protocol",
    "UDP": "User Datagram Protocol",
    "IP": "Internet Protocol",
    "LAN": "Local Area Network",
    "WAN": "Wide Area Network",
    "ALU": "Arithmetic Logic Unit",
}

# Compiled regex for exact word-boundary abbreviation lookup
ABBR_REGEXES = {
    abbr: re.compile(rf"\b{re.escape(abbr)}\b")
    for abbr in DOMAIN_ABBREVIATIONS
}

# Common exam question prefixes to strip for pure semantic embedding
QUESTION_NUMBER_PREFIX = re.compile(
    r"^(?:Q(?:uestion|\.)?\s*)?\d{1,3}\s*[\.\:\)\-]\s*",
    re.IGNORECASE,
)
MARKS_SUFFIX = re.compile(
    r"(?:\[|\()(?:\d+\s*(?:marks?|pts?|points?|M)|\d+)(?:\]|\))\s*$",
    re.IGNORECASE,
)


def _load_blooms_action_verbs() -> set[str]:
    """Load canonical action verbs from blooms_keywords.json."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "blooms_keywords.json"
    verbs = set()
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for level_info in data.get("levels", {}).values():
                for v in level_info.get("action_verbs", []):
                    verbs.add(v.lower().strip())
        except Exception:
            pass
    # Fallback / baseline verbs if file not loaded
    if not verbs:
        verbs = {
            "define", "list", "state", "name", "identify", "recall", "explain",
            "describe", "discuss", "compare", "contrast", "differentiate",
            "derive", "calculate", "compute", "solve", "design", "evaluate",
            "analyze", "illustrate", "apply", "implement", "construct",
        }
    return verbs


BLOOM_ACTION_VERBS = _load_blooms_action_verbs()


class NLPPreprocessorError(Exception):
    """Raised when NLP preprocessing encounters an unrecoverable error."""
    pass


class NLPPreprocessor:
    """
    NLP Preprocessing Engine powered by spaCy.

    Features:
    - Normalization and OCR artifact cleanup.
    - Domain abbreviation expansion with mapping preservation.
    - Lemmatization, POS tagging, and stopword filtering.
    - Action verb extraction (Bloom's taxonomy indicators).
    - Technical noun phrase chunking and keyword aggregation.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the NLPPreprocessor with a spaCy model.

        Args:
            model_name: spaCy language model name. Defaults to settings.spacy_model.
        """
        settings = get_settings()
        self.model_name = model_name or settings.spacy_model

        try:
            self.nlp = spacy.load(self.model_name)
            logger.info("spacy_model_loaded", model=self.model_name)
        except Exception as e:
            logger.error("spacy_load_failed", model=self.model_name, error=str(e))
            raise NLPPreprocessorError(
                f"Failed to load spaCy model '{self.model_name}': {e}"
            ) from e

    def expand_abbreviations(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Expand recognized academic domain abbreviations in the text.

        Args:
            text: Input string.

        Returns:
            Tuple of (expanded_text, dict_of_expansions_made).
        """
        expanded = text
        expansions_made: dict[str, str] = {}

        for abbr, pattern in ABBR_REGEXES.items():
            if pattern.search(expanded):
                full_term = DOMAIN_ABBREVIATIONS[abbr]
                expanded = pattern.sub(f"{abbr} ({full_term})", expanded)
                expansions_made[abbr] = full_term

        return expanded, expansions_made

    def clean_for_embedding(self, text: str) -> str:
        """
        Strip numbering prefixes, marks suffixes, and extraneous formatting
        to produce a clean semantic representation suitable for embeddings.
        """
        cleaned = text.strip()
        cleaned = QUESTION_NUMBER_PREFIX.sub("", cleaned)
        cleaned = MARKS_SUFFIX.sub("", cleaned)
        cleaned = normalize_question_text(cleaned)
        return cleaned

    def extract_action_verbs(self, doc: Doc) -> list[str]:
        """
        Extract primary action verbs directing the student's task.
        Considers canonical Bloom's action verbs and syntactic VERB tokens.
        """
        action_verbs: list[str] = []
        doc_text_lower = doc.text.lower()

        # Check multi-word action phrases first
        multi_word_actions = [
            "compare and contrast", "give examples", "write a program",
            "distinguish between", "find the value",
        ]
        for phrase in multi_word_actions:
            if phrase in doc_text_lower:
                action_verbs.append(phrase)

        # Check single-word tokens
        for token in doc:
            token_clean = token.text.lower().strip()
            lemma_clean = token.lemma_.lower().strip()

            # Check known Bloom's verb catalog
            if token_clean in BLOOM_ACTION_VERBS:
                if token_clean not in action_verbs:
                    action_verbs.append(token_clean)
            elif lemma_clean in BLOOM_ACTION_VERBS:
                if lemma_clean not in action_verbs:
                    action_verbs.append(lemma_clean)
            elif token.pos_ == "VERB" and not token.is_stop and len(token_clean) > 2:
                if lemma_clean not in action_verbs:
                    action_verbs.append(lemma_clean)

        return action_verbs

    def extract_keywords_and_chunks(self, doc: Doc) -> tuple[list[str], list[str]]:
        """
        Extract meaningful technical keywords and multi-word noun chunks.
        """
        # Multi-word noun chunks
        noun_chunks: list[str] = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()
            # Filter trivial chunks (single stop words or single characters)
            if len(chunk_text) > 2 and chunk_text not in noun_chunks:
                noun_chunks.append(chunk_text)

        # Individual content keywords (nouns, proper nouns, adjectives)
        keywords: list[str] = []
        for token in doc:
            if (
                token.pos_ in ("NOUN", "PROPN", "ADJ")
                and not token.is_stop
                and token.is_alpha
                and len(token.text) > 2
            ):
                lemma = token.lemma_.lower()
                if lemma not in keywords:
                    keywords.append(lemma)

        return keywords, noun_chunks

    def preprocess(
        self,
        text: str,
        expand_abbr: bool = True,
    ) -> NLPProcessedQuestion:
        """
        Perform complete linguistic preprocessing on a single question text.

        Args:
            text: Raw question text.
            expand_abbr: Whether to expand technical acronyms.

        Returns:
            NLPProcessedQuestion with tokens, lemmas, verbs, and keywords.
        """
        norm_text = normalize_question_text(text)
        expansions: dict[str, str] = {}

        if expand_abbr:
            text_for_spacy, expansions = self.expand_abbreviations(norm_text)
        else:
            text_for_spacy = norm_text

        doc = self.nlp(text_for_spacy)

        tokens: list[NLPToken] = []
        cleaned_tokens: list[str] = []

        for t in doc:
            tokens.append(
                NLPToken(
                    text=t.text,
                    lemma=t.lemma_,
                    pos=t.pos_,
                    tag=t.tag_,
                    is_stop=t.is_stop,
                    is_alpha=t.is_alpha,
                )
            )
            if not t.is_stop and t.is_alpha and len(t.text) > 1:
                cleaned_tokens.append(t.lemma_.lower())

        action_verbs = self.extract_action_verbs(doc)
        keywords, noun_chunks = self.extract_keywords_and_chunks(doc)

        entities: list[dict[str, str]] = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

        logger.debug(
            "question_preprocessed",
            token_count=len(tokens),
            cleaned_count=len(cleaned_tokens),
            keywords_count=len(keywords),
            action_verbs=action_verbs,
        )

        return NLPProcessedQuestion(
            original_text=text,
            normalized_text=norm_text,
            cleaned_tokens=cleaned_tokens,
            tokens=tokens,
            keywords=keywords,
            noun_chunks=noun_chunks,
            action_verbs=action_verbs,
            abbreviations_expanded=expansions,
            entities=entities,
        )

    # Convenience alias
    process = preprocess

    def preprocess_batch(
        self,
        texts: list[str],
        expand_abbr: bool = True,
    ) -> list[NLPProcessedQuestion]:
        """
        Batch-preprocess multiple questions efficiently using nlp.pipe.
        """
        results: list[NLPProcessedQuestion] = []
        for text in texts:
            results.append(self.preprocess(text, expand_abbr=expand_abbr))
        return results

    def structure_page_text(self, raw_text: str) -> str:
        """
        Structure unstructured OCR text from an entire page into well-organized,
        coherent academic text with demarcated headers, sections, reflowed
        grammatical question sentences, sub-parts, and standardized marks.

        Args:
            raw_text: Raw multi-line OCR text of the page.

        Returns:
            Structured, readable text for the entire page.
        """
        if not hasattr(self, "_page_structurer") or self._page_structurer is None:
            self._page_structurer = NLPPageStructurer(spacy_nlp=self.nlp)
        return self._page_structurer.structure_page(raw_text)


class NLPPageStructurer:
    """
    Linguistic Document Structuring Engine powered by spaCy.

    Converts unstructured OCR text from an entire page into well-organized,
    coherent academic text with demarcated headers, sections, reflowed
    grammatical question sentences, sub-parts, and standardized marks.
    """

    def __init__(self, spacy_nlp: Any | None = None):
        """Initialize with a loaded spaCy language model."""
        if spacy_nlp is not None:
            self.nlp = spacy_nlp
        else:
            settings = get_settings()
            self.nlp = spacy.load(settings.spacy_model)

        # Structural Demarcation Patterns
        self.sec_pat = re.compile(
            r"^(?:(?:SECTION|PART|GROUP|MODULE|UNIT)\s*[-:]*\s*([A-Z0-9]+|\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b))"
            r"|(?:(?:SECTION|PART|GROUP|MODULE|UNIT)\s*[-–]\s*([A-Z0-9]+))",
            re.IGNORECASE,
        )
        self.q_start_pat = re.compile(
            r"^(?:Q(?:uestion|\.)?\s*|\(\s*|\[\s*)?(\d{1,3})\s*[\.\:\)\-]\s*(.*)$",
            re.IGNORECASE,
        )
        self.q_standalone_pat = re.compile(
            r"^(?:(?:Q(?:uestion|\.)?\s*(\d{1,3})\s*[\.\:\)\-]?)|(?:\(?(\d{1,3})[\.\:\)\-]))\s*$",
            re.IGNORECASE,
        )
        self.sub_part_pat = re.compile(
            r"^(?:\(([a-h]|i{1,3}|iv|vi{0,3}|ix|x)\)|([a-h]|i{1,3}|iv|vi{0,3}|ix|x)[\.\)])\s*(.*)$",
            re.IGNORECASE,
        )
        self.or_pat = re.compile(
            r"^\s*(?:[-=*~_]{2,}\s*)?(?:OR|\[OR\]|\(OR\))(?:\s*[-=*~_]{2,})?\s*$",
            re.IGNORECASE,
        )
        self.pure_marks_pat = re.compile(
            r"^\s*\[?\s*(\d+)\s*(?:marks?|pts?|points?|M)\s*\]?\s*$",
            re.IGNORECASE,
        )
        self.marks_extractor = re.compile(
            r"\[?\b(\d+)\s*(?:marks?|pts?|points?|M)\b\]?",
            re.IGNORECASE,
        )

    def _normalize_ocr_spacing(self, text: str) -> str:
        """Fix OCR spacing anomalies around dots, brackets, and colons."""
        # Normalize question prefix spacing (e.g. "1 ." -> "1. ")
        t = re.sub(r"(\d+)\s*\.\s*", r"\1. ", text)
        # Normalize parentheses: "( a )" -> "(a)"
        t = re.sub(r"\(\s*([a-zA-Z0-9]+)\s*\)", r"(\1)", t)
        # Normalize marks: "[ 5 marks ]" -> "[5 Marks]"
        t = re.sub(r"\[\s*(\d+)\s*(?:marks?|pts?|points?|M)\s*\]", r"[\1 Marks]", t, flags=re.IGNORECASE)
        # Normalize colons
        t = re.sub(r"\s*:\s*", r": ", t)
        # Collapse multiple spaces
        t = re.sub(r"[^\S\n]+", " ", t)
        return t.strip()

    def _reflow_sentence(self, text_or_lines: str | list[str]) -> str:
        """
        Use spaCy linguistic analysis to reflow fragmented lines into a unified,
        grammatically coherent sentence with proper terminal punctuation.
        """
        if isinstance(text_or_lines, list):
            raw = " ".join(t.strip() for t in text_or_lines if t.strip())
        else:
            raw = text_or_lines.strip()

        if not raw:
            return ""

        # Normalize spacing and OCR artifacts
        raw = self._normalize_ocr_spacing(raw)

        # De-hyphenate broken words across lines (e.g. "algo- rithm" -> "algorithm")
        raw = re.sub(r"(\b[a-zA-Z]{2,})-\s+([a-zA-Z]{2,}\b)", r"\1\2", raw)

        # Extract marks if present to ensure they stay cleanly placed at the end
        detected_marks = None
        m_match = self.marks_extractor.search(raw)
        if m_match:
            detected_marks = m_match.group(1)
            raw = self.marks_extractor.sub("", raw).strip()

        # Parse with spaCy for sentence boundary and syntax analysis
        doc = self.nlp(raw)
        sentences: list[str] = []

        for sent in doc.sents:
            st = sent.text.strip()
            if not st:
                continue
            # Ensure capitalized start
            if len(st) > 1 and st[0].islower() and not st.startswith(("http", "www", "i.e.", "e.g.")):
                st = st[0].upper() + st[1:]
            sentences.append(st)

        combined_sentence = " ".join(sentences).strip()

        # Ensure sentence has terminal punctuation if it didn't end with marks or punctuation
        if combined_sentence and combined_sentence[-1] not in (".", "?", "!", ":", ";"):
            combined_sentence += "."

        # Re-attach standardized marks at the end
        if detected_marks:
            combined_sentence = f"{combined_sentence} [{detected_marks} Marks]"

        return combined_sentence

    def structure_page(self, raw_text: str) -> str:
        """
        Structure raw page text into a well-organized academic document.

        1. Demarcates Header metadata (University, Exam, Course, Time, Max Marks, Notes).
        2. Organizes Sections (SECTION A, PART B, etc.).
        3. Reflows Question prompts into flowing grammatical sentences.
        4. Aligns Sub-parts ((a), (b), (i)) with indentation.
        5. Formats Choice / OR separators clearly.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return ""

        header_lines: list[str] = []
        structured_sections: list[tuple[str | None, list[dict[str, Any]]]] = []
        current_sec: str | None = None
        current_q: dict[str, Any] | None = None
        questions_in_sec: list[dict[str, Any]] = []
        in_header = True

        for line in lines:
            # 1. Section Header (e.g. "SECTION A", "PART - II")
            s_m = self.sec_pat.match(line)
            if s_m:
                in_header = False
                if current_q:
                    questions_in_sec.append(current_q)
                    current_q = None
                if current_sec is not None or questions_in_sec:
                    structured_sections.append((current_sec, questions_in_sec))
                current_sec = line.strip().upper()
                questions_in_sec = []
                continue

            # 2. Choice / OR delimiter
            if self.or_pat.match(line):
                in_header = False
                if current_q:
                    questions_in_sec.append(current_q)
                    current_q = None
                questions_in_sec.append({"type": "OR"})
                continue

            # 3. Pure Marks line on its own (e.g. "[5 Marks]" or "(10M)")
            pm_m = self.pure_marks_pat.match(line)
            if pm_m and current_q is not None:
                current_q["lines"].append(f"[{pm_m.group(1)} Marks]")
                continue

            # 4. Question Start (e.g. "1. Explain...", "Q1. Define...", "[1] What...")
            q_m = self.q_start_pat.match(line)
            if q_m:
                # Prevent pure marks from matching as a question
                content = q_m.group(2).strip()
                if not self.pure_marks_pat.match(line):
                    in_header = False
                    if current_q:
                        questions_in_sec.append(current_q)
                    current_q = {
                        "type": "QUESTION",
                        "q_num": q_m.group(1),
                        "lines": [content] if content else [],
                    }
                    continue

            # 5. Standalone Question Number (e.g. "1.", "Q1")
            q_std = self.q_standalone_pat.match(line)
            if q_std:
                num = q_std.group(1) or q_std.group(2)
                if not self.pure_marks_pat.match(line):
                    in_header = False
                    if current_q:
                        questions_in_sec.append(current_q)
                    current_q = {
                        "type": "QUESTION",
                        "q_num": num,
                        "lines": [],
                    }
                    continue

            # Accumulate line into Header or Current Question
            if in_header:
                header_lines.append(self._normalize_ocr_spacing(line))
            else:
                if current_q is not None:
                    current_q["lines"].append(self._normalize_ocr_spacing(line))
                else:
                    if current_sec is None:
                        header_lines.append(self._normalize_ocr_spacing(line))
                    else:
                        # Unattached instructions or lines under section
                        questions_in_sec.append({"type": "TEXT", "text": line})

        # Finalize last question and section
        if current_q:
            questions_in_sec.append(current_q)
        if current_sec is not None or questions_in_sec:
            structured_sections.append((current_sec, questions_in_sec))

        # Reconstruct structured page text
        output_blocks: list[str] = []

        # 1. Header Block
        if header_lines:
            output_blocks.append("\n".join(header_lines))

        # 2. Sections and Questions
        for sec_name, q_list in structured_sections:
            sec_lines: list[str] = []
            if sec_name:
                sec_lines.append(f"{sec_name}\n")

            for item in q_list:
                item_type = item.get("type")
                if item_type == "OR":
                    sec_lines.append("[ OR ]\n")
                    continue
                if item_type == "TEXT":
                    sec_lines.append(f"{item.get('text', '')}\n")
                    continue

                q_num = item["q_num"]
                q_raw_lines = item["lines"]

                # Decompose into sub-parts (a, b, c) if present
                sub_parts: list[dict[str, Any]] = []
                curr_sp: dict[str, Any] | None = None
                main_prompt_lines: list[str] = []

                for l in q_raw_lines:
                    sp_m = self.sub_part_pat.match(l)
                    if sp_m:
                        if curr_sp:
                            sub_parts.append(curr_sp)
                        lbl = sp_m.group(1) or sp_m.group(2)
                        sp_content = sp_m.group(3).strip()
                        curr_sp = {
                            "label": lbl,
                            "lines": [sp_content] if sp_content else [],
                        }
                    else:
                        if curr_sp:
                            curr_sp["lines"].append(l)
                        else:
                            main_prompt_lines.append(l)

                if curr_sp:
                    sub_parts.append(curr_sp)

                main_text = self._reflow_sentence(main_prompt_lines)

                # Format question
                if sub_parts:
                    q_header = f"{q_num}. {main_text}".rstrip() if main_text else f"{q_num}."
                    sp_formatted = []
                    for sp in sub_parts:
                        sp_lbl = sp["label"].lower()
                        sp_text = self._reflow_sentence(sp["lines"])
                        sp_formatted.append(f"   ({sp_lbl}) {sp_text}")
                    sec_lines.append(q_header + "\n" + "\n".join(sp_formatted) + "\n")
                else:
                    sec_lines.append(f"{q_num}. {main_text}\n")

            if sec_lines:
                output_blocks.append("\n".join(sec_lines).strip())

        return "\n\n".join(output_blocks).strip()
