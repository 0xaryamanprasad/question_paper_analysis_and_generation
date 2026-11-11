"""
Unit and Integration Tests for Module 4: NLP Preprocessing Service.

Tests:
- Tokenization, lemmatization, and POS tagging
- Academic abbreviation expansion (ADT, BST, DFS, BFS)
- Cognitive action verb extraction (Bloom's taxonomy cues)
- Technical noun phrase chunking and content keywords
- Preprocessing for embeddings (stripping Q numbers & marks)
- Batch preprocessing
- Integration between QuestionDetector (Module 3) and NLPPreprocessor (Module 4)
"""

import pytest

from backend.api.schemas.nlp import NLPProcessedQuestion
from backend.services.nlp_preprocessor import (
    NLPPreprocessor,
    NLPPreprocessorError,
)
from backend.services.question_detector import QuestionDetector


@pytest.fixture
def preprocessor() -> NLPPreprocessor:
    return NLPPreprocessor()


class TestNLPPreprocessor:
    """Tests for NLPPreprocessor service."""

    def test_basic_tokenization_and_lemmas(self, preprocessor: NLPPreprocessor):
        """Should tokenize and lemmatize accurately without stopwords."""
        text = "Explain the fundamental principles of Object Oriented Programming."
        res = preprocessor.preprocess(text, expand_abbr=False)

        assert isinstance(res, NLPProcessedQuestion)
        assert res.original_text == text
        assert len(res.tokens) > 5

        # Check non-stop lemmas
        assert "explain" in res.cleaned_tokens
        assert "fundamental" in res.cleaned_tokens
        assert "principle" in res.cleaned_tokens
        # Stopwords should be filtered from cleaned_tokens
        assert "the" not in res.cleaned_tokens
        assert "of" not in res.cleaned_tokens

    def test_abbreviation_expansion(self, preprocessor: NLPPreprocessor):
        """Should detect and expand known domain acronyms while preserving mapping."""
        text = "Explain the difference between BST and AVL trees for search operations."
        res = preprocessor.preprocess(text, expand_abbr=True)

        assert "BST" in res.abbreviations_expanded
        assert res.abbreviations_expanded["BST"] == "Binary Search Tree"
        assert "AVL" in res.abbreviations_expanded
        assert "Adelson-Velsky and Landis Tree" in res.abbreviations_expanded["AVL"]

    def test_action_verbs_extraction(self, preprocessor: NLPPreprocessor):
        """Should accurately extract operational action verbs directing the question."""
        text = "Compare and contrast Merge Sort and Quick Sort algorithms. Derive their complexities."
        res = preprocessor.preprocess(text)

        verbs = res.action_verbs
        assert "compare" in verbs
        assert "contrast" in verbs
        assert "derive" in verbs

    def test_keywords_and_noun_chunks(self, preprocessor: NLPPreprocessor):
        """Should extract domain technical keywords and multi-word noun phrases."""
        text = "Construct a minimum spanning tree using Kruskal's algorithm on a weighted graph."
        res = preprocessor.preprocess(text)

        assert any("minimum spanning tree" in chunk for chunk in res.noun_chunks)
        assert "graph" in res.keywords
        assert "algorithm" in res.keywords

    def test_clean_for_embedding(self, preprocessor: NLPPreprocessor):
        """Should strip question numbers, bullet labels, and marks for semantic embedding."""
        raw_q = "Q. 12 - Explain Dijkstra's shortest path algorithm. [10 Marks]"
        cleaned = preprocessor.clean_for_embedding(raw_q)

        assert not cleaned.startswith("Q. 12")
        assert not cleaned.startswith("12")
        assert "[10 Marks]" not in cleaned
        assert "Explain Dijkstra's shortest path algorithm." in cleaned

    def test_batch_preprocessing(self, preprocessor: NLPPreprocessor):
        """Should batch-process a collection of questions."""
        texts = [
            "Define Abstract Data Type.",
            "Explain Breadth First Search.",
            "What is a Hash Table?",
        ]
        results = preprocessor.preprocess_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, NLPProcessedQuestion) for r in results)
        assert results[0].action_verbs == ["define"]
        assert results[1].action_verbs == ["explain"]

    def test_invalid_model_name(self):
        """Should raise NLPPreprocessorError when spaCy model cannot be loaded."""
        with pytest.raises(NLPPreprocessorError):
            NLPPreprocessor(model_name="non_existent_model_xyz")

    def test_structure_page_text_full_exam(self, preprocessor: NLPPreprocessor):
        """Should structure unstructured OCR lines into a coherent document with reflowed sentences."""
        messy_ocr_page = """
        ANTIGRAVITY UNIVERSITY
        END SEMESTER DEGREE EXAMINATIONS 2023
        Course: CS201 Data Structures
        Time: 3 Hours Max Marks: 100
        Instructions: Answer all questions.

        SECTION A

        1.
        Define an Abstract Data Type (ADT)
        with two examples.
        [5 Marks]

        2. What is an AVL Tree? State
        its advantage over standard BST.
        [5 Marks]
        """
        structured = preprocessor.structure_page_text(messy_ocr_page)

        assert "ANTIGRAVITY UNIVERSITY" in structured
        assert "SECTION A" in structured
        # Lines should be reflowed into complete, continuous sentences
        assert "1. Define an Abstract Data Type (ADT) with two examples. [5 Marks]" in structured
        assert "2. What is an AVL Tree? State its advantage over standard BST. [5 Marks]" in structured

    def test_structure_page_sub_parts_and_or_choices(self, preprocessor: NLPPreprocessor):
        """Should format sub-parts with indentation and align OR choices clearly."""
        raw_text = """
        SECTION B

        3. (a) Explain Dijkstra's shortest path
        algorithm with an example graph.
        [10 Marks]
        (b) Calculate the minimum spanning tree
        using Kruskal's algorithm.
        [10 Marks]

        OR

        4. Explain Prim's algorithm for finding
        Minimum Spanning Tree.
        [20 Marks]
        """
        structured = preprocessor.structure_page_text(raw_text)

        assert "SECTION B" in structured
        assert "(a) Explain Dijkstra's shortest path algorithm with an example graph. [10 Marks]" in structured
        assert "(b) Calculate the minimum spanning tree using Kruskal's algorithm. [10 Marks]" in structured
        assert "[ OR ]" in structured
        assert "4. Explain Prim's algorithm for finding Minimum Spanning Tree. [20 Marks]" in structured

    def test_structure_page_empty_and_minimal(self, preprocessor: NLPPreprocessor):
        """Should handle empty or single-line text gracefully."""
        assert preprocessor.structure_page_text("") == ""
        assert preprocessor.structure_page_text("   \n\n  ") == ""
        minimal = preprocessor.structure_page_text("1. Explain Quick Sort algorithm.")
        assert "1. Explain Quick Sort algorithm." in minimal


class TestDetectorToNLPIntegration:
    """Integration test: Module 3 (Question Detector) -> Module 4 (NLP Preprocessor)."""

    def test_segment_and_preprocess_pipeline(self, preprocessor: NLPPreprocessor):
        """Extracted questions from detector should flow directly through NLP pipeline."""
        raw_exam = """
        SECTION A
        1. Define Abstract Data Type (ADT). Give two examples. [5 Marks]
        2. Explain Dijkstra's algorithm for Single Source Shortest Path. [10 Marks]
        """
        detector = QuestionDetector()
        seg_res = detector.segment_text(raw_exam)

        assert seg_res.total_questions == 2

        # Process through NLP
        nlp_results = [
            preprocessor.preprocess(q.cleaned_text)
            for q in seg_res.questions
        ]

        assert len(nlp_results) == 2
        # Q1 checks
        q1_nlp = nlp_results[0]
        assert "define" in q1_nlp.action_verbs
        assert "ADT" in q1_nlp.abbreviations_expanded
        assert "Abstract Data Type" in q1_nlp.abbreviations_expanded["ADT"]

        # Q2 checks
        q2_nlp = nlp_results[1]
        assert "explain" in q2_nlp.action_verbs
        assert any("algorithm" in k for k in q2_nlp.keywords)
