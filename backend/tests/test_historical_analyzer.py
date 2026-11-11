"""
Unit and Integration Tests for Module 8: Historical Question Analysis.
"""

import pytest

from backend.services.historical_analyzer import HistoricalAnalyzer
from backend.services.similarity_engine import SimilarityEngine


@pytest.fixture(scope="module")
def similarity_engine():
    return SimilarityEngine()


@pytest.fixture(scope="module")
def historical_analyzer(similarity_engine):
    return HistoricalAnalyzer(similarity_engine=similarity_engine)


@pytest.fixture
def sample_multi_year_questions():
    """Synthetic realistic question dataset spanning 3 academic years (2021, 2022, 2023)."""
    return [
        # Year 2021
        {
            "id": "2021_q1",
            "paper_id": "p_2021",
            "year": "2021",
            "text": "What is an Abstract Data Type (ADT)? Explain with an example of stack.",
            "unit_number": 1,
            "unit_name": "Introduction to Data Structures",
            "topic": "Abstract Data Types",
            "marks": 5.0,
            "blooms_level": "Understand",
        },
        {
            "id": "2021_q2",
            "paper_id": "p_2021",
            "year": "2021",
            "text": "Explain Dijkstra's algorithm to compute the shortest path on a weighted graph.",
            "unit_number": 3,
            "unit_name": "Graphs",
            "topic": "Shortest path algorithms (Dijkstra, Bellman-Ford)",
            "marks": 10.0,
            "blooms_level": "Apply",
        },
        {
            "id": "2021_q3",
            "paper_id": "p_2021",
            "year": "2021",
            "text": "Define Big-O notation with asymptotic graphs.",
            "unit_number": 5,
            "unit_name": "Algorithm Design and Analysis",
            "topic": "Asymptotic notations (Big-O, Omega, Theta)",
            "marks": 5.0,
            "blooms_level": "Remember",
        },
        # Year 2022
        {
            "id": "2022_q1",
            "paper_id": "p_2022",
            "year": "2022",
            "text": "Define the term Abstract Data Type (ADT) with suitable stack examples.",  # Semantic duplicate of 2021_q1!
            "unit_number": 1,
            "unit_name": "Introduction to Data Structures",
            "topic": "Abstract Data Types",
            "marks": 5.0,
            "blooms_level": "Remember",
        },
        {
            "id": "2022_q2",
            "paper_id": "p_2022",
            "year": "2022",
            "text": "Describe the AVL tree rotation rules and construct an AVL tree.",
            "unit_number": 2,
            "unit_name": "Trees",
            "topic": "AVL Trees",
            "marks": 10.0,
            "blooms_level": "Apply",
        },
        {
            "id": "2022_q3",
            "paper_id": "p_2022",
            "year": "2022",
            "text": "Explain Dijkstra's shortest path algorithm with an example.",  # Semantic duplicate of 2021_q2!
            "unit_number": 3,
            "unit_name": "Graphs",
            "topic": "Shortest path algorithms (Dijkstra, Bellman-Ford)",
            "marks": 10.0,
            "blooms_level": "Understand",
        },
        # Year 2023
        {
            "id": "2023_q1",
            "paper_id": "p_2023",
            "year": "2023",
            "text": "What is an Abstract Data Type (ADT)? Explain with stack implementation.",  # Recurrent again in 2023!
            "unit_number": 1,
            "unit_name": "Introduction to Data Structures",
            "topic": "Abstract Data Types",
            "marks": 5.0,
            "blooms_level": "Understand",
        },
        {
            "id": "2023_q2",
            "paper_id": "p_2023",
            "year": "2023",
            "text": "Compare and contrast Quick Sort and Merge Sort in worst-case time complexity.",
            "unit_number": 4,
            "unit_name": "Sorting and Searching",
            "topic": "Quick Sort",
            "marks": 10.0,
            "blooms_level": "Analyze",
        },
        {
            "id": "2023_q3",
            "paper_id": "p_2023",
            "year": "2023",
            "text": "Explain Dijkstra algorithm to find shortest path in a graph.",  # Recurrent in 2023!
            "unit_number": 3,
            "unit_name": "Graphs",
            "topic": "Shortest path algorithms (Dijkstra, Bellman-Ford)",
            "marks": 10.0,
            "blooms_level": "Apply",
        },
    ]


class TestHistoricalAnalyzer:
    """Test suite for HistoricalAnalyzer service."""

    def test_empty_analysis(self, historical_analyzer: HistoricalAnalyzer):
        """Should return safe default structure when no questions are passed."""
        res = historical_analyzer.analyze([], subject="DSA")
        assert res.subject == "DSA"
        assert res.total_questions == 0
        assert res.total_papers_analyzed == 0
        assert res.years_covered == []
        assert res.overall_repetition_rate == 0.0

    def test_year_and_paper_discovery(
        self, historical_analyzer: HistoricalAnalyzer, sample_multi_year_questions
    ):
        """Should detect distinct years and papers correctly."""
        res = historical_analyzer.analyze(sample_multi_year_questions, subject="DSA")
        assert res.years_covered == ["2021", "2022", "2023"]
        assert res.total_papers_analyzed == 3
        assert res.total_questions == 9

    def test_topic_frequency_aggregation(
        self, historical_analyzer: HistoricalAnalyzer, sample_multi_year_questions
    ):
        """Should aggregate topic frequencies by year and total occurrences."""
        res = historical_analyzer.analyze(sample_multi_year_questions, subject="DSA")
        topic_freq = res.topic_frequency

        adt_topic = "Abstract Data Types"
        assert adt_topic in topic_freq
        assert topic_freq[adt_topic]["2021"] == 1
        assert topic_freq[adt_topic]["2022"] == 1
        assert topic_freq[adt_topic]["2023"] == 1
        assert topic_freq[adt_topic]["total"] == 3

    def test_blooms_taxonomy_distribution(
        self, historical_analyzer: HistoricalAnalyzer, sample_multi_year_questions
    ):
        """Should aggregate counts and percentages of Bloom's levels accurately."""
        res = historical_analyzer.analyze(sample_multi_year_questions, subject="DSA")
        blooms_dist = res.blooms_distribution
        blooms_pct = res.blooms_percentage

        # Total 9 questions
        assert sum(blooms_dist.values()) == 9
        assert "Understand" in blooms_dist
        assert "Apply" in blooms_dist
        assert "Remember" in blooms_dist
        assert "Analyze" in blooms_dist

        # Percentages sum approximately to 100%
        assert 99.0 <= sum(blooms_pct.values()) <= 101.0

    def test_repetition_rate_and_recurrent_clusters(
        self, historical_analyzer: HistoricalAnalyzer, sample_multi_year_questions
    ):
        """Should identify questions repeated across multiple years and compute repetition rate."""
        res = historical_analyzer.analyze(
            sample_multi_year_questions, subject="DSA", enable_clustering=True
        )

        # ADT and Dijkstra appeared in all 3 years
        assert len(res.recurring_questions) >= 1
        assert res.overall_repetition_rate > 20.0

        # Verify cluster structure
        for group in res.recurring_questions:
            assert len(group.years_appeared) >= 2
            assert group.occurrence_count >= 2
            assert len(group.question_ids) >= 2

    def test_topic_trends_and_top_repeated(
        self, historical_analyzer: HistoricalAnalyzer, sample_multi_year_questions
    ):
        """Should compute trend directions and identify top repeated topics."""
        res = historical_analyzer.analyze(sample_multi_year_questions, subject="DSA")
        assert len(res.topic_trends) > 0
        assert len(res.top_repeated_topics) > 0

        # ADT and Dijkstra should be among the top repeated topics
        top_topics_str = " ".join(res.top_repeated_topics)
        assert "Abstract Data Types" in top_topics_str or "Dijkstra" in top_topics_str
