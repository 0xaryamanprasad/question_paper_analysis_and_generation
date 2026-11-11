"""
Historical Question Analysis Service (Module 8).

Aggregates classified exam questions across multiple examination years to discover:
1. Topic Frequency & Longitudinal Trends (yearly counts, direction: increasing/decreasing/stable).
2. Recurring Questions & Duplicate Clusters across academic years.
3. Repetition Rate (percentage of repeated or near-verbatim questions).
4. Bloom's Taxonomy Cognitive Distribution over time.
5. High-importance 'hot topics' based on recurrence and marks allocation.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from backend.api.schemas.analysis import (
    HistoricalAnalysisResult,
    RecurrenceGroup,
    TopicTrend,
)
from backend.services.similarity_engine import SimilarityEngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class HistoricalAnalyzer:
    """
    Analyzes historical patterns across past-year question papers.
    """

    def __init__(self, similarity_engine: Optional[SimilarityEngine] = None):
        """
        Initialize the HistoricalAnalyzer.

        Args:
            similarity_engine: Optional SimilarityEngine for clustering recurring questions.
        """
        self._similarity_engine = similarity_engine

    @property
    def similarity_engine(self) -> SimilarityEngine:
        """Lazy-load SimilarityEngine if needed."""
        if self._similarity_engine is None:
            self._similarity_engine = SimilarityEngine()
        return self._similarity_engine

    def _determine_trend(self, yearly_counts: Dict[str, int], sorted_years: List[str]) -> str:
        """
        Determine trend direction (increasing, decreasing, stable, intermittent).
        """
        if len(sorted_years) < 2:
            return "stable"

        counts = [yearly_counts.get(y, 0) for y in sorted_years]
        active_years = sum(1 for c in counts if c > 0)

        if active_years <= 1:
            return "intermittent"

        # Linear regression slope over consecutive year indices
        x = np.arange(len(sorted_years), dtype=float)
        y = np.array(counts, dtype=float)

        # Variance of y
        if np.all(y == y[0]):
            return "stable"

        # Slope calculation
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return "stable"

        slope = np.sum((x - x_mean) * (y - y_mean)) / denominator

        if slope > 0.35:
            return "increasing"
        elif slope < -0.35:
            return "decreasing"
        else:
            return "stable"

    def analyze(
        self,
        questions: List[Dict[str, Any]],
        subject: str = "Unknown Subject",
        enable_clustering: bool = True,
    ) -> HistoricalAnalysisResult:
        """
        Run complete historical analysis over a set of classified questions.

        Args:
            questions: List of question records with fields:
                       id, text, year, unit_number, unit_name, topic, marks, blooms_level
            subject: Name of the academic subject.
            enable_clustering: Whether to run semantic clustering across years.

        Returns:
            HistoricalAnalysisResult containing all statistical and longitudinal insights.
        """
        if not questions:
            return HistoricalAnalysisResult(
                subject=subject,
                total_papers_analyzed=0,
                years_covered=[],
                total_questions=0,
                topic_frequency={},
                topic_trends=[],
                top_repeated_topics=[],
                blooms_distribution={},
                blooms_percentage={},
                recurring_questions=[],
                overall_repetition_rate=0.0,
            )

        # 1. Discover years and papers
        years_set: Set[str] = set()
        papers_set: Set[str] = set()

        for q in questions:
            year = str(q.get("year", "Unknown")).strip()
            if year and year != "None":
                years_set.add(year)
            paper_id = str(q.get("paper_id", "")).strip()
            if paper_id:
                papers_set.add(paper_id)

        sorted_years = sorted(list(years_set))
        total_papers = max(len(papers_set), len(sorted_years), 1)

        # 2. Topic frequency by year
        topic_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        topic_marks: Dict[str, List[float]] = defaultdict(list)
        topic_units: Dict[str, Tuple[int, str]] = {}

        for q in questions:
            topic = q.get("topic") or "General"
            year = str(q.get("year", "Unknown")).strip()
            unit_no = int(q.get("unit_number") or 1)
            unit_name = str(q.get("unit_name") or "Unit 1")
            marks = float(q.get("marks") or 0.0)

            topic_freq[topic][year] += 1
            topic_freq[topic]["total"] += 1
            topic_marks[topic].append(marks)
            topic_units[topic] = (unit_no, unit_name)

        # Convert to normal dict
        topic_frequency_dict = {
            t: dict(counts) for t, counts in topic_freq.items()
        }

        # 3. Topic Trends & Importance Scores
        total_q_count = len(questions)
        max_occurrences = max((counts["total"] for counts in topic_frequency_dict.values()), default=1)

        topic_trends: List[TopicTrend] = []
        for topic, counts in topic_frequency_dict.items():
            unit_no, unit_name = topic_units.get(topic, (1, "Unit 1"))
            total_occ = counts.get("total", 0)
            marks_list = topic_marks.get(topic, [0.0])
            avg_marks = round(float(np.mean(marks_list)), 2) if marks_list else 0.0

            yearly_breakdown = {y: counts.get(y, 0) for y in sorted_years}
            trend_dir = self._determine_trend(yearly_breakdown, sorted_years)

            # Importance score: 60% frequency weight + 40% marks weight
            norm_freq = total_occ / max_occurrences if max_occurrences > 0 else 0.0
            norm_marks = min(avg_marks / 15.0, 1.0)
            importance = round(0.60 * norm_freq + 0.40 * norm_marks, 4)

            topic_trends.append(
                TopicTrend(
                    topic=topic,
                    unit_number=unit_no,
                    unit_name=unit_name,
                    total_occurrences=total_occ,
                    yearly_counts=yearly_breakdown,
                    trend_direction=trend_dir,
                    average_marks=avg_marks,
                    importance_score=importance,
                )
            )

        # Sort topic trends descending by importance score
        topic_trends.sort(key=lambda t: t.importance_score, reverse=True)

        # Top repeated topics
        top_repeated_topics = [t.topic for t in topic_trends[:5]]

        # 4. Bloom's Taxonomy Distribution
        blooms_counts: Dict[str, int] = defaultdict(int)
        for q in questions:
            b_level = str(q.get("blooms_level") or "Understand").strip()
            # Normalize casing
            b_level = b_level.capitalize()
            blooms_counts[b_level] += 1

        blooms_pct: Dict[str, float] = {}
        for level, count in blooms_counts.items():
            blooms_pct[level] = round((count / total_q_count) * 100.0, 2)

        # 5. Semantic Recurrence & Duplicate Clustering
        recurring_groups: List[RecurrenceGroup] = []
        recurrent_question_ids: Set[str] = set()

        if enable_clustering and len(questions) > 1:
            try:
                # Prepare inputs for SimilarityEngine
                items_for_clustering = [
                    {"id": str(q.get("id", i)), "text": q.get("text", "")}
                    for i, q in enumerate(questions)
                ]

                clustering_report = self.similarity_engine.find_duplicates(
                    questions=items_for_clustering,
                )

                q_lookup = {str(q.get("id", i)): q for i, q in enumerate(questions)}

                for cluster in clustering_report.clusters:
                    q_ids = cluster.member_ids
                    cluster_years = set()
                    for q_id in q_ids:
                        orig_q = q_lookup.get(q_id)
                        if orig_q:
                            yr = str(orig_q.get("year", "")).strip()
                            if yr:
                                cluster_years.add(yr)

                    # Only count as historical recurrence if it spans >= 2 distinct years!
                    if len(cluster_years) >= 2:
                        for q_id in q_ids:
                            recurrent_question_ids.add(q_id)

                        first_q = q_lookup.get(q_ids[0], {})
                        canonical_text = cluster.canonical_question

                        recurring_groups.append(
                            RecurrenceGroup(
                                cluster_id=cluster.cluster_id,
                                canonical_question=canonical_text,
                                question_ids=q_ids,
                                years_appeared=sorted(list(cluster_years)),
                                occurrence_count=len(q_ids),
                                topic=first_q.get("topic"),
                                unit_number=first_q.get("unit_number"),
                                average_similarity=0.88,
                            )
                        )
            except Exception as e:
                logger.warning("recurrence_clustering_failed", error=str(e))

        # Overall repetition rate percentage
        repetition_rate = 0.0
        if total_q_count > 0:
            repetition_rate = round((len(recurrent_question_ids) / total_q_count) * 100.0, 2)

        return HistoricalAnalysisResult(
            subject=subject,
            total_papers_analyzed=total_papers,
            years_covered=sorted_years,
            total_questions=total_q_count,
            topic_frequency=topic_frequency_dict,
            topic_trends=topic_trends,
            top_repeated_topics=top_repeated_topics,
            blooms_distribution=dict(blooms_counts),
            blooms_percentage=blooms_pct,
            recurring_questions=recurring_groups,
            overall_repetition_rate=repetition_rate,
        )
