"""
Marks and Unit Weightage Analysis Service (Module 9).

Calculates comprehensive examination marks distribution, unit-wise weightage percentages,
topic allocations, question-type marks breakdown, and blueprint deviations across papers.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from backend.api.schemas.analysis import (
    TopicWeightage,
    UnitWeightage,
    WeightageAnalysisResult,
)
from backend.api.schemas.classification import SyllabusDefinition
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class WeightageAnalyzer:
    """
    Computes unit and topic marks weightage statistics and blueprint comparisons.
    """

    def analyze(
        self,
        questions: List[Dict[str, Any]],
        syllabus: Optional[SyllabusDefinition] = None,
        expected_blueprint: Optional[Dict[int, float]] = None,
        subject: str = "Unknown Subject",
    ) -> WeightageAnalysisResult:
        """
        Analyze unit-wise and topic-wise marks allocations across past examination papers.

        Args:
            questions: List of question records with unit_number, unit_name, topic, marks, paper_id.
            syllabus: Optional SyllabusDefinition containing all expected units.
            expected_blueprint: Optional dict mapping unit_number -> expected weightage percentage.
            subject: Academic subject name.

        Returns:
            WeightageAnalysisResult with unit weightages, topic weightages, and blueprint deviations.
        """
        if not questions:
            return WeightageAnalysisResult(
                subject=subject,
                total_papers=0,
                total_marks_all_papers=0.0,
                average_paper_total_marks=0.0,
                unit_weightages=[],
                topic_weightages=[],
                marks_by_question_type={"short": 0.0, "medium": 0.0, "long": 0.0},
                most_weighty_units=[],
                underrepresented_units=[],
            )

        # 1. Discover papers
        papers_set: Set[str] = set()
        years_set: Set[str] = set()
        for q in questions:
            p_id = str(q.get("paper_id", "")).strip()
            if p_id:
                papers_set.add(p_id)
            yr = str(q.get("year", "")).strip()
            if yr and yr != "None":
                years_set.add(yr)

        total_papers = max(len(papers_set), len(years_set), 1)

        # 2. Total marks and question-type breakdown
        total_marks_all = 0.0
        marks_by_type = {"short": 0.0, "medium": 0.0, "long": 0.0}

        unit_marks: Dict[int, float] = defaultdict(float)
        unit_names: Dict[int, str] = {}
        unit_topics: Dict[int, Set[str]] = defaultdict(set)

        topic_marks: Dict[str, float] = defaultdict(float)
        topic_units: Dict[str, int] = {}

        for q in questions:
            marks = float(q.get("marks") or 0.0)
            unit_no = int(q.get("unit_number") or 1)
            unit_name = str(q.get("unit_name") or f"Unit {unit_no}")
            topic = str(q.get("topic") or "General")

            total_marks_all += marks
            unit_marks[unit_no] += marks
            unit_names[unit_no] = unit_name
            unit_topics[unit_no].add(topic)

            topic_marks[topic] += marks
            topic_units[topic] = unit_no

            # Classify question marks type
            if marks <= 2.0:
                marks_by_type["short"] += marks
            elif marks < 8.0:
                marks_by_type["medium"] += marks
            else:
                marks_by_type["long"] += marks

        avg_paper_marks = round(total_marks_all / total_papers, 2) if total_papers > 0 else 0.0

        # 3. Known units from syllabus (if provided)
        all_unit_numbers = set(unit_marks.keys())
        if syllabus:
            for u in syllabus.units:
                all_unit_numbers.add(u.unit_number)
                if u.unit_number not in unit_names:
                    unit_names[u.unit_number] = u.unit_name

        total_units_count = max(len(all_unit_numbers), 1)
        default_expected_pct = round(100.0 / total_units_count, 2)

        # 4. Compute Unit Weightages
        unit_weightages: List[UnitWeightage] = []
        most_weighty_units: List[int] = []
        underrepresented_units: List[int] = []

        for u_num in sorted(list(all_unit_numbers)):
            u_name = unit_names.get(u_num, f"Unit {u_num}")
            u_mark = unit_marks.get(u_num, 0.0)
            avg_u_mark = round(u_mark / total_papers, 2) if total_papers > 0 else 0.0

            actual_pct = (
                round((u_mark / total_marks_all) * 100.0, 2)
                if total_marks_all > 0
                else 0.0
            )

            exp_pct = (
                expected_blueprint.get(u_num, default_expected_pct)
                if expected_blueprint
                else default_expected_pct
            )

            dev_pct = round(actual_pct - exp_pct, 2)

            if dev_pct > 3.0:
                most_weighty_units.append(u_num)
            elif dev_pct < -3.0 or u_mark == 0.0:
                underrepresented_units.append(u_num)

            unit_weightages.append(
                UnitWeightage(
                    unit_number=u_num,
                    unit_name=u_name,
                    total_marks=u_mark,
                    average_marks_per_paper=avg_u_mark,
                    weightage_percentage=actual_pct,
                    expected_percentage=exp_pct,
                    deviation_percentage=dev_pct,
                    topics_covered=sorted(list(unit_topics.get(u_num, set()))),
                )
            )

        # 5. Compute Topic Weightages
        topic_weightages: List[TopicWeightage] = []
        for t_name, t_mark in topic_marks.items():
            t_unit = topic_units.get(t_name, 1)
            avg_t_mark = round(t_mark / total_papers, 2) if total_papers > 0 else 0.0
            t_pct = (
                round((t_mark / total_marks_all) * 100.0, 2)
                if total_marks_all > 0
                else 0.0
            )

            topic_weightages.append(
                TopicWeightage(
                    topic=t_name,
                    unit_number=t_unit,
                    total_marks=t_mark,
                    average_marks_per_paper=avg_t_mark,
                    weightage_percentage=t_pct,
                )
            )

        # Sort topics descending by total marks
        topic_weightages.sort(key=lambda t: t.total_marks, reverse=True)

        return WeightageAnalysisResult(
            subject=subject,
            total_papers=total_papers,
            total_marks_all_papers=total_marks_all,
            average_paper_total_marks=avg_paper_marks,
            unit_weightages=unit_weightages,
            topic_weightages=topic_weightages,
            marks_by_question_type=marks_by_type,
            most_weighty_units=most_weighty_units,
            underrepresented_units=underrepresented_units,
        )
