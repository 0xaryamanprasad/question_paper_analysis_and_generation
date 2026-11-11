"""
Question Paper Generation Agent (Module 11).

Assembles complete examination papers according to formal blueprint configurations:
1. Section-wise Question Generation & Formatting (Part A, Part B, Part C).
2. Internal Choice Pairing (Question OR Alternative Question) within syllabus units.
3. Syllabus Unit Coverage & Marks Weightage Balancing.
4. Bloom's Revised Taxonomy Cognitive Distribution Optimization.
5. Historical Trend Integration (prioritizing high-importance topics).
6. Cross-Question De-duplication.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from backend.api.schemas.analysis import HistoricalAnalysisResult
from backend.api.schemas.classification import SyllabusDefinition
from backend.api.schemas.generation import (
    GeneratedPaperResult,
    GeneratedQuestionItem,
    PaperBlueprintConfig,
    PaperSectionResult,
    SectionConstraint,
)
from backend.services.question_generator import QuestionGenerator
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PaperGenerator:
    """
    Agent responsible for generating full examination papers satisfying multi-dimensional constraints.
    """

    def __init__(
        self,
        question_generator: Optional[QuestionGenerator] = None,
    ):
        """
        Initialize the Paper Generator Agent.

        Args:
            question_generator: QuestionGenerator instance.
        """
        self.question_generator = question_generator or QuestionGenerator()

    def generate_paper(
        self,
        blueprint: PaperBlueprintConfig,
        syllabus: SyllabusDefinition,
        historical_analysis: Optional[HistoricalAnalysisResult] = None,
        exemplar_questions: Optional[Dict[int, List[str]]] = None,
    ) -> GeneratedPaperResult:
        """
        Generate a complete university examination question paper.

        Args:
            blueprint: PaperBlueprintConfig with sections, marks, and constraints.
            syllabus: SyllabusDefinition containing units, topics, and keywords.
            historical_analysis: Optional HistoricalAnalysisResult for trend prioritization.

        Returns:
            GeneratedPaperResult fully populated and structured.
        """
        logger.info(
            "paper_generation_started",
            subject=blueprint.subject,
            total_marks=blueprint.total_marks,
            sections=len(blueprint.sections),
        )

        units = syllabus.units
        unit_count = len(units)
        if unit_count == 0:
            raise ValueError("Syllabus must have at least one unit.")

        # Map units for fast lookup
        unit_map = {u.unit_number: u for u in units}
        available_unit_numbers = sorted(list(unit_map.keys()))

        # Prioritize topics from historical analysis if available
        hot_topics = set()
        if historical_analysis and blueprint.incorporate_historical_trends:
            hot_topics = set(historical_analysis.top_repeated_topics)

        # Standard Bloom's distribution cycle
        blooms_cycle = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

        paper_sections: List[PaperSectionResult] = []
        overall_unit_marks: Dict[int, float] = defaultdict(float)
        overall_blooms_marks: Dict[str, float] = defaultdict(float)
        all_generated_texts: List[str] = []

        global_q_counter = 1
        unit_pointer = 0
        blooms_pointer = 0

        for sec in blueprint.sections:
            section_questions: List[GeneratedQuestionItem] = []
            if sec.unit_coverage:
                target_units = [u for u in sec.unit_coverage if u in unit_map]
                if not target_units:
                    target_units = available_unit_numbers
            else:
                target_units = available_unit_numbers

            q_types = sec.allowed_question_types or ["medium"]

            for q_idx in range(sec.question_count):
                # Select unit round-robin from allowed units for this section
                unit_no = target_units[unit_pointer % len(target_units)]
                unit_pointer += 1
                curr_unit = unit_map[unit_no]

                # Select topic (prefer hot topics if available for this unit)
                chosen_topic = curr_unit.topics[0] if curr_unit.topics else curr_unit.unit_name
                for t in curr_unit.topics:
                    if t in hot_topics:
                        chosen_topic = t
                        break

                blooms_level = blooms_cycle[blooms_pointer % len(blooms_cycle)]
                blooms_pointer += 1

                # Determine question type from section constraint (cycling through allowed types)
                q_type = q_types[q_idx % len(q_types)]

                unit_exemplars = (exemplar_questions or {}).get(unit_no, [])

                # 1. Generate primary question
                q_primary = self.question_generator.generate_question(
                    subject=blueprint.subject,
                    unit_number=unit_no,
                    unit_name=curr_unit.unit_name,
                    topic=chosen_topic,
                    keywords=curr_unit.keywords,
                    blooms_level=blooms_level,
                    marks=sec.marks_per_question,
                    difficulty="medium",
                    question_type=q_type,
                    avoid_questions=all_generated_texts[-8:],
                    exemplar_questions=unit_exemplars,
                )

                q_primary.question_number = f"Q{global_q_counter}"
                all_generated_texts.append(q_primary.question_text)
                section_questions.append(q_primary)

                overall_unit_marks[unit_no] += sec.marks_per_question
                overall_blooms_marks[blooms_level] += sec.marks_per_question

                # 2. If internal choice (OR question), generate alternative question for same unit
                if sec.has_internal_choice:
                    alt_topic = (
                        curr_unit.topics[(q_idx + 1) % len(curr_unit.topics)]
                        if len(curr_unit.topics) > 1
                        else chosen_topic
                    )
                    alt_blooms = blooms_cycle[(blooms_pointer + 1) % len(blooms_cycle)]

                    q_alt = self.question_generator.generate_question(
                        subject=blueprint.subject,
                        unit_number=unit_no,
                        unit_name=curr_unit.unit_name,
                        topic=alt_topic,
                        keywords=curr_unit.keywords,
                        blooms_level=alt_blooms,
                        marks=sec.marks_per_question,
                        difficulty="medium",
                        question_type=q_type,
                        avoid_questions=all_generated_texts[-8:],
                        exemplar_questions=unit_exemplars,
                    )
                    q_alt.question_number = f"Q{global_q_counter} (OR)"
                    q_alt.or_choice_alternative = q_primary.id
                    q_primary.or_choice_alternative = q_alt.id

                    all_generated_texts.append(q_alt.question_text)
                    section_questions.append(q_alt)

                global_q_counter += 1

            # Total marks for section based on attempt count or question count
            attempt_count = sec.questions_to_attempt or sec.question_count
            sec_marks = attempt_count * sec.marks_per_question

            # Auto-generate descriptive instructions if not explicitly provided
            sec_desc = sec.description
            if not sec_desc or not sec_desc.strip():
                if sec.questions_to_attempt and sec.questions_to_attempt < sec.question_count:
                    sec_desc = (
                        f"Answer any {sec.questions_to_attempt} questions out of {sec.question_count}. "
                        f"({sec.questions_to_attempt} x {sec.marks_per_question:g} = {sec_marks:g} Marks)"
                    )
                elif sec.has_internal_choice:
                    sec_desc = (
                        f"Answer ALL questions. Each question has an internal choice. "
                        f"({sec.question_count} x {sec.marks_per_question:g} = {sec_marks:g} Marks)"
                    )
                else:
                    sec_desc = (
                        f"Answer ALL questions. "
                        f"({sec.question_count} x {sec.marks_per_question:g} = {sec_marks:g} Marks)"
                    )

            paper_sections.append(
                PaperSectionResult(
                    section_name=sec.section_name,
                    description=sec_desc,
                    questions_to_attempt=sec.questions_to_attempt,
                    total_section_marks=sec_marks,
                    questions=section_questions,
                )
            )

        paper_id = str(uuid.uuid4())
        total_questions_count = sum(len(s.questions) for s in paper_sections)

        result = GeneratedPaperResult(
            id=paper_id,
            subject=blueprint.subject,
            course_code=blueprint.course_code,
            institution_name=blueprint.institution_name,
            examination_name=blueprint.examination_name,
            total_marks=blueprint.total_marks,
            duration_hours=blueprint.duration_hours,
            sections=paper_sections,
            total_questions=total_questions_count,
            unit_coverage_summary=dict(overall_unit_marks),
            blooms_coverage_summary=dict(overall_blooms_marks),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "paper_generation_completed",
            paper_id=paper_id,
            total_questions=total_questions_count,
            unit_coverage=list(overall_unit_marks.keys()),
        )

        return result
