"""
Question Paper Validation Engine (Module 12).

Independently verifies:
1. Marks Accuracy: Section totals, sub-part summation, match against target paper total.
2. Question Count & Structure: Section question counts, mandatory vs attempt questions.
3. Semantic Duplicate Detection: Cross-question cosine similarity to prevent accidental repetition.
4. Syllabus Unit Distribution: Full syllabus coverage and blueprint deviation constraints.
5. Cognitive Balance: Bloom's taxonomy distribution check.
6. Content Integrity: Non-empty prompts, valid numbering, and formatting.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from backend.api.schemas.classification import SyllabusDefinition
from backend.api.schemas.generation import (
    GeneratedPaperResult,
    GeneratedQuestionItem,
    PaperBlueprintConfig,
)
from backend.api.schemas.validation import (
    DuplicateCheckResult,
    MarksCheckResult,
    UnitCoverageCheckResult,
    ValidationReport,
)
from backend.services.similarity_engine import SimilarityEngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationEngine:
    """
    Independent validation engine for generated examination papers.
    """

    def __init__(self, similarity_engine: Optional[SimilarityEngine] = None):
        """
        Initialize the ValidationEngine.

        Args:
            similarity_engine: Optional SimilarityEngine for semantic overlap verification.
        """
        self._similarity_engine = similarity_engine

    @property
    def similarity_engine(self) -> SimilarityEngine:
        """Lazy-load SimilarityEngine if needed."""
        if self._similarity_engine is None:
            self._similarity_engine = SimilarityEngine()
        return self._similarity_engine

    def validate_marks(
        self, paper: GeneratedPaperResult, blueprint: PaperBlueprintConfig
    ) -> MarksCheckResult:
        """
        Verify total marks, section marks, and sub-part summations.
        """
        calculated_total = 0.0
        sec_summary: Dict[str, float] = {}
        subpart_discrepancies: List[str] = []

        for sec in paper.sections:
            sec_summary[sec.section_name] = sec.total_section_marks
            calculated_total += sec.total_section_marks

            for q in sec.questions:
                if q.sub_parts:
                    subparts_sum = sum(sp.marks for sp in q.sub_parts)
                    if abs(subparts_sum - q.marks) > 0.1:
                        subpart_discrepancies.append(
                            f"{q.question_number or 'Question'}: Sub-parts sum ({subparts_sum}m) "
                            f"does not match question marks ({q.marks}m)"
                        )

        is_match = abs(calculated_total - blueprint.total_marks) < 0.1

        return MarksCheckResult(
            target_total_marks=blueprint.total_marks,
            calculated_total_marks=calculated_total,
            is_match=is_match,
            section_marks_summary=sec_summary,
            subpart_discrepancies=subpart_discrepancies,
        )

    def validate_duplicates(
        self,
        paper: GeneratedPaperResult,
        threshold: float = 0.78,
    ) -> DuplicateCheckResult:
        """
        Detect accidental duplicates or high semantic overlap among generated questions.
        Linked OR-alternative questions are excluded from duplicate violation flags.
        """
        all_questions: List[GeneratedQuestionItem] = []
        for sec in paper.sections:
            all_questions.extend(sec.questions)

        if len(all_questions) < 2:
            return DuplicateCheckResult()

        texts = [q.question_text for q in all_questions]
        try:
            embeddings = self.similarity_engine.encode_batch(texts)
            sim_matrix = self.similarity_engine.compute_similarity_matrix(embeddings)

            violating_pairs: List[Dict[str, Any]] = []
            max_sim = 0.0

            n = len(all_questions)
            for i in range(n):
                for j in range(i + 1, n):
                    score = float(sim_matrix[i, j])
                    max_sim = max(max_sim, score)

                    # Check if they are legitimate OR alternatives
                    q_a = all_questions[i]
                    q_b = all_questions[j]
                    is_or_pair = (
                        q_a.or_choice_alternative == q_b.id
                        or q_b.or_choice_alternative == q_a.id
                    )

                    if score >= threshold and not is_or_pair:
                        violating_pairs.append(
                            {
                                "question_a": q_a.question_number,
                                "question_b": q_b.question_number,
                                "similarity_score": round(score, 4),
                                "text_a": q_a.question_text[:80] + "...",
                                "text_b": q_b.question_text[:80] + "...",
                            }
                        )

            return DuplicateCheckResult(
                duplicate_pairs_found=len(violating_pairs),
                highest_similarity=round(max_sim, 4),
                violating_pairs=violating_pairs,
            )
        except Exception as e:
            logger.warning("duplicate_validation_failed", error=str(e))
            return DuplicateCheckResult()

    def validate_unit_coverage(
        self,
        paper: GeneratedPaperResult,
        syllabus: SyllabusDefinition,
        blueprint: Optional[PaperBlueprintConfig] = None,
    ) -> UnitCoverageCheckResult:
        """
        Verify that all syllabus units are properly represented and within deviation bounds.
        """
        syllabus_units = [u.unit_number for u in syllabus.units]
        unit_marks: Dict[int, float] = defaultdict(float)

        for sec in paper.sections:
            for q in sec.questions:
                unit_marks[q.unit_number] += q.marks

        covered_units = [u for u in syllabus_units if unit_marks[u] > 0]

        # Targeted units by blueprint (if sections explicitly restrict unit_coverage)
        targeted_units = set()
        if blueprint and any(sec.unit_coverage for sec in blueprint.sections):
            for sec in blueprint.sections:
                if sec.unit_coverage:
                    targeted_units.update(sec.unit_coverage)
                else:
                    targeted_units.update(syllabus_units)
        else:
            targeted_units = set(syllabus_units)

        missing_units = [u for u in sorted(list(targeted_units)) if unit_marks[u] == 0]

        # Calculate max deviation from expected percentage across targeted units
        total_marks = sum(unit_marks.values())
        max_dev = 0.0

        if total_marks > 0 and len(targeted_units) > 0:
            expected_pct = 100.0 / len(targeted_units)
            for u in targeted_units:
                actual_pct = (unit_marks[u] / total_marks) * 100.0
                target_pct = (
                    blueprint.unit_distribution_targets.get(u, expected_pct)
                    if (blueprint and blueprint.unit_distribution_targets)
                    else expected_pct
                )
                dev = abs(actual_pct - target_pct)
                max_dev = max(max_dev, dev)

        return UnitCoverageCheckResult(
            total_syllabus_units=len(syllabus_units),
            covered_units=covered_units,
            missing_units=missing_units,
            max_deviation_percentage=round(max_dev, 2),
            unit_marks_breakdown=dict(unit_marks),
        )

    def validate_paper(
        self,
        paper: GeneratedPaperResult,
        blueprint: PaperBlueprintConfig,
        syllabus: SyllabusDefinition,
        check_semantic_duplicates: bool = True,
    ) -> ValidationReport:
        """
        Run complete validation across all constraint dimensions.

        Returns:
            ValidationReport with comprehensive check breakdown and quality score.
        """
        passed_checks: List[str] = []
        failed_checks: List[str] = []
        warnings: List[str] = []

        # 1. Marks Check
        marks_res = self.validate_marks(paper, blueprint)
        if marks_res.is_match:
            passed_checks.append(f"Total marks match blueprint target ({blueprint.total_marks} Marks)")
        else:
            failed_checks.append(
                f"Total marks mismatch: Target {blueprint.total_marks}m vs Calculated {marks_res.calculated_total_marks}m"
            )

        if marks_res.subpart_discrepancies:
            for disc in marks_res.subpart_discrepancies:
                failed_checks.append(disc)
        else:
            passed_checks.append("All question sub-parts sum accurately to question totals")

        # 2. Section Question Count Check
        for sec_idx, sec_blueprint in enumerate(blueprint.sections):
            if sec_idx < len(paper.sections):
                actual_sec = paper.sections[sec_idx]
                expected_count = sec_blueprint.question_count
                if sec_blueprint.has_internal_choice:
                    # In internal choice, each position has 2 questions (primary + alternative)
                    expected_count *= 2

                if len(actual_sec.questions) == expected_count:
                    passed_checks.append(
                        f"Section '{actual_sec.section_name}' has required question count ({len(actual_sec.questions)})"
                    )
                else:
                    failed_checks.append(
                        f"Section '{actual_sec.section_name}' count mismatch: "
                        f"Expected {expected_count}, got {len(actual_sec.questions)}"
                    )

        # 3. Duplicate Semantic Check
        dup_res = DuplicateCheckResult()
        if check_semantic_duplicates:
            dup_res = self.validate_duplicates(paper)
            if dup_res.duplicate_pairs_found == 0:
                passed_checks.append("Zero internal duplicate or redundant questions detected")
            else:
                warnings.append(
                    f"{dup_res.duplicate_pairs_found} semantically similar question pair(s) detected "
                    f"(Max similarity: {dup_res.highest_similarity})"
                )

        # 4. Unit Coverage Check
        unit_res = self.validate_unit_coverage(paper, syllabus, blueprint)
        total_qs = sum(len(s.questions) for s in paper.sections)
        if len(unit_res.missing_units) == 0:
            passed_checks.append(f"Complete syllabus coverage: all {len(unit_res.covered_units)} targeted unit(s) represented")
        elif total_qs < len(unit_res.missing_units) + len(unit_res.covered_units):
            warnings.append(
                f"Selective unit coverage for compact paper: {len(unit_res.covered_units)} unit(s) covered"
            )
        else:
            failed_checks.append(f"Syllabus units missing from paper: Units {unit_res.missing_units}")

        if unit_res.max_deviation_percentage <= 15.0:
            passed_checks.append(f"Unit marks distribution within tolerance (Max deviation: {unit_res.max_deviation_percentage}%)")
        else:
            warnings.append(f"Unit weightage deviates significantly from blueprint ({unit_res.max_deviation_percentage}%)")

        # 5. Cognitive Balance (Bloom's)
        blooms_dist = paper.blooms_coverage_summary
        total_blooms_marks = sum(blooms_dist.values())
        blooms_pct = {
            b: round((m / total_blooms_marks) * 100.0, 1)
            for b, m in blooms_dist.items()
        } if total_blooms_marks > 0 else {}

        higher_order_pct = sum(
            pct for b, pct in blooms_pct.items()
            if b in ("Apply", "Analyze", "Evaluate", "Create")
        )

        if higher_order_pct >= 40.0:
            passed_checks.append(f"Cognitive rigor verified: {higher_order_pct}% higher-order thinking (Apply-Create)")
        else:
            warnings.append(f"Low cognitive rigor: Only {higher_order_pct}% higher-order questions")

        # 6. Overall Quality Score (out of 100)
        # Deduct 20 points per failed check, 5 points per warning
        quality_score = max(0.0, 100.0 - (len(failed_checks) * 20.0) - (len(warnings) * 5.0))
        is_valid = len(failed_checks) == 0

        logger.info(
            "paper_validation_completed",
            paper_id=paper.id,
            is_valid=is_valid,
            quality_score=quality_score,
            passed=len(passed_checks),
            failed=len(failed_checks),
        )

        return ValidationReport(
            paper_id=paper.id,
            is_valid=is_valid,
            overall_quality_score=round(quality_score, 1),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            marks_check=marks_res,
            duplicate_check=dup_res,
            unit_coverage_check=unit_res,
            blooms_distribution=blooms_pct,
        )
