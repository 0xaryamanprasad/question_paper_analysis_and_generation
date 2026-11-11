"""
Demonstration Script for the Advanced 6-Module Question Paper Analysis Pipeline.

Executes:
- Module 1: Ingestion, Page Rendering, PaddleOCR & Mathematical Formula Recognition
- Module 2: Multimodal OCR Correction & Verification Flagging
- Module 3: Question Paper Structuring with Pydantic Schemas
- Module 4: Conceptual DSA Topic Classification (Primary/Secondary topics)
- Module 5: Validation Engine (Numbering gaps, missing subquestions, marks arithmetic, math syntax)
- Module 6: PostgreSQL Database Persistence (with SQLite local fallback)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import backend.database as db_module
from backend.database import init_db
from backend.services.pipeline_orchestrator import PipelineOrchestrator
from backend.models.question import Question
from backend.models.paper import Paper
from sqlalchemy import select


def create_sample_dsa_pdf(output_path: Path) -> Path:
    """Creates a realistic Data Structures and Algorithms exam paper PDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Text content representing an academic DSA examination
    exam_text = (
        "NATIONAL INSTITUTE OF TECHNOLOGY\n"
        "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\n"
        "END SEMESTER EXAMINATION - NOVEMBER 2024\n"
        "Course: CS301 Data Structures and Algorithms\n"
        "Time: 3 Hours                                                Max Marks: 50\n"
        "----------------------------------------------------------------------------------------\n\n"
        "SECTION A: ALGORITHM ANALYSIS & RECURRENCES\n\n"
        "1. Asymptotic Analysis and Divide & Conquer [10 Marks]\n"
        "   (a) State Master Theorem and solve the recurrence relation T(n) = 2T(n/2) + O(n). [5 Marks]\n"
        "   (b) Compare the worst-case and best-case time complexity of QuickSort: O(n^2) vs O(n log n). [5 Marks]\n\n"
        "2. Tree Structures and Balancing [10 Marks]\n"
        "   (a) Construct an AVL tree by inserting keys [15, 20, 24, 10, 13, 7] and demonstrate RR rotation. [6 Marks]\n"
        "   (b) Prove that the height of an AVL tree with n nodes satisfies h <= 1.44 log2(n). [4 Marks]\n\n"
        "SECTION B: GRAPHS & DYNAMIC PROGRAMMING\n\n"
        "3. Graph Algorithms [10 Marks]\n"
        "   (a) Trace Dijkstra's single-source shortest path algorithm on a directed graph with positive edge weights. [6 Marks]\n"
        "   (b) What is the time complexity using a min-heap priority queue: O((V + E) log V)? [4 Marks]\n\n"
        "4. Dynamic Programming Optimization [10 Marks]\n"
        "   (a) Formulate the optimal substructure and write the recurrence for the 0/1 Knapsack problem. [5 Marks]\n"
        "   (b) Compute the Longest Common Subsequence (LCS) for strings X = 'ALGORITHM' and Y = 'LOGARITHM'. [5 Marks]\n\n"
        "5. Hashing Techniques [10 Marks]\n"
        "   (a) Explain collision resolution using open addressing with quadratic probing h(k, i) = (h'(k) + c1*i + c2*i^2) mod m. [5 Marks]\n"
        "   (b) How does separate chaining maintain O(1) average lookup time under load factor alpha <= 0.75? [5 Marks]\n"
    )

    page.insert_text((40, 50), exam_text, fontsize=10, fontname="helv", color=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


async def main():
    print("=" * 80)
    print(" ADVANCED QUESTION PAPER PIPELINE DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize Database
    print("\n[DB] Initializing Database Connection...")
    await init_db()

    # 2. Generate or locate sample question paper
    sample_pdf_path = PROJECT_ROOT / "uploads" / "sample_dsa_exam_2024.pdf"
    create_sample_dsa_pdf(sample_pdf_path)
    print(f"[PDF] Created Sample Examination Paper: {sample_pdf_path}")

    # 3. Instantiate Pipeline Orchestrator
    orchestrator = PipelineOrchestrator()

    # 4. Run Complete 6-Module Pipeline
    print("\n[PIPELINE] Running Modules 1 through 6...")
    result = await orchestrator.run_pipeline(
        file_path=sample_pdf_path,
        subject="Data Structures and Algorithms",
        examination_year="2024",
    )

    # 5. Display Module-by-Module Output
    print("\n" + "=" * 80)
    print(" DEMONSTRATION RESULTS")
    print("=" * 80)

    print(f"\n[MODULE 1: OCR & Math Recognition]")
    print(f"  Total Pages Processed   : {result.ocr_result.total_pages}")
    print(f"  Average OCR Confidence  : {result.ocr_result.average_confidence:.2%}")
    print(f"  Detected LaTeX Formulas : {len(result.ocr_result.all_latex_expressions)}")
    for i, expr in enumerate(result.ocr_result.all_latex_expressions[:6], 1):
        print(f"    Formula {i}: {expr}")

    print(f"\n[MODULE 3: Question Structuring]")
    print(f"  Subject                 : {result.structured_paper.subject}")
    print(f"  Examination Year        : {result.structured_paper.examination_year}")
    print(f"  Total Extracted Marks   : {result.structured_paper.total_marks}")
    print(f"  Detected Sections       : {result.structured_paper.sections}")
    print(f"  Structured Questions    : {len(result.structured_paper.questions)}")

    print(f"\n[MODULE 4: DSA Topic Classification & MODULE 6: Stored Questions]")
    for q in result.saved_questions:
        print(f"\n  --------------------------------------------------")
        print(f"  Question #{q['question_number']} | Section: {q['section']} | Marks: {q['marks']}")
        print(f"  Primary Topic      : {q['dsa_topic']}")
        print(f"  Secondary Topic    : {q['dsa_secondary_topic']}")
        print(f"  Subtopic           : {q['subtopic']}")
        print(f"  DSA Confidence     : {q['dsa_confidence']:.2%}")
        print(f"  OCR Confidence     : {q['ocr_confidence']:.2%}")
        print(f"  LaTeX Expressions  : {q['latex_expressions']}")
        print(f"  Subquestions Count : {q['subquestions_count']}")
        print(f"  Verification Flag  : {q['is_flagged_for_verification']}")
        snippet = q['cleaned_text'].splitlines()[0] if q['cleaned_text'] else ""
        print(f"  Prompt Preview     : {snippet[:80]}...")

    print(f"\n[MODULE 5: Validation Engine Report]")
    summary = result.validation_summary
    print(f"  Paper Valid             : {summary.is_valid}")
    print(f"  Total Questions         : {summary.total_questions}")
    print(f"  Discontinuities in Seq  : {summary.numbering_discontinuities}")
    print(f"  Missing Subquestions    : {summary.missing_subquestions_count}")
    print(f"  Marks Discrepancies     : {summary.marks_discrepancies_count}")
    print(f"  Suspicious Math Formulas: {summary.suspicious_math_count}")
    print(f"  Total Validation Issues : {len(summary.issues)}")

    # 6. Verify Records Directly in Database
    print(f"\n[MODULE 6: Direct Database Verification]")
    async with db_module.async_session_factory() as session:
        db_paper = (await session.execute(
            select(Paper).where(Paper.id == result.paper_id)
        )).scalar_one_or_none()
        db_questions = (await session.execute(
            select(Question).where(Question.paper_id == result.paper_id)
        )).scalars().all()

        print(f"  Database Paper Record   : {db_paper.subject} ({db_paper.year}) [ID: {db_paper.id}]")
        print(f"  Database Question Count : {len(db_questions)} questions confirmed persisted in DB.")

    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
