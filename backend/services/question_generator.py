"""
Question Generation Agent (Module 10).

Leverages Generative AI (Google Gemini API via google-genai) and domain knowledge to generate
authentic, syllabus-compliant examination questions based on:
1. Syllabus Unit, Topic, and Keywords
2. Bloom's Taxonomy Cognitive Level & Canonical Action Verbs
3. Target Marks and Question Type (short, medium, long, sub-parts)
4. Historical Recurrence & Topic Weightage Patterns
5. Strict De-duplication against existing/prior-year questions

Includes intelligent fallback generation for offline and air-gapped test environments.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from backend.api.schemas.generation import GeneratedQuestionItem, SubPartItem
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# Standard Bloom's verbs for high-quality academic prompts
BLOOMS_PROMPT_VERBS = {
    "Remember": ["Define", "State", "List", "Identify", "Enumerate", "Describe"],
    "Understand": ["Explain", "Discuss", "Illustrate with a diagram", "Summarize", "Compare"],
    "Apply": ["Calculate", "Compute", "Solve", "Construct", "Implement", "Demonstrate"],
    "Analyze": ["Compare and contrast", "Analyze the time complexity of", "Derive", "Differentiate between", "Prove that"],
    "Evaluate": ["Critique", "Justify why", "Evaluate the efficiency of", "Assess the trade-offs between", "Recommend"],
    "Create": ["Design an algorithm to", "Devise a data structure for", "Formulate a solution to", "Write a complete program to"],
}


class QuestionGenerator:
    """
    AI-powered question generation agent using Google Gemini GenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the Question Generation Agent.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY from environment or settings.
            model_name: Model name to use (e.g. 'gemini-2.5-flash', 'gemini-1.5-flash').
        """
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "gemini_api_key", "") or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or getattr(settings, "gemini_model", "gemini-2.5-flash") or "gemini-2.5-flash"
        self._client = None

        if self.api_key:
            try:
                import google.genai as genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("gemini_client_initialized", model=self.model_name)
            except Exception as e:
                logger.warning("gemini_init_warning", error=str(e), note="Fallback mode active")

    @property
    def is_gemini_active(self) -> bool:
        """Whether authentic Gemini API calls are available."""
        return self._client is not None

    def _build_system_prompt(self, subject: str) -> str:
        """Create authoritative examination board persona prompt."""
        is_math = any(kw in subject.lower() for kw in ["math", "calculus", "differential", "algebra", "integral", "analysis"])
        math_directive = (
            " When authoring mathematics questions, ensure equations, integrals, derivatives, and formulas are formatted in standard LaTeX ($...$ for inline, $$...$$ for display). "
            "Questions must be mathematically precise, soluble, and adhere strictly to undergraduate university mathematics curricula."
            if is_math else ""
        )
        return (
            f"You are a distinguished university professor and senior examiner in {subject}. "
            f"Your task is to author rigorous, unambiguous, syllabus-compliant university examination questions.{math_directive} "
            "Format the output strictly as valid JSON adhering precisely to the requested schema. "
            "Do not include Markdown formatting or code fences."
        )

    def _build_user_prompt(
        self,
        subject: str,
        unit_number: int,
        unit_name: str,
        topic: str,
        keywords: List[str],
        blooms_level: str,
        marks: float,
        difficulty: str,
        question_type: str,
        avoid_questions: Optional[List[str]] = None,
        historical_context: Optional[str] = None,
        exemplar_questions: Optional[List[str]] = None,
    ) -> str:
        """Format detailed instruction prompt for Gemini."""
        verbs = BLOOMS_PROMPT_VERBS.get(blooms_level, ["Explain", "Analyze"])
        avoid_str = ""
        if avoid_questions:
            avoid_list_formatted = "\n- ".join(avoid_questions[:5])
            avoid_str = (
                f"\nCRITICAL REQUIREMENT - DO NOT duplicate or closely paraphrase any of these questions:\n"
                f"- {avoid_list_formatted}\n"
            )

        exemplar_str = ""
        if exemplar_questions:
            clean_exemplars = [re.sub(r"\s+", " ", ex.strip()) for ex in exemplar_questions[:4] if len(ex.strip()) > 8]
            if clean_exemplars:
                exemplar_str = (
                    f"\nAUTHENTIC REFERENCE QUESTIONS FROM PREVIOUS-YEAR EXAMINATIONS IN THIS SUBJECT ({subject}):\n"
                    + "\n".join(f"- {ex}" for ex in clean_exemplars)
                    + f"\nCRITICAL INSTRUCTION: Model the question's academic tone, mathematical rigor, and difficulty after the exemplar questions above. Do NOT copy them verbatim; create an original, non-duplicate variation.\n"
                )

        hist_str = f"\nHistorical Trend Note: {historical_context}\n" if historical_context else ""

        return (
            f"Subject: {subject}\n"
            f"Unit {unit_number}: {unit_name}\n"
            f"Topic: {topic}\n"
            f"Keywords: {', '.join(keywords[:8])}\n"
            f"Bloom's Taxonomy Cognitive Level: {blooms_level} (Use directives like: {', '.join(verbs[:3])})\n"
            f"Target Total Marks: {marks}\n"
            f"Difficulty: {difficulty}\n"
            f"Question Type: {question_type}\n"
            f"{hist_str}"
            f"{exemplar_str}"
            f"{avoid_str}\n"
            f"Generate a university-level question for this topic.\n"
            f"If marks >= 8, divide the question into sub-parts (e.g. (a) and (b)) whose marks sum exactly to {marks}.\n"
            f"Provide a marking scheme breakdown and a concise model answer.\n\n"
            f"Output must be a JSON object with this exact structure:\n"
            f"{{\n"
            f'  "question_text": "Complete question text",\n'
            f'  "sub_parts": [\n'
            f'    {{"label": "a", "text": "sub-question prompt", "marks": {marks / 2 if marks >= 8 else marks}}},\n'
            f'    {{"label": "b", "text": "sub-question prompt", "marks": {marks / 2 if marks >= 8 else 0}}}\n'
            f"  ],\n"
            f'  "marking_scheme": ["Step 1 breakdown", "Step 2 breakdown"],\n'
            f'  "sample_answer": "Concise model solution outline"\n'
            f"}}"
        )

    def _generate_fallback_question(
        self,
        subject: str,
        unit_number: int,
        unit_name: str,
        topic: str,
        keywords: List[str],
        blooms_level: str,
        marks: float,
        difficulty: str,
        question_type: str,
        exemplar_questions: Optional[List[str]] = None,
        avoid_questions: Optional[List[str]] = None,
    ) -> GeneratedQuestionItem:
        """
        Rule-based academic question synthesizer for offline / testing execution.
        Generates genuine syllabus-aligned questions without placeholders.
        """
        verb_list = BLOOMS_PROMPT_VERBS.get(blooms_level, ["Explain", "Describe"])
        chosen_verb = verb_list[0]
        secondary_verb = verb_list[1] if len(verb_list) > 1 else "discuss"

        kw_sample = keywords[0] if keywords else topic
        kw_secondary = keywords[1] if len(keywords) > 1 else topic

        is_math = any(kw in subject.lower() for kw in ["math", "calculus", "differential", "algebra", "integral", "analysis"])

        sub_parts: List[SubPartItem] = []
        marking_scheme: List[str] = []

        # Avoid list check for distinct alternative choice generation
        avoid_str = " ".join(avoid_questions or []).lower()

        # Detect if generating an internal choice alternative against an existing primary question
        is_alternate_choice = any(marker in avoid_str for marker in [
            "possesses an integrating factor",
            "variation of parameters",
            "apply green's theorem",
            "analytic everywhere",
            "residue theorem",
            "circular queue",
            "avl tree by sequentially",
            "dijkstra's single source",
            "partition subroutine in quick sort",
        ])

        if marks >= 8.0:
            # Composite question with (a) and (b)
            marks_a = round(marks * 0.5, 1) if round(marks * 0.5, 1).is_integer() else round(marks * 0.4, 1)
            marks_b = round(marks - marks_a, 1)

            if is_math:
                t_lower = topic.lower()
                if "differential" in t_lower or "ode" in t_lower or "exact" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Find the general solution of the differential equation $y'' + 2y' + 4y = \\cos 4x$ by using the method of undetermined coefficients."
                        part_b_text = "Solve the initial value problem $x^2 y'' - 3x y' + 4y = 0$, with initial conditions $y(1) = 1, y'(1) = 1$."
                    else:
                        part_a_text = "Solve the differential equation $(x^2 + y^2 + 2x)dx + 2y dy = 0$ by showing that it possesses an integrating factor $e^x$."
                        part_b_text = "Solve the differential equation $y'' - 2y' + y = x e^x \\log x, x > 0$ by using the method of variation of parameters."
                elif "power series" in t_lower or "legendre" in t_lower or "undetermined" in t_lower or "wronskian" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Show that the Legendre polynomial satisfies the orthogonality relation $\\int_{-1}^1 P_m(x)P_n(x)dx = 0$ for $m \\neq n$."
                        part_b_text = "Find the general solution of the Cauchy-Euler equation $x^2 y'' - 4xy' + 6y = x^3$ for $x > 0$."
                    else:
                        part_a_text = "Find the power series solution of the differential equation $y'' - 2y' + y = 0$ about $x = 0$."
                        part_b_text = "Solve the initial value problem $x^2 y'' - 3x y' + 4y = 0$, with initial conditions $y(1) = 1, y'(1) = 1$."
                elif "green" in t_lower or "vector" in t_lower or "surface" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Verify Stokes' theorem for $\\vec{F} = (2x - y)\\hat{i} - yz^2\\hat{j} - y^2z\\hat{k}$ over the upper hemisphere $x^2 + y^2 + z^2 = 1, z \\geq 0$."
                        part_b_text = "Find the directional derivative of $f(x, y) = \\ln(x^2 + 3y)$ at the point $(1, 1)$ in the direction of $\\vec{v} = \\hat{i} + \\hat{j}$."
                    else:
                        part_a_text = "Apply Green's theorem to evaluate $\\oint_C (2x^2 - y^2)dx + (x^2 + y^2)dy$, where $C$ is the boundary of the surface enclosed by the $x$-axis and the semi-circle $y = \\sqrt{4 - x^2}$."
                        part_b_text = "Evaluate $\\iint_R (x^2 + y^2) dx dy$, where $R$ is the triangular region in the first quadrant bounded by $x = 0, y = 0, x + y = 1$."
                elif "analytic" in t_lower or "cauchy" in t_lower or "complex" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Determine the harmonic conjugate $v(x, y)$ of $u(x, y) = x^2 - y^2 + 2x$ such that $f(z) = u + iv$ is an entire analytic function."
                        part_b_text = "Evaluate the line integral $\\oint_{|z|=3} \\frac{z + i}{z^2 + 4iz - 4} dz$."
                    else:
                        part_a_text = "Show that the function $f(z) = e^x(\\cos y + i \\sin y)$ is analytic everywhere and determine its complex derivative $f'(z)$ in terms of $z$."
                        part_b_text = "Using Cauchy's Integral Formula, evaluate $\\oint_{|z-1|=2} \\frac{1}{z^2 + 1} dz$."
                elif "residue" in t_lower or "laurent" in t_lower or "contour" in t_lower or "pole" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Using contour integration around the unit circle, evaluate $\\int_0^{2\\pi} \\frac{d\\theta}{5 + 4\\cos\\theta}$."
                        part_b_text = "Find all the poles and calculate the residues of $f(z) = \\frac{1}{z(z-1)^2}$."
                    else:
                        part_a_text = "Evaluate the contour integral $\\oint_{|z-3|=1} \\frac{z^2}{(z-1)^3(z-2)} dz$ using Cauchy's Residue Theorem."
                        part_b_text = "Develop $f(z) = \\frac{2z - 3i}{z^2 - 3iz - 2}$ in a Laurent series valid for the annular region $1 < |z| < 2$."
                else:
                    part_a_text = f"State the governing conditions and analytical principles of {topic} ({kw_sample})."
                    part_b_text = f"Derive the step-by-step solution for the boundary problem governed by {kw_secondary}."
            else:
                t_lower = topic.lower()
                if "stack" in t_lower or "queue" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Design a Double Ended Queue (Deque) using a doubly linked list. Write algorithms for insert_front and delete_rear."
                        part_b_text = "Demonstrate the evaluation of the postfix expression $6 \\, 2 \\, 3 \\, + \\, * \\, 3 \\, 8 \\, 2 \\, / \\, + \\, * \\, 2 \\, \\uparrow \\, 3 \\, +$ using a runtime stack."
                    else:
                        part_a_text = "Explain the array implementation of a circular queue. Formulate the conditions for queue overflow and underflow."
                        part_b_text = "Write an algorithm to convert an infix expression to postfix notation using an explicit stack. Trace your algorithm on $(A + B) * (C - D) / E$."
                elif "tree" in t_lower or "bst" in t_lower or "avl" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Construct a Red-Black Tree or B-Tree of order 3 by inserting keys: 10, 20, 30, 40, 50, 25. Explain the node splitting rules."
                        part_b_text = "Prove that an $n$-node Binary Search Tree with minimum height has height $\\lfloor \\log_2 n \\rfloor$."
                    else:
                        part_a_text = "Construct an AVL tree by sequentially inserting the following keys: 45, 23, 78, 12, 34, 89, 56. Specify the rotation type performed at each imbalance."
                        part_b_text = "Write an algorithm for deletion of a node with two children in a Binary Search Tree and analyze its worst-case time complexity."
                elif "graph" in t_lower or "dijkstra" in t_lower or "span" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Explain Floyd-Warshall's all-pairs shortest path dynamic programming algorithm. State its recurrence relation and running time."
                        part_b_text = "Formulate Bellman-Ford algorithm for single-source shortest paths in graphs containing negative-weight edges. How does it detect negative cycles?"
                    else:
                        part_a_text = "Explain Dijkstra's Single Source Shortest Path algorithm. Trace the algorithm on a graph with 5 vertices and derive its running time with a min-heap."
                        part_b_text = "Differentiate between Prim's and Kruskal's algorithms for finding a Minimum Spanning Tree. State when each algorithm is preferred."
                elif "sort" in t_lower or "search" in t_lower:
                    if is_alternate_choice:
                        part_a_text = "Explain Heap Sort with max-heapify and build-max-heap procedures. Trace Heap Sort on the array $[16, 4, 10, 14, 7, 9, 3, 2, 8, 1]$."
                        part_b_text = "Explain Radix Sort using Counting Sort as a stable sorting subroutine. Derive its time complexity for $n$ numbers with $d$ digits."
                    else:
                        part_a_text = "Explain the partition subroutine in Quick Sort. State and solve the recurrence relation for its average and worst-case time complexity."
                        part_b_text = "Describe Merge Sort with a recursion tree. Prove why Merge Sort requires $O(n)$ auxiliary space while Heap Sort operates in-place."
                else:
                    part_a_text = f"Explain the conceptual design and asymptotic bounds of {topic} with respect to {kw_sample}."
                    part_b_text = f"Formulate the algorithmic procedure for {kw_secondary} and analyze its time and space complexity."

            sub_parts = [
                SubPartItem(label="a", text=part_a_text, marks=marks_a),
                SubPartItem(label="b", text=part_b_text, marks=marks_b),
            ]

            full_text = f"(a) {part_a_text}\n(b) {part_b_text}"
            marking_scheme = [
                f"Part (a): Accurate conceptual formulation and derivation: {marks_a} marks",
                f"Part (b): Step-by-step analytical solution and final result: {marks_b} marks",
            ]
            sample_answer = (
                f"Rigorous treatment of {topic} covering {kw_sample}, "
                f"followed by complete analytical solution for {kw_secondary}."
            )
        else:
            # Single question (short or medium answer)
            if is_math:
                t_lower = topic.lower()
                if "differential" in t_lower or "ode" in t_lower or "exact" in t_lower:
                    full_text = "Give an example of a non-exact differential equation which possesses more than one integrating factor."
                elif "particular integral" in t_lower or "wronskian" in t_lower:
                    full_text = "Find the particular integral of the differential equation $y'' + y = \\cos x$."
                elif "vector" in t_lower or "curl" in t_lower or "divergence" in t_lower:
                    full_text = "If $\\vec{r} = x\\hat{i} + y\\hat{j} + z\\hat{k}$ and $r = |\\vec{r}|$, evaluate $\\nabla \\times (r^2 \\vec{r})$."
                elif "cauchy" in t_lower or "analytic" in t_lower:
                    full_text = "State the Cauchy-Riemann equations in polar coordinates. Check whether $f(z) = \\bar{z}$ is analytic anywhere."
                elif "residue" in t_lower or "pole" in t_lower or "integral" in t_lower:
                    full_text = "Evaluate $\\oint_{|z|=2} \\frac{e^z}{z - i} dz$ using Cauchy's Integral Formula."
                else:
                    full_text = f"Evaluate the analytical expression for {topic} addressing {kw_sample}."

                marking_scheme = [
                    f"Statement of governing formula / theorem: {round(marks * 0.5, 1)} marks",
                    f"Step-by-step mathematical evaluation: {round(marks * 0.5, 1)} marks",
                ]
                sample_answer = f"Analytical derivation and exact result for {topic}."
            else:
                t_lower = topic.lower()
                if "stack" in t_lower or "queue" in t_lower:
                    full_text = "State the difference between linear queue and circular queue. Why is circular queue preferred?"
                elif "tree" in t_lower or "bst" in t_lower:
                    full_text = "Define the balance factor of an AVL tree node and state the conditions requiring double rotation."
                elif "graph" in t_lower:
                    full_text = "Explain how Breadth-First Search (BFS) is used to determine the connected components of an undirected graph."
                elif "sort" in t_lower:
                    full_text = "Compare the worst-case and best-case time complexity of Quick Sort and Merge Sort."
                else:
                    full_text = f"{chosen_verb} the structural properties and operations of {topic} ({kw_sample})."

                marking_scheme = [
                    f"Core definition and governing equations: {round(marks * 0.5, 1)} marks",
                    f"Algorithmic trace and concrete example: {round(marks * 0.5, 1)} marks",
                ]
                sample_answer = f"Clear academic explanation and trace of {topic}."

        return GeneratedQuestionItem(
            id=str(uuid.uuid4()),
            question_text=full_text,
            sub_parts=sub_parts,
            marks=marks,
            unit_number=unit_number,
            unit_name=unit_name,
            topic=topic,
            blooms_level=blooms_level,
            difficulty=difficulty,
            question_type=question_type,
            marking_scheme=marking_scheme,
            sample_answer=sample_answer,
        )

    def generate_question(
        self,
        subject: str,
        unit_number: int,
        unit_name: str,
        topic: str,
        keywords: List[str],
        blooms_level: str,
        marks: float,
        difficulty: str = "medium",
        question_type: str = "medium",
        avoid_questions: Optional[List[str]] = None,
        historical_context: Optional[str] = None,
        exemplar_questions: Optional[List[str]] = None,
    ) -> GeneratedQuestionItem:
        """
        Generate a single exam question matching constraints.

        Uses Gemini API if configured and online; otherwise invokes fallback synthesizer.
        """
        # If Gemini client is active, attempt generation
        if self._client:
            try:
                system_prompt = self._build_system_prompt(subject)
                user_prompt = self._build_user_prompt(
                    subject=subject,
                    unit_number=unit_number,
                    unit_name=unit_name,
                    topic=topic,
                    keywords=keywords,
                    blooms_level=blooms_level,
                    marks=marks,
                    difficulty=difficulty,
                    question_type=question_type,
                    avoid_questions=avoid_questions,
                    historical_context=historical_context,
                    exemplar_questions=exemplar_questions,
                )

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                    config={"response_mime_type": "application/json"},
                )

                raw_json = response.text.strip()
                # Clean any markdown fences if present
                clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json, flags=re.MULTILINE).strip()
                parsed = json.loads(clean_json)

                # Convert subparts
                sub_parts_data = parsed.get("sub_parts", [])
                sub_parts = []
                for sp in sub_parts_data:
                    if sp.get("text") and float(sp.get("marks", 0)) > 0:
                        sub_parts.append(
                            SubPartItem(
                                label=str(sp.get("label", "")),
                                text=str(sp.get("text", "")),
                                marks=float(sp.get("marks", 1.0)),
                            )
                        )

                return GeneratedQuestionItem(
                    id=str(uuid.uuid4()),
                    question_text=parsed.get("question_text", ""),
                    sub_parts=sub_parts,
                    marks=marks,
                    unit_number=unit_number,
                    unit_name=unit_name,
                    topic=topic,
                    blooms_level=blooms_level,
                    difficulty=difficulty,
                    question_type=question_type,
                    marking_scheme=parsed.get("marking_scheme", []),
                    sample_answer=parsed.get("sample_answer"),
                )
            except Exception as e:
                logger.warning("gemini_generation_call_failed", error=str(e), note="Falling back to synthesizer")

        # Fallback synthesizer
        return self._generate_fallback_question(
            subject=subject,
            unit_number=unit_number,
            unit_name=unit_name,
            topic=topic,
            keywords=keywords,
            blooms_level=blooms_level,
            marks=marks,
            difficulty=difficulty,
            question_type=question_type,
            exemplar_questions=exemplar_questions,
            avoid_questions=avoid_questions,
        )

    def generate_candidate_pool(
        self,
        subject: str,
        units_data: List[Dict[str, Any]],
        target_marks_list: List[float],
        blooms_levels: List[str],
        count_per_unit: int = 3,
        avoid_questions: Optional[List[str]] = None,
    ) -> List[GeneratedQuestionItem]:
        """
        Generate a diverse candidate pool of questions covering all units, marks, and Bloom's levels.
        """
        pool: List[GeneratedQuestionItem] = []
        for unit in units_data:
            unit_no = unit["unit_number"]
            unit_name = unit["unit_name"]
            topics = unit.get("topics", [unit_name])
            keywords = unit.get("keywords", [])

            for i in range(count_per_unit):
                topic = topics[i % len(topics)]
                marks = target_marks_list[i % len(target_marks_list)]
                blooms = blooms_levels[i % len(blooms_levels)]
                q_type = "short" if marks <= 2 else ("long" if marks >= 8 else "medium")

                q = self.generate_question(
                    subject=subject,
                    unit_number=unit_no,
                    unit_name=unit_name,
                    topic=topic,
                    keywords=keywords,
                    blooms_level=blooms,
                    marks=marks,
                    difficulty="medium",
                    question_type=q_type,
                    avoid_questions=avoid_questions,
                )
                pool.append(q)

        return pool
