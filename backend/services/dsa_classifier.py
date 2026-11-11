"""
Module 4: DSA Topic Classification Service.

Enforces:
- Strict classification into predefined DSA categories:
  Arrays, Linked Lists, Stacks, Queues, Trees, BST, Heaps, Graphs, Sorting, Searching,
  Greedy Algorithms, Dynamic Programming, Backtracking, Divide and Conquer, Hashing, Unclassified.
- Supports primary and secondary topic labels.
- Classifies based on actual algorithmic concepts rather than keyword matching alone.
- Never invents categories.
- Returns classification confidence (0.0 to 1.0).
- Assigns 'Unclassified' category when uncertain (< 0.40 confidence).
"""

import json
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DSACategory(str, Enum):
    """Predefined syllabus DSA categories. No invented categories allowed."""
    ARRAYS = "Arrays"
    LINKED_LISTS = "Linked Lists"
    STACKS = "Stacks"
    QUEUES = "Queues"
    TREES = "Trees"
    BST = "BST"
    HEAPS = "Heaps"
    GRAPHS = "Graphs"
    SORTING = "Sorting"
    SEARCHING = "Searching"
    GREEDY_ALGORITHMS = "Greedy Algorithms"
    DYNAMIC_PROGRAMMING = "Dynamic Programming"
    BACKTRACKING = "Backtracking"
    DIVIDE_AND_CONQUER = "Divide and Conquer"
    HASHING = "Hashing"
    UNCLASSIFIED = "Unclassified"


class DSATopicClassificationResult(BaseModel):
    """Result of DSA topic classification for a single question."""
    primary_topic: DSACategory = Field(..., description="Main DSA category")
    secondary_topic: Optional[DSACategory] = Field(
        default=None, description="Optional secondary overlapping category"
    )
    subtopic: Optional[str] = Field(
        default=None, description="Granular algorithmic concept (e.g., Dijkstra, AVL Rotations)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    reasoning: str = Field(..., description="Conceptual justification for this classification")

    @field_validator("primary_topic", mode="before")
    @classmethod
    def validate_primary_topic(cls, v):
        if isinstance(v, str):
            for cat in DSACategory:
                if v.strip().lower() == cat.value.lower():
                    return cat
            return DSACategory.UNCLASSIFIED
        return v

    @field_validator("secondary_topic", mode="before")
    @classmethod
    def validate_secondary_topic(cls, v):
        if not v:
            return None
        if isinstance(v, str):
            for cat in DSACategory:
                if v.strip().lower() == cat.value.lower():
                    return cat
            return None
        return v


class DSATopicClassifier:
    """
    Classifies questions into predefined DSA categories using algorithmic reasoning
    via Gemini or deep conceptual ontology rules.
    """

    CLASSIFICATION_SYSTEM_PROMPT = """You are an expert Computer Science professor in Data Structures and Algorithms.
Your task is to classify an exam question into predefined DSA categories based on ACTUAL ALGORITHMIC CONCEPTS, not superficial keyword matching.

ALLOWED CATEGORIES (DO NOT INVENT ANY OTHER CATEGORY):
- Arrays
- Linked Lists
- Stacks
- Queues
- Trees
- BST
- Heaps
- Graphs
- Sorting
- Searching
- Greedy Algorithms
- Dynamic Programming
- Backtracking
- Divide and Conquer
- Hashing
- Unclassified

RULES:
1. Provide a primary_topic from the allowed list.
2. If the problem fundamentally involves a second concept (e.g., Dijkstra involves Graphs and Greedy Algorithms / Heaps, or LRU Cache involves Hashing and Linked Lists), provide a secondary_topic from the allowed list; otherwise null.
3. Identify the granular subtopic (e.g., "Kruskal's MST", "Bellman-Ford", "Red-Black Tree Insertion", "0/1 Knapsack", "MergeSort Recurrence").
4. Return a confidence score between 0.0 and 1.0.
5. If the question does not match any DSA concept or is ambiguous/administrative, set primary_topic to "Unclassified" and confidence < 0.4.
6. Provide a concise explanation of the algorithmic reasoning.

Return valid JSON matching:
{
  "primary_topic": "Graphs",
  "secondary_topic": "Greedy Algorithms",
  "subtopic": "Dijkstra's Shortest Path",
  "confidence": 0.95,
  "reasoning": "Question asks to trace Dijkstra algorithm on a weighted directed graph."
}
"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning("gemini_dsa_init_failed", error=str(e))

    def classify_question(
        self,
        question_text: str,
        latex_expressions: Optional[List[str]] = None,
    ) -> DSATopicClassificationResult:
        """
        Classifies question into strict predefined DSA categories.
        """
        text = question_text.strip()
        latex_list = latex_expressions or []

        # 1. Attempt Gemini classification
        if self.client and len(text) > 10:
            try:
                return self._call_gemini_classifier(text, latex_list)
            except Exception as err:
                logger.warning(
                    "gemini_dsa_classification_failed_fallback_to_concept_rules",
                    error=str(err),
                )

        # 2. Conceptual ontology rule fallback
        return self._conceptual_classify(text, latex_list)

    def _call_gemini_classifier(
        self,
        question_text: str,
        latex_list: List[str],
    ) -> DSATopicClassificationResult:
        """Invokes Gemini model for algorithmic reasoning classification."""
        prompt = (
            f"Question text:\n```\n{question_text}\n```\n"
            f"LaTeX expressions in question: {latex_list}\n"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "system_instruction": self.CLASSIFICATION_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )

        data = json.loads(response.text.strip())
        res = DSATopicClassificationResult.model_validate(data)

        # Enforce threshold for Unclassified
        if res.confidence < 0.40:
            res.primary_topic = DSACategory.UNCLASSIFIED
            res.secondary_topic = None

        return res

    def _conceptual_classify(
        self,
        text: str,
        latex_list: List[str],
    ) -> DSATopicClassificationResult:
        """
        Deterministic algorithmic concept reasoning engine.
        Evaluates algorithmic paradigms, data structure mechanics, and theoretical problems.
        """
        t = text.lower()
        full_ctx = t + " " + " ".join(latex_list).lower()

        # Concept rules with weighted indicators
        concepts = [
            # Dynamic Programming
            (
                DSACategory.DYNAMIC_PROGRAMMING,
                [
                    "dynamic programming", "optimal substructure", "overlapping subproblem",
                    "memoization", "tabulation", "knapsack", "longest common subsequence",
                    "matrix chain multiplication", "bellman-ford", "floyd-warshall",
                    "edit distance", "subset sum dp", "coin change",
                ],
                "Problem requires dynamic programming via memoization, tabulation, or optimal substructure.",
                "Optimal Substructure / Tabulation",
            ),
            # Divide and Conquer
            (
                DSACategory.DIVIDE_AND_CONQUER,
                [
                    "divide and conquer", "master theorem", "recurrence relation",
                    "t(n) =", "strassen", "merge sort", "quicksort partition",
                    "binary search recurrence",
                ],
                "Question addresses divide and conquer paradigms or recurrence relations.",
                "Recurrence / Divide & Conquer Paradigm",
            ),
            # Backtracking
            (
                DSACategory.BACKTRACKING,
                [
                    "backtracking", "n-queens", "n queens", "hamiltonian cycle",
                    "graph coloring", "subset sum backtracking", "sudoku",
                    "state space tree", "branch and bound",
                ],
                "Focuses on combinatorial search with backtracking or state-space exploration.",
                "State-Space Search & Backtracking",
            ),
            # Greedy Algorithms
            (
                DSACategory.GREEDY_ALGORITHMS,
                [
                    "greedy", "huffman", "fractional knapsack", "activity selection",
                    "kruskal", "prim's", "prims", "dijkstra", "job sequencing",
                    "optimal merge",
                ],
                "Involves greedy choice property and locally optimal selections.",
                "Greedy Choice Strategy",
            ),
            # Graphs
            (
                DSACategory.GRAPHS,
                [
                    "graph", "adjacency matrix", "adjacency list", "bfs", "dfs",
                    "breadth first", "depth first", "topological sort", "shortest path",
                    "minimum spanning tree", "mst", "bipartite", "cycle detection",
                    "strongly connected", "eulerian", "hamiltonian", "dag",
                ],
                "Deals with graph algorithms, representations, or connectivity traversals.",
                "Graph Representation & Traversal",
            ),
            # Trees & BST
            (
                DSACategory.BST,
                [
                    "binary search tree", "bst", "avl", "red-black", "red black",
                    "splay tree", "tree rotation", "inorder predecessor",
                ],
                "Focuses specifically on Binary Search Tree properties or balanced search trees.",
                "Binary Search Tree / Balancing",
            ),
            (
                DSACategory.TREES,
                [
                    "binary tree", "tree traversal", "inorder", "preorder", "postorder",
                    "level order", "b-tree", "b+ tree", "trie", "segment tree",
                    "fenwick", "height of tree", "complete binary tree",
                ],
                "Addresses general tree structures, hierarchical representations, or traversals.",
                "Tree Structures & Traversals",
            ),
            # Heaps
            (
                DSACategory.HEAPS,
                [
                    "heap", "priority queue", "heapify", "heapsort", "min-heap",
                    "max-heap", "binomial heap", "fibonacci heap",
                ],
                "Involves priority queues, heap properties, or heapify algorithms.",
                "Heap & Priority Queue Mechanics",
            ),
            # Hashing
            (
                DSACategory.HASHING,
                [
                    "hashing", "hash table", "hash function", "collision resolution",
                    "linear probing", "quadratic probing", "double hashing",
                    "separate chaining", "lru cache", "load factor",
                ],
                "Focuses on hash tables, collision resolution strategies, or load factors.",
                "Hashing & Collision Resolution",
            ),
            # Stacks & Queues
            (
                DSACategory.STACKS,
                [
                    "stack", "push", "pop", "infix to postfix", "postfix evaluation",
                    "balanced parentheses", "monotonic stack", "call stack",
                ],
                "Focuses on LIFO Stack data structure operations and applications.",
                "Stack LIFO Operations",
            ),
            (
                DSACategory.QUEUES,
                [
                    "queue", "circular queue", "deque", "enqueue", "dequeue",
                    "double ended queue",
                ],
                "Deals with FIFO Queue and Deque mechanics.",
                "Queue FIFO Operations",
            ),
            # Linked Lists
            (
                DSACategory.LINKED_LISTS,
                [
                    "linked list", "singly linked", "doubly linked", "circular linked",
                    "node pointer", "reverse linked list", "skip list",
                ],
                "Addresses linked list operations, pointer manipulations, and dynamic sequences.",
                "Linked List Dynamic Sequences",
            ),
            # Sorting
            (
                DSACategory.SORTING,
                [
                    "sorting", "quicksort", "merge sort", "bubble sort", "insertion sort",
                    "selection sort", "radix sort", "counting sort", "bucket sort",
                    "stability in sorting",
                ],
                "Addresses comparison or non-comparison sorting algorithms and complexities.",
                "Sorting Algorithms",
            ),
            # Searching
            (
                DSACategory.SEARCHING,
                [
                    "binary search", "linear search", "interpolation search",
                    "exponential search", "ternary search",
                ],
                "Addresses searching algorithms in ordered or unordered sequences.",
                "Search Space Pruning",
            ),
            # Arrays
            (
                DSACategory.ARRAYS,
                [
                    "array", "multidimensional array", "row major", "column major",
                    "prefix sum", "two pointer", "sliding window", "matrix representation",
                ],
                "Involves array manipulations, indexing calculations, or linear storage.",
                "Array Manipulations",
            ),
        ]

        scored: List[tuple[DSACategory, int, str, str]] = []
        for cat, keywords, reason, subtopic in concepts:
            matches = sum(1 for kw in keywords if kw in full_ctx)
            if matches > 0:
                scored.append((cat, matches, reason, subtopic))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return DSATopicClassificationResult(
                primary_topic=DSACategory.UNCLASSIFIED,
                secondary_topic=None,
                subtopic=None,
                confidence=0.20,
                reasoning="No clear DSA algorithmic concept detected in the question statement.",
            )

        top_cat, top_count, top_reason, top_subtopic = scored[0]
        conf = min(0.95, 0.55 + 0.12 * top_count)

        sec_cat = None
        if len(scored) > 1 and scored[1][1] >= 1 and scored[1][0] != top_cat:
            sec_cat = scored[1][0]

        return DSATopicClassificationResult(
            primary_topic=top_cat,
            secondary_topic=sec_cat,
            subtopic=top_subtopic,
            confidence=round(conf, 2),
            reasoning=top_reason,
        )
