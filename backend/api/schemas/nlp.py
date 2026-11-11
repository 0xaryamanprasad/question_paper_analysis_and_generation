"""
Pydantic Schemas for NLP Preprocessing.

Defines schemas for linguistic tokens, lemmas, extracted keywords,
noun chunks, and normalized question representations.
"""

from pydantic import BaseModel, Field


class NLPToken(BaseModel):
    """Represents an individual token with POS and morphological attributes."""
    text: str
    lemma: str
    pos: str
    tag: str
    is_stop: bool
    is_alpha: bool


class NLPProcessedQuestion(BaseModel):
    """Detailed linguistic analysis and normalization of a question."""
    original_text: str = Field(..., description="Verbatim question text")
    normalized_text: str = Field(..., description="Cleaned, normalized text")
    cleaned_tokens: list[str] = Field(
        default_factory=list, description="Non-stop, alphanumeric lemmatized tokens"
    )
    tokens: list[NLPToken] = Field(default_factory=list, description="All tokens with grammatical metadata")
    keywords: list[str] = Field(
        default_factory=list, description="Extracted domain keywords and technical concepts"
    )
    noun_chunks: list[str] = Field(
        default_factory=list, description="Base noun phrases (e.g., 'binary search tree')"
    )
    action_verbs: list[str] = Field(
        default_factory=list, description="Action/task verbs indicating cognitive demand"
    )
    abbreviations_expanded: dict[str, str] = Field(
        default_factory=dict, description="Domain acronyms expanded to full terms"
    )
    entities: list[dict[str, str]] = Field(
        default_factory=list, description="Named entities identified in the text"
    )
