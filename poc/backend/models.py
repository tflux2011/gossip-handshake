"""
Pydantic models for request/response validation.

All user inputs are validated and sanitised here.
"""

from pydantic import BaseModel, Field, field_validator
import re


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The user's question about African agriculture.",
    )
    model_scale: str = Field(
        default="0.5B",
        description="Model scale to use: '0.5B' or '1.5B'.",
    )
    compare_merged: bool = Field(
        default=False,
        description="Whether to also generate from the TIES-merged model.",
    )
    router_type: str = Field(
        default="keyword",
        description="Router type: 'keyword' or 'cosine'.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="Generation temperature.",
    )

    @field_validator("query")
    @classmethod
    def sanitise_query(cls, v: str) -> str:
        """Remove control characters and excessive whitespace."""
        # Strip control characters (keep newlines and tabs)
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        # Collapse multiple whitespace
        v = re.sub(r"\s+", " ", v).strip()
        return v

    @field_validator("model_scale")
    @classmethod
    def validate_model_scale(cls, v: str) -> str:
        if v not in ("0.5B", "1.5B"):
            raise ValueError("model_scale must be '0.5B' or '1.5B'")
        return v

    @field_validator("router_type")
    @classmethod
    def validate_router_type(cls, v: str) -> str:
        if v not in ("keyword", "cosine"):
            raise ValueError("router_type must be 'keyword' or 'cosine'")
        return v


class RoutingResult(BaseModel):
    """Result of the domain routing step."""

    domain: str
    domain_label: str
    domain_icon: str
    domain_color: str
    keyword_matches: dict[str, list[str]]
    scores: dict[str, int]
    confidence: float
    router_type: str


class GenerationResult(BaseModel):
    """Result of model generation."""

    response: str
    adapter_used: str
    generation_time_ms: float


class QueryResponse(BaseModel):
    """Full response to a query."""

    routing: RoutingResult
    specialist: GenerationResult
    merged: GenerationResult | None = None
    model_scale: str
    total_time_ms: float


class DomainInfo(BaseModel):
    """Information about an available domain."""

    key: str
    label: str
    description: str
    icon: str
    color: str
    adapter_available: bool


class ModelStatus(BaseModel):
    """Current model loading status."""

    model_scale: str
    model_id: str
    loaded: bool
    domains_loaded: list[str]
    merged_loaded: bool


class SwitchModelRequest(BaseModel):
    """Request to switch model scale."""

    model_scale: str

    @field_validator("model_scale")
    @classmethod
    def validate_model_scale(cls, v: str) -> str:
        if v not in ("0.5B", "1.5B"):
            raise ValueError("model_scale must be '0.5B' or '1.5B'")
        return v


class SampleQuestion(BaseModel):
    """A sample question for the user to try."""

    question: str
    domain: str
    domain_icon: str
