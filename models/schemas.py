from typing import List

from pydantic import BaseModel, Field, field_validator


class Hyperparameters(BaseModel):
    token_budget: int = Field(default=1024, ge=256, le=4096)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)  # Dense vs BM25 ratio


class EvaluationRequest(BaseModel):
    student_id: str = Field(max_length=128)
    submission_text: str = Field(max_length=20000)
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)

    @field_validator("student_id", "submission_text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class AgentResult(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    feedback: str
    flags: List[str] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    student_id: str
    overall_eqi: float = Field(ge=0.0, le=100.0)
    tone_score: AgentResult
    impact_score: AgentResult
    risk_score: AgentResult
    latency_seconds: float = Field(ge=0.0)
    feishu_synced: bool