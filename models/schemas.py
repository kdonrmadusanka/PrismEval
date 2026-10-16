from pydantic import BaseModel, Field
from typing import List, Optional

class Hyperparameters(BaseModel):
    token_budget: int = Field(default=1024, ge=256, le=4096)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0) # Dense vs BM25 ratio

class EvaluationRequest(BaseModel):
    student_id: str
    submission_text: str
    hyperparameters: Hyperparameters

class AgentResult(BaseModel):
    score: float
    feedback: str
    flags: List[str] = []

class EvaluationResponse(BaseModel):
    student_id: str
    overall_eqi: float
    tone_score: AgentResult
    impact_score: AgentResult
    risk_score: AgentResult
    latency_seconds: float
    feishu_synced: bool