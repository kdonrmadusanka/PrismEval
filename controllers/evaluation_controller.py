import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

import database
from models.schemas import EvaluationRequest, EvaluationResponse
from repositories.evaluation_repository import EvaluationRepository
from services.agent_service import AgentService, CozeToneAgentError
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_submission(request: EvaluationRequest) -> EvaluationResponse:
    """RAG retrieval -> tone/impact/risk agents -> EQI -> Feishu sync -> persist."""
    started = time.perf_counter()

    try:
        context = await RAGService.retrieve_context(
            request.submission_text,
            request.hyperparameters.token_budget,
            request.hyperparameters.hybrid_alpha,
        )
        # The EQI latency penalty is applied before the agents run, so it is
        # based on the time spent so far (retrieval stage).
        pre_agent_latency = time.perf_counter() - started

        tone, impact, risk, overall_eqi = await AgentService.evaluate_with_multi_agents(
            request.submission_text, context, pre_agent_latency
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except CozeToneAgentError as exc:
        # Full detail goes to the logs; the client gets a generic message.
        logger.error("Agent evaluation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI evaluation service failed. Please try again later.",
        )
    except Exception:
        logger.exception("Unexpected error during evaluation")
        raise HTTPException(status_code=500, detail="Internal error during evaluation.")

    try:
        feishu_synced = await EvaluationRepository.sync_to_feishu_bitable(
            request.student_id, overall_eqi, risk.flags
        )
    except Exception:
        logger.exception("Feishu sync raised unexpectedly")
        feishu_synced = False

    total_latency = round(time.perf_counter() - started, 3)

    response = EvaluationResponse(
        student_id=request.student_id,
        overall_eqi=overall_eqi,
        tone_score=tone,
        impact_score=impact,
        risk_score=risk,
        latency_seconds=total_latency,
        feishu_synced=feishu_synced,
    )

    # A storage failure must not lose an evaluation the client already paid for.
    try:
        await database.save_evaluation(
            student_id=request.student_id,
            submission_text=request.submission_text,
            overall_eqi=overall_eqi,
            tone=tone.model_dump(),
            impact=impact.model_dump(),
            risk=risk.model_dump(),
            latency_seconds=total_latency,
            feishu_synced=feishu_synced,
        )
    except Exception:
        logger.exception("Failed to persist evaluation")

    return response


@router.get("/evaluations/{student_id}")
async def list_student_evaluations(
    student_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    """Evaluation history for one student, newest first."""
    return await database.list_evaluations(student_id, limit, offset)


@router.get("/evaluation/{evaluation_id}")
async def get_single_evaluation(evaluation_id: int) -> Dict[str, Any]:
    record = await database.get_evaluation(evaluation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    return record