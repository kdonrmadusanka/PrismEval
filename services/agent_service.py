import asyncio
import json
import os
import re
from typing import Any, Optional, Tuple

import httpx
from dotenv import load_dotenv

from models.schemas import AgentResult

load_dotenv()


class CozeToneAgentError(Exception):
    """Raised when Coze cannot return a valid agent analysis."""


class AgentService:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _get_config(bot_env_var: str = "COZE_BOT_ID") -> tuple[str, str, str]:
        """
        Read Coze settings at call time so env changes are respected.
        A per-agent bot id (e.g. COZE_IMPACT_BOT_ID) falls back to COZE_BOT_ID.
        """
        base = os.getenv("COZE_API_BASE", "https://api.coze.com").rstrip("/")
        token = os.getenv("COZE_API_TOKEN")
        bot_id = os.getenv(bot_env_var) or os.getenv("COZE_BOT_ID")

        if not token or not bot_id:
            raise CozeToneAgentError(
                f"Missing COZE_API_TOKEN or {bot_env_var}/COZE_BOT_ID in environment."
            )
        return base, token, bot_id

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(text: str) -> str:
        """Neutralize our delimiter tags so user text can't break out of them."""
        return re.sub(
            r"<\s*/?\s*(context|submission)\s*>", "", text or "", flags=re.IGNORECASE
        )

    @staticmethod
    def _build_prompt(task: str, output_spec: str, submission: str, context: str) -> str:
        return f"""
{task}

Return ONLY a JSON object (no Markdown, no extra text) with:
{output_spec}

Base the assessment only on the provided evidence. Do not invent facts,
statistics, or cultural assumptions.

CONTEXT:
<context>
{AgentService._sanitize(context)}
</context>

SUBMISSION:
<submission>
{AgentService._sanitize(submission)}
</submission>
""".strip()

    # ------------------------------------------------------------------
    # JSON parsing / validation
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str, expect_flags: bool = False) -> dict[str, Any]:
        """Parse and validate JSON returned by a Coze agent."""
        text = text.strip()

        # Handle an agent that wraps JSON in Markdown fences.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: agent added prose around the JSON object.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise CozeToneAgentError(
                    "Coze returned invalid JSON. Check the agent instructions."
                )
            try:
                result = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise CozeToneAgentError(
                    "Coze returned invalid JSON. Check the agent instructions."
                ) from exc

        if not isinstance(result, dict):
            raise CozeToneAgentError("Coze response must be a JSON object.")

        score = result.get("score")
        feedback = result.get("feedback")

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise CozeToneAgentError("Coze response must contain a numeric score.")

        if not 0 <= score <= 100:
            raise CozeToneAgentError("Coze score must be between 0 and 100.")

        if not isinstance(feedback, str) or not feedback.strip():
            raise CozeToneAgentError("Coze response must contain non-empty feedback.")

        parsed: dict[str, Any] = {"score": float(score), "feedback": feedback.strip()}

        if expect_flags:
            flags = result.get("flags", [])
            if flags is None:
                flags = []
            if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
                raise CozeToneAgentError("Coze 'flags' must be a list of strings.")
            parsed["flags"] = [f.strip() for f in flags if f.strip()]

        return parsed

    # ------------------------------------------------------------------
    # Shared Coze streaming call
    # ------------------------------------------------------------------
    @staticmethod
    async def _call_coze(prompt: str, user_id: str, bot_env_var: str = "COZE_BOT_ID") -> str:
        """Send a prompt to a published Coze bot and return the full answer text."""
        api_base, token, bot_id = AgentService._get_config(bot_env_var)

        payload = {
            "bot_id": bot_id,
            "user_id": user_id,
            "stream": True,
            "auto_save_history": False,
            "additional_messages": [
                {"role": "user", "content": prompt, "content_type": "text"}
            ],
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        endpoint = f"{api_base}/v3/chat"

        answer_parts: list[str] = []
        current_event = ""
        data_lines: list[str] = []

        def process_event(event_name: str, data: str) -> None:
            """Collect the completed answer from a Coze SSE event."""
            if not data or data == "[DONE]":
                return

            try:
                event_data = json.loads(data)
            except json.JSONDecodeError:
                return

            if not isinstance(event_data, dict):
                return

            name = event_name.lower()

            if name.endswith("chat.failed") or name == "error":
                detail = event_data.get("last_error") or event_data
                raise CozeToneAgentError(f"The Coze chat failed: {str(detail)[:500]}")

            # Only the completed message holds the full text. Delta events
            # hold partial chunks and would duplicate the content.
            if name != "conversation.message.completed":
                return

            message = event_data.get("message", event_data)
            if not isinstance(message, dict):
                return

            content = message.get("content")
            if (
                message.get("type") == "answer"
                and message.get("role") == "assistant"
                and isinstance(content, str)
            ):
                answer_parts.append(content)

        try:
            timeout = httpx.Timeout(90.0, connect=15.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", endpoint, headers=headers, json=payload
                ) as response:
                    if response.is_error:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        # Never include request headers or the token.
                        raise CozeToneAgentError(
                            f"Coze HTTP error {response.status_code}: {body[:1000]}"
                        )

                    # Coze may return HTTP 200 with a JSON error body
                    # (e.g. bot not published) instead of an SSE stream.
                    if "application/json" in response.headers.get("content-type", ""):
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        raise CozeToneAgentError(
                            f"Coze rejected the request: {body[:1000]}"
                        )

                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            current_event = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                        elif line == "":
                            if data_lines:
                                process_event(current_event, "\n".join(data_lines))
                            current_event = ""
                            data_lines = []

                    # Process a final event without a trailing blank line.
                    if data_lines:
                        process_event(current_event, "\n".join(data_lines))

        except httpx.TimeoutException as exc:
            raise CozeToneAgentError(
                "Coze timed out while analyzing the submission."
            ) from exc
        except httpx.RequestError as exc:
            raise CozeToneAgentError("Could not connect to the Coze API.") from exc

        if not answer_parts:
            raise CozeToneAgentError(
                "Coze returned no completed answer. Check that the agent "
                "is published and inspect the Coze API response."
            )

        return "".join(answer_parts)

    @staticmethod
    def _require_submission(submission: str) -> None:
        if not submission or not submission.strip():
            raise ValueError("Submission cannot be empty.")

    # ------------------------------------------------------------------
    # Micro-agents
    # ------------------------------------------------------------------
    @staticmethod
    async def _run_tone_agent(submission: str, context: str) -> AgentResult:
        """Analyze tone using a published Coze agent."""
        AgentService._require_submission(submission)

        prompt = AgentService._build_prompt(
            task="Analyze the tone of the following submission in context.",
            output_spec=(
                "- score: a number between 0 and 100 (higher = better tone)\n"
                "- feedback: a concise explanation of the assessment\n\n"
                "Evaluate empathy, respect, professionalism, clarity, and\n"
                "appropriateness for the intended audience."
            ),
            submission=submission,
            context=context,
        )
        answer = await AgentService._call_coze(
            prompt, "prismeval-tone-analyzer", "COZE_TONE_BOT_ID"
        )
        parsed = AgentService._extract_json(answer)
        return AgentResult(score=parsed["score"], feedback=parsed["feedback"])

    @staticmethod
    async def _run_impact_agent(submission: str, context: str) -> AgentResult:
        """Estimate engagement/impact potential using a published Coze agent."""
        AgentService._require_submission(submission)

        prompt = AgentService._build_prompt(
            task=(
                "Assess the likely engagement and impact potential of the "
                "following submission for its intended audience."
            ),
            output_spec=(
                "- score: a number between 0 and 100 (higher = stronger impact)\n"
                "- feedback: a concise explanation citing specific strengths "
                "and weaknesses in the text\n\n"
                "Evaluate hook strength, relevance to the audience, clarity of the\n"
                "message, call to action, and emotional resonance. Only refer to\n"
                "historical campaign performance if it appears in the context or\n"
                "your knowledge base; otherwise say it is not available."
            ),
            submission=submission,
            context=context,
        )
        answer = await AgentService._call_coze(
            prompt, "prismeval-impact-analyzer", "COZE_IMPACT_BOT_ID"
        )
        parsed = AgentService._extract_json(answer)
        return AgentResult(score=parsed["score"], feedback=parsed["feedback"])

    @staticmethod
    async def _run_risk_agent(submission: str, context: str) -> AgentResult:
        """Check compliance/safety risk using Coze plus a deterministic keyword check."""
        AgentService._require_submission(submission)

        prompt = AgentService._build_prompt(
            task=(
                "Review the following submission for compliance and safety risks "
                "such as unverified or absolute claims, misleading statements, "
                "discriminatory or harassing language, privacy leaks, and legal "
                "or regulatory problems."
            ),
            output_spec=(
                "- score: a number between 0 and 100 (higher = SAFER; 100 means "
                "no risk found)\n"
                "- feedback: a concise explanation of the assessment\n"
                "- flags: an array of short strings, one per specific risk found "
                "(empty array if none)"
            ),
            submission=submission,
            context=context,
        )
        answer = await AgentService._call_coze(
            prompt, "prismeval-risk-analyzer", "COZE_RISK_BOT_ID"
        )
        parsed = AgentService._extract_json(answer, expect_flags=True)

        score: float = parsed["score"]
        feedback: str = parsed["feedback"]
        flags: list[str] = list(parsed["flags"])

        # Deterministic safety net: the LLM may miss obvious red flags.
        if re.search(r"\babsolute(ly)?\b", submission, flags=re.IGNORECASE):
            local_flag = "Warning: Unverified absolute claim detected."
            if local_flag not in flags:
                flags.append(local_flag)
            score = min(score, 70.0)

        return AgentResult(score=score, feedback=feedback, flags=flags)

    # ------------------------------------------------------------------
    # Meta-evaluator
    # ------------------------------------------------------------------
    @classmethod
    async def evaluate_with_multi_agents(
        cls, submission: str, context: str, latency: float
    ) -> Tuple[AgentResult, AgentResult, AgentResult, float]:
        """
        Runs all micro-agents concurrently and aggregates via the Meta-Evaluator.
        EQI = 0.4*Tone + 0.4*Impact + 0.2*Risk - LatencyPenalty
        """
        tone, impact, risk = await asyncio.gather(
            cls._run_tone_agent(submission, context),
            cls._run_impact_agent(submission, context),
            cls._run_risk_agent(submission, context),
        )

        latency_penalty = max(0.0, (latency - 1.5) * 2.0)
        overall_eqi = round(
            (0.4 * tone.score) + (0.4 * impact.score) + (0.2 * risk.score)
            - latency_penalty,
            2,
        )
        overall_eqi = max(0.0, overall_eqi)

        return tone, impact, risk, overall_eqi