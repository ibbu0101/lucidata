import json
from typing import Any

import instructor
import litellm
from pydantic import ValidationError

from lucidata.ai.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from lucidata.config import settings
from lucidata.core.exceptions import LLMProviderError
from lucidata.core.schema import ExecutiveSummaryNarrative

_completion_client: instructor.Instructor | None = None


def _get_client() -> instructor.Instructor:
    global _completion_client  # noqa: PLW0603
    if _completion_client is None:
        _completion_client = instructor.from_litellm(litellm.completion)
    return _completion_client


def generate_narrative(
    payload: dict[str, Any],
    *,
    provider: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> ExecutiveSummaryNarrative:
    """Generate an executive summary narrative using an LLM.

    Args:
        payload: Sanitized statistical metadata from build_payload().
        provider: LLM provider name (e.g., "ollama", "openai", "anthropic").
        model: Specific model name (e.g., "llama3", "gpt-4o-mini").
        timeout: Request timeout in seconds.

    Returns:
        Validated ExecutiveSummaryNarrative instance.

    Raises:
        LLMProviderError: On timeout, auth failure, connection error,
            or schema validation failure after retries.
    """
    resolved_provider = provider or settings.default_provider
    resolved_model = model or settings.default_model
    resolved_timeout = timeout or settings.default_timeout_seconds

    model_str = f"{resolved_provider}/{resolved_model}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                metadata_payload=json.dumps(payload, default=str)
            ),
        },
    ]

    client = _get_client()

    try:
        narrative = client.chat.completions.create(
            model=model_str,
            messages=messages,
            response_model=ExecutiveSummaryNarrative,
            temperature=settings.temperature,
            timeout=resolved_timeout,
            max_retries=settings.max_validation_retries,
        )
    except (TimeoutError, litellm.Timeout) as e:
        raise LLMProviderError(f"LLM request timed out after {resolved_timeout}s") from e
    except litellm.AuthenticationError as e:
        raise LLMProviderError(f"LLM authentication failed: {e}") from e
    except litellm.APIConnectionError as e:
        raise LLMProviderError(f"LLM connection failed: {e}") from e
    except ValidationError as e:
        raise LLMProviderError(f"LLM response failed schema validation: {e}") from e
    except litellm.APIError as e:
        raise LLMProviderError(f"LLM API error: {e}") from e

    return narrative
