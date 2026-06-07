import time
from collections.abc import Iterator

import litellm
import litellm.exceptions as llm_exc

# Suppress litellm's verbose success/failure logging
litellm.suppress_debug_info = True


class LLMClient:
    def __init__(self, api_key: str, model: str) -> None:
        """
        `model` must be a fully-qualified litellm model string, e.g.:
            'claude-3-5-sonnet-20241022'   (Anthropic)
            'gemini/gemini-2.0-flash'      (Gemini)
        The API key is passed explicitly per call so it is never stored
        in a global litellm config.
        """
        self.model = model
        self._api_key = api_key

    def stream(self, history: list[dict]) -> Iterator[str]:
        """Stream response tokens for the given message history.

        Yields text delta strings as they arrive. Retries on transient
        rate limit errors with exponential backoff (max 3 attempts).
        Raises RuntimeError immediately on authentication errors.
        """
        max_retries = 3
        delay = 1.0

        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=history,
                    api_key=self._api_key,
                    stream=True,
                )
                for chunk in response:
                    text = chunk.choices[0].delta.content
                    if text:
                        yield text
                return  # success — stop retrying

            except llm_exc.AuthenticationError as exc:
                raise RuntimeError(
                    f"Authentication failed — check your API key. ({exc})"
                ) from exc

            except llm_exc.RateLimitError as exc:
                # Daily / hard quota exhaustion (limit: 0) — retrying won't help
                if "limit: 0" in str(exc) and "PerDay" in str(exc):
                    raise RuntimeError(
                        "Daily free-tier quota exhausted for this model. "
                        "It resets at midnight Pacific time — see https://ai.dev/rate-limit."
                    ) from exc
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Rate limit hit and retries exhausted. Try again in a moment. ({exc})"
                    ) from exc
                time.sleep(delay)
                delay *= 2

            except (llm_exc.APIConnectionError, llm_exc.APIError):
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
