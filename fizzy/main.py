from pathlib import Path

import typer
from rich.console import Console

from fizzy.chat_loop import ChatLoop
from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.llm_client import LLMClient
from fizzy.session import Session
from fizzy.token_tracker import TokenTracker

app = typer.Typer(add_completion=False, help="fizzy — terminal AI agent")

# Context window sizes by litellm model string.
# Week 5 will replace this with a proper registry.
_MODEL_MAX_TOKENS: dict[str, int] = {
    # Anthropic
    "claude-opus-4-5":               200_000,
    "claude-sonnet-4-5":             200_000,
    "claude-haiku-4-5":              200_000,
    "claude-3-5-sonnet-20241022":    200_000,
    "claude-3-5-haiku-20241022":     200_000,
    "claude-3-opus-20240229":        200_000,
    "claude-3-haiku-20240307":       200_000,
    # Gemini — litellm prefix: "gemini/<model>"
    "gemini/gemini-2.5-flash":       1_048_576,
    "gemini/gemini-2.0-flash":       1_048_576,
    "gemini/gemini-1.5-flash":       1_048_576,
    "gemini/gemini-1.5-pro":         2_097_152,
}
_DEFAULT_MAX_TOKENS = 200_000

# Default model per provider (short name, before litellm prefix is applied)
_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini":    "gemini-2.5-flash",
}

# litellm model prefix per provider ("" = no prefix needed)
_PROVIDER_PREFIX = {
    "anthropic": "",
    "gemini":    "gemini/",
}

# Env var to surface in error messages
_PROVIDER_ENVVAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}


@app.command()
def main(
    provider: str = typer.Option(
        "anthropic",
        "--provider", "-p",
        help="LLM provider: 'anthropic' or 'gemini'.",
    ),
    model: str = typer.Option(
        None,
        "--model", "-m",
        help="Model name (without provider prefix). Defaults to provider's recommended model.",
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        envvar=["ANTHROPIC_API_KEY", "GEMINI_API_KEY"],
        help="API key. Can also be set via ANTHROPIC_API_KEY or GEMINI_API_KEY env var.",
    ),
    working_dir: Path = typer.Option(
        Path("."),
        "--working-dir", "-d",
        exists=True,
        file_okay=False,
        help="Working directory for the session.",
    ),
) -> None:
    console = Console()
    try:
        if provider not in _PROVIDER_DEFAULT_MODEL:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose: {', '.join(_PROVIDER_DEFAULT_MODEL)}"
            )

        if not api_key:
            env_var = _PROVIDER_ENVVAR[provider]
            raise ValueError(
                f"No API key provided. Set --api-key or the {env_var} environment variable."
            )

        base_model = model or _PROVIDER_DEFAULT_MODEL[provider]
        litellm_model = _PROVIDER_PREFIX[provider] + base_model
        max_tokens = _MODEL_MAX_TOKENS.get(litellm_model, _DEFAULT_MAX_TOKENS)

        session = Session(
            model=litellm_model,
            working_dir=working_dir.resolve(),
            max_tokens=max_tokens,
        )
        client = LLMClient(api_key=api_key, model=litellm_model)
        reader = InputReader()
        renderer = OutputRenderer()
        tracker = TokenTracker(session=session)

        loop = ChatLoop(
            session=session,
            client=client,
            reader=reader,
            renderer=renderer,
            tracker=tracker,
        )
        loop.run()

    except Exception as exc:
        console.print(f"[bold red]Fatal error:[/bold red] {exc}")
        raise SystemExit(1)
