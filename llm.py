"""Anthropic client factory — Amazon Bedrock (AWS credits) or first-party API.

Backend is chosen by environment, so no caller needs to know which is active:
  * AWS_BEARER_TOKEN_BEDROCK set  -> Amazon Bedrock (billed against AWS credits)
  * otherwise                     -> first-party Claude API via ANTHROPIC_API_KEY

Bedrock notes (verified against this AWS account):
  * Uses the Mantle client with bare "anthropic.claude-*" model IDs. The older
    InvokeModel client + "us." inference-profile IDs also works, but only for
    models up to Opus 4.6 — Opus 5 / Sonnet 5 are served through Mantle.
  * thinking={"type":"adaptive"}, output_config.effort and cache_control all
    work. Structured outputs (output_config.format) and strict tool use do NOT
    — both return "Extra inputs are not permitted". Use call_structured()
    below, which gets schema-shaped JSON via forced tool use instead.
  * Schemas cannot carry minimum/maximum, so numeric ranges are prompt-enforced
    only; clamp on the way out (models do occasionally return out-of-range).
"""

import os
from typing import Optional

import anthropic

DEFAULT_AWS_REGION = "us-east-1"

# tier -> model id, per backend
BEDROCK_MODELS = {
    "fast": "anthropic.claude-sonnet-5",
    "deep": "anthropic.claude-opus-5",
}
DIRECT_MODELS = {
    "fast": "claude-sonnet-5",
    "deep": "claude-opus-5",
}


def use_bedrock() -> bool:
    """True when AWS Bedrock credentials are present."""
    return bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))


def model(tier: str) -> str:
    """Model id for a tier ("fast" or "deep") on the active backend."""
    table = BEDROCK_MODELS if use_bedrock() else DIRECT_MODELS
    try:
        return table[tier]
    except KeyError:
        raise ValueError(f"Bilinmeyen model katmanı: {tier!r}") from None


def backend_name() -> str:
    if use_bedrock():
        region = os.environ.get("AWS_REGION", DEFAULT_AWS_REGION)
        return f"Amazon Bedrock ({region})"
    return "Claude API (birinci taraf)"


def get_client():
    """Return an Anthropic client for the active backend.

    Both clients expose the same messages.create surface, so callers are
    backend-agnostic.
    """
    if use_bedrock():
        region = os.environ.get("AWS_REGION", DEFAULT_AWS_REGION)
        # The client reads AWS_BEARER_TOKEN_BEDROCK from the environment.
        return anthropic.AnthropicBedrockMantle(aws_region=region)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Ne AWS_BEARER_TOKEN_BEDROCK ne de ANTHROPIC_API_KEY ayarlı."
        )
    return anthropic.Anthropic(api_key=api_key)


def has_credentials() -> bool:
    """True when either backend is configured."""
    return use_bedrock() or bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_structured(
    client,
    *,
    model: str,
    system,
    user_content: str,
    schema: dict,
    max_tokens: int,
    tool_name: str = "sonuc",
    tool_description: str = "Sonucu bu şemaya göre döndür.",
    effort: Optional[str] = None,
    thinking: bool = False,
) -> dict:
    """Return schema-shaped JSON from the model.

    Uses forced tool use rather than output_config.format: Bedrock's current
    models reject structured outputs, but a forced tool call gives the same
    guarantee — the response carries a tool_use block whose .input already
    matches the schema, with no text parsing.
    """
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "tools": [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": tool_name},
        "messages": [{"role": "user", "content": user_content}],
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    if effort:
        kwargs["output_config"] = {"effort": effort}

    response = client.messages.create(**kwargs)
    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        raise ValueError(f"Yanıtta tool_use bloğu yok (stop={response.stop_reason})")
    return block.input
