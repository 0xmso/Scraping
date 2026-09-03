"""Anthropic client factory — Amazon Bedrock (AWS credits) or first-party API.

Backend is chosen by environment, so no caller needs to know which is active:
  * AWS_BEARER_TOKEN_BEDROCK set  -> Amazon Bedrock (billed against AWS credits)
  * otherwise                     -> first-party Claude API via ANTHROPIC_API_KEY

Bedrock notes (verified against this AWS account):
  * The Mantle client (AnthropicBedrockMantle) 404s here — that endpoint isn't
    provisioned for the account — so we use the InvokeModel client instead.
  * Models must be addressed by cross-region *inference profile* ID (the "us."
    prefix). The bare "anthropic.claude-*" IDs return 403/404.
  * Only the profiles below are granted on the account. Opus 5 / 4.7 / 4.8 and
    the Fable family return "not available for this account" until model access
    is enabled in the Bedrock console; when that happens, bump DEEP here.
  * Structured outputs, output_config.effort and thinking={"type":"adaptive"}
    all work on this path. Prompt caching must use explicit cache_control
    breakpoints — the legacy Bedrock integration rejects top-level cache_control.
"""

import os

import anthropic

DEFAULT_AWS_REGION = "us-east-1"

# tier -> model id, per backend
BEDROCK_MODELS = {
    "fast": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "deep": "us.anthropic.claude-opus-4-6-v1",
}
DIRECT_MODELS = {
    "fast": "claude-haiku-4-5",
    "deep": "claude-opus-4-7",
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
        # AnthropicBedrock reads AWS_BEARER_TOKEN_BEDROCK from the environment.
        return anthropic.AnthropicBedrock(aws_region=region)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Ne AWS_BEARER_TOKEN_BEDROCK ne de ANTHROPIC_API_KEY ayarlı."
        )
    return anthropic.Anthropic(api_key=api_key)


def has_credentials() -> bool:
    """True when either backend is configured."""
    return use_bedrock() or bool(os.environ.get("ANTHROPIC_API_KEY"))
