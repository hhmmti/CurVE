"""CurVE-scoped Bedrock Converse wrapper.

This is a **net-new, CurVE-only** Converse wrapper. It deliberately does NOT
import or subclass the shared ``esp_resources_v2.llm.bedrock`` wrapper — that one
ships in the monorepo, is read-only reference here, and carries a known
``inferenceConfig`` bug (its ``get_response`` binds ``kwargs["inferenceConfig"]``
to an empty dict *before* the if/else rebinds the local ``inference_config`` to a
new object, so ``maxTokens``/``temperature``/``topP`` never reach ``converse`` and
Bedrock defaults silently apply).

This wrapper fixes that by assembling the ``inferenceConfig`` dict **fully** and
only then putting it into the request — so the inference params actually transmit.

Extended thinking is enabled by default. When thinking is on, Bedrock requires
``temperature = 1.0`` and ``topP`` to be omitted; this wrapper enforces that.
"""

import os
from typing import Any, Dict, List, Optional

import boto3

from curve import config

# Cross-region inference profile for Claude Sonnet 4.6 (US). On-demand invocation
# of the bare base model id is not supported; the prefixed profile id is required.
# Sourced from curve.config (env-overridable; defaults equal the prior literals).
CURVE_MODEL_ID = config.BEDROCK_MODEL_ID
CURVE_REGION = config.AWS_REGION
CURVE_DEFAULT_PROFILE = config.AWS_PROFILE

# Inference defaults. Note maxTokens is modest here vs. the VE's 131k — M1 routes
# on stubs, it does not need huge completions.
CURVE_MAX_TOKENS = 4096
CURVE_TEMPERATURE = 0.2
CURVE_TOP_P = 0.95
CURVE_THINKING_BUDGET = 2000


class CurveBedrockWrapper:
    """Thin Bedrock Converse wrapper for CurVE.

    Args:
        profile_name: AWS profile to build the boto3 session from. When ``None``
            (the default), fall back to the standard credential chain — this keeps
            the role-based path working when CurVE later runs in-Lambda. The CLI
            passes ``roam-ai`` explicitly; library/Lambda callers leave it ``None``.
        region_name: AWS region (Bedrock). Defaults to ``us-east-1``.
        model_id: Bedrock model / inference-profile id passed to Converse.
        enable_thinking: Extended thinking. **Default ON.** Toggleable.
        thinking_budget: Token budget for thinking when enabled.
        max_tokens / temperature / top_p: Inference config. ``temperature`` and
            ``top_p`` are overridden by the thinking constraint when thinking is on.
        client: Optional pre-built bedrock-runtime client. Mainly for tests — when
            supplied, no boto3 session/credentials are needed.
    """

    def __init__(
        self,
        profile_name: Optional[str] = None,
        region_name: str = CURVE_REGION,
        model_id: str = CURVE_MODEL_ID,
        enable_thinking: bool = True,
        thinking_budget: int = CURVE_THINKING_BUDGET,
        max_tokens: int = CURVE_MAX_TOKENS,
        temperature: float = CURVE_TEMPERATURE,
        top_p: float = CURVE_TOP_P,
        client: Any = None,
    ):
        self.profile_name = profile_name
        self.region_name = region_name
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._client = client

    # -- client construction (lazy, so building the wrapper needs no creds) ------

    def _build_client(self) -> Any:
        """Build the bedrock-runtime client.

        Profile precedence: explicit ``profile_name`` → ``AWS_PROFILE`` env →
        default credential chain (role-based). Never hardcodes a profile in a way
        that breaks the in-Lambda role path.
        """
        if self.profile_name:
            session = boto3.Session(profile_name=self.profile_name)
        elif os.environ.get("AWS_PROFILE"):
            session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
        else:
            session = boto3.Session()  # default credential chain (role, env, etc.)
        return session.client("bedrock-runtime", region_name=self.region_name)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # -- the Converse call ------------------------------------------------------

    def build_inference_config(self) -> Dict[str, Any]:
        """Assemble the inferenceConfig dict **fully** before it's used.

        This is the bug fix: the shared wrapper bound an empty dict into the
        request and then rebound a *local* — here we build the complete dict and
        return it, and the caller puts exactly this into the request.
        """
        inference_config: Dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.enable_thinking:
            # Thinking requires temperature == 1.0 and topP omitted.
            inference_config["temperature"] = 1.0
        else:
            inference_config["temperature"] = self.temperature
            if self.top_p is not None:
                inference_config["topP"] = self.top_p
        return inference_config

    def converse(
        self,
        messages: List[Dict[str, Any]],
        system: List[Dict[str, Any]],
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call Bedrock Converse once and return the raw response.

        Args:
            messages: Converse ``messages`` list (roles user/assistant, content blocks).
            system: Converse ``system`` list, e.g. ``[{"text": "..."}]``.
            tool_config: Converse ``toolConfig`` (``{"tools": [...]}``) or ``None``.
        """
        inference_config = self.build_inference_config()

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "system": system,
            # Populated config goes in directly — not an aliased empty dict.
            "inferenceConfig": inference_config,
        }

        if tool_config is not None:
            kwargs["toolConfig"] = tool_config

        if self.enable_thinking:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}
            }

        return self.client.converse(**kwargs)
