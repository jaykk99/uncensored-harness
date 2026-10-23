"""Omni provider pool for the uncensored harness.

Ports the keyless + keyed provider definitions from Error Inbox's omni route
(lib/llm/llm.ts) into Python so the harness is not limited to Horde +
Pollinations. Every provider here is OpenAI-compatible (/chat/completions)
except Horde (async queue, handled in models.py) and Cohere (/v2/chat).

Uncensored priority: Horde's Stheno/Heretic/Rocinante workers stay primary.
Keyless lightly-filtered models (LLM7, Kilo) are the reliability backstop.
Keyed no-refusal models (DeepInfra/Together Dolphin, OpenRouter Hermes)
lead the keyed section when their keys are set.
Never routed: GPT, Claude, Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Reuse the proxy/CA helpers from models (same process, same egress).
from .models import _client, RETRYABLE_STATUS


@dataclass
class ProviderDef:
    name: str            # 'kilo', 'llm7', 'groq', ...
    label: str           # 'Kilo Gateway' (shown in UI)
    company: str         # 'Kilo AI' (smaller text in UI)
    base_url: str        # full /chat/completions endpoint
    key_env: str | None  # env var holding the key; None = keyless
    models: list[str] = field(default_factory=list)
    uncensored: bool = False   # genuinely uncensored lineage (dolphin/hermes/heretic)
    special: str = ""          # 'horde' | 'cohere' | '' (non-standard call shape)

    @property
    def keyless(self) -> bool:
        return self.key_env is None


def _key(p: ProviderDef) -> str:
    return (os.environ.get(p.key_env, "").strip() if p.key_env else "")


def key_is_set(p: ProviderDef) -> bool:
    return p.keyless or bool(_key(p))


# ---------------------------------------------------------------------------
# Keyless providers (work with zero keys)
# ---------------------------------------------------------------------------

KILO_MODELS = [
    "kilo-auto/free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "deepseek-v4-flash:free",
    "stepfun/step-3.7-flash:free",
    "dots-studio/dots-3-note-preview:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "inclusionai/ling-3.1-flash",
    "liquid/lfm-2.5-2.6b:free",
    "thinkingmachines/inkling-small:free",
    "minimax-m2.7:free",
]

LLM7_MODELS = [
    "codestral-latest",
    "minimax-m2.7",
    "mistral-Nemo-Instruct-2407",
]

# ---------------------------------------------------------------------------
# Keyed providers (code included; UI marks "needs key" until env var is set)
# ---------------------------------------------------------------------------

KEYED_PROVIDERS: list[ProviderDef] = [
    ProviderDef(
        name="deepinfra", label="DeepInfra", company="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai/chat/completions",
        key_env="DEEPINFRA_API_KEY",
        models=[
            "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        uncensored=True,
    ),
    ProviderDef(
        name="together", label="Together AI", company="Together",
        base_url="https://api.together.xyz/v1/chat/completions",
        key_env="TOGETHER_API_KEY",
        models=[
            "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ],
        uncensored=True,
    ),
    ProviderDef(
        name="openrouter", label="OpenRouter", company="OpenRouter",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        key_env="OPENROUTER_API_KEY",
        models=[
            "nousresearch/hermes-3-llama-3.1-70b",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "nvidia/nemotron-3.5-lightning:free",
            "thinkingmachines/inkling:free",
            "thinkingmachines/inkling-small:free",
            "qwen/qwen3.8-27b:free",
            "dots-studio/dots-3-note-preview:free",
            "liquid/lfm-2.5-2.6b:free",
            "inclusionai/ling-3.0-flash-sante:free",
            "cohere/north-mini-code:free",
        ],
        uncensored=True,
    ),
    ProviderDef(
        name="groq", label="Groq", company="Groq",
        base_url="https://api.groq.com/openai/v1/chat/completions",
        key_env="GROQ_API_KEY",
        models=["openai/gpt-oss-120b"],
    ),
    ProviderDef(
        name="cerebras", label="Cerebras", company="Cerebras",
        base_url="https://api.cerebras.ai/v1/chat/completions",
        key_env="CEREBRAS_API_KEY",
        models=["llama-3.3-70b", "qwen-3-32b", "gpt-oss-120b"],
    ),
    ProviderDef(
        name="sambanova", label="SambaNova", company="SambaNova",
        base_url="https://api.sambanova.ai/v1/chat/completions",
        key_env="SAMBANOVA_API_KEY",
        models=[
            "Meta-Llama-3.3-70B-Instruct",
            "Llama-4-Maverick-17B-128E-Instruct",
            "DeepSeek-V3-0324",
        ],
    ),
    ProviderDef(
        name="nvidia", label="NVIDIA NIM", company="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        key_env="NVIDIA_API_KEY",
        models=["z-ai/glm-5.2", "z-ai/glm-5.1", "poolside/laguna-xs-2.1"],
    ),
    ProviderDef(
        name="mistral", label="Mistral", company="Mistral AI",
        base_url="https://api.mistral.ai/v1/chat/completions",
        key_env="MISTRAL_API_KEY",
        models=["mistral-small-latest", "open-mistral-nemo", "codestral-latest"],
    ),
    ProviderDef(
        name="fireworks", label="Fireworks", company="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1/chat/completions",
        key_env="FIREWORKS_API_KEY",
        models=["accounts/fireworks/models/llama-v3p3-70b-instruct"],
    ),
    ProviderDef(
        name="huggingface", label="HuggingFace", company="HuggingFace",
        base_url="https://router.huggingface.co/v1/chat/completions",
        key_env="HF_TOKEN",
        models=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
    ),
    ProviderDef(
        name="cohere", label="Cohere", company="Cohere",
        base_url="https://api.cohere.com/v2/chat",
        key_env="COHERE_API_KEY",
        models=["command-a-03-2025", "command-r-plus-08-2024"],
        special="cohere",
    ),
    ProviderDef(
        name="gratisfy", label="Gratisfy", company="Gratisfy",
        base_url="https://api.gratisfy.xyz/v1/chat/completions",
        key_env="GRATISFY_API_KEY",
        models=[],
    ),
]

KEYLESS_PROVIDERS: list[ProviderDef] = [
    ProviderDef(
        name="llm7", label="LLM7", company="LLM7",
        base_url="https://api.llm7.io/v1/chat/completions",
        key_env=None,
        models=LLM7_MODELS,
    ),
    ProviderDef(
        name="kilo", label="Kilo Gateway", company="Kilo AI",
        base_url="https://api.kilo.ai/api/gateway/chat/completions",
        key_env=None,
        models=KILO_MODELS,
    ),
]

ALL_PROVIDERS: list[ProviderDef] = KEYLESS_PROVIDERS + KEYED_PROVIDERS


def get_provider(name: str) -> ProviderDef | None:
    for p in ALL_PROVIDERS:
        if p.name == name:
            return p
    return None


def parse_model_id(model_id: str) -> tuple[str | None, str]:
    """Split 'provider:model' into (provider_name, model). Returns (None, id)
    when there is no known provider prefix (e.g. a bare Horde model id)."""
    if ":" in model_id:
        prefix, rest = model_id.split(":", 1)
        if get_provider(prefix):
            return prefix, rest
    return None, model_id


def call_openai_compatible(p: ProviderDef, model: str, messages: list[dict],
                           max_tokens: int = 1200, temperature: float = 0.7,
                           timeout: int = 60,
                           tools: list[dict] | None = None,
                           on_token=None) -> dict:
    """POST to an OpenAI-compatible /chat/completions endpoint.

    Returns the raw assistant message dict. Raises on any failure.
    Supports SSE streaming via on_token (each content chunk forwarded).
    """
    key = _key(p)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body: dict = {
        "model": model,
        "max_tokens": max(700, max_tokens),
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    if on_token is not None:
        return _stream_openai_compatible(p, headers, body, timeout, on_token)

    with _client(timeout) as c:
        r = c.post(p.base_url, json=body, headers=headers)
    if r.status_code in RETRYABLE_STATUS:
        raise RuntimeError(f"{p.name} HTTP {r.status_code}")
    if r.status_code >= 400:
        raise RuntimeError(f"{p.name} HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"{p.name}: empty choices")
    msg = choices[0].get("message") or {}
    # Some Kilo free models are reasoning-only: content null, text in reasoning.
    content = (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()
    msg["content"] = content
    if not content and not msg.get("tool_calls"):
        raise RuntimeError(f"{p.name}: empty response")
    return msg


def _stream_openai_compatible(p: ProviderDef, headers: dict, body: dict,
                             timeout: int, on_token) -> dict:
    """SSE-streaming variant. Forwards each content chunk to on_token."""
    stream_body = dict(body)
    stream_body["stream"] = True
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    with _client(timeout) as c:
        with c.stream("POST", p.base_url, json=stream_body,
                      headers=headers) as r:
            if r.status_code in RETRYABLE_STATUS:
                raise RuntimeError(f"{p.name} HTTP {r.status_code}")
            if r.status_code >= 400:
                raise RuntimeError(f"{p.name} HTTP {r.status_code}")
            for raw_line in r.iter_lines():
                line = (raw_line if isinstance(raw_line, str)
                        else raw_line.decode("utf-8", "replace")).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    try:
                        on_token(piece)
                    except Exception:
                        pass
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    entry = tool_calls.setdefault(
                        idx, {"id": "", "type": "function",
                              "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]
    if not content_parts and not tool_calls:
        raise RuntimeError(f"{p.name}: stream ended with no content")
    msg: dict = {"content": "".join(content_parts)}
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg


def call_cohere(p: ProviderDef, model: str, messages: list[dict],
                max_tokens: int = 1200, temperature: float = 0.7,
                timeout: int = 60) -> dict:
    """Cohere /v2/chat has its own shape. Returns an OpenAI-style message."""
    key = _key(p)
    # Flatten to a single user string; Cohere v2 takes messages but the
    # simple path is reliable across model variants.
    parts = []
    for m in messages:
        role, content = m.get("role", ""), m.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") for x in content
                               if isinstance(x, dict))
        parts.append(f"{role}: {content}")
    with _client(timeout) as c:
        r = c.post(
            p.base_url,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "\n\n".join(parts)}],
                  "max_tokens": max(700, max_tokens),
                  "temperature": temperature},
        )
    if r.status_code in RETRYABLE_STATUS:
        raise RuntimeError(f"cohere HTTP {r.status_code}")
    if r.status_code >= 400:
        raise RuntimeError(f"cohere HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    text = ""
    msg = data.get("message") or {}
    content = msg.get("content") or []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    text = text.strip()
    if not text:
        raise RuntimeError("cohere: empty response")
    return {"content": text}


def call_provider(p: ProviderDef, model: str, messages: list[dict],
                  max_tokens: int = 1200, temperature: float = 0.7,
                  timeout: int = 60, tools: list[dict] | None = None,
                  on_token=None, progress=None) -> dict:
    """Dispatch to the right caller for this provider. Returns raw message."""
    if progress:
        progress(f"{p.label}: trying {model}…")
    if p.special == "cohere":
        return call_cohere(p, model, messages, max_tokens, temperature, timeout)
    return call_openai_compatible(p, model, messages, max_tokens, temperature,
                                  timeout, tools, on_token)


def list_models() -> list[dict]:
    """Every model from every wired provider, with live key status.

    Returned shape (for /api/models and the UI picker):
      { id, label, provider, company, keyless, live, uncensored }
    id is 'provider:model' so the router can parse it back.
    """
    out: list[dict] = []
    # Horde uncensored set lives in models.py; import lazily to avoid cycles.
    try:
        from .models import HORDE_UNCENSORED_MODELS
        for m in HORDE_UNCENSORED_MODELS:
            short = m.split("/")[-1]
            out.append({
                "id": f"horde:{m}", "label": short,
                "provider": "horde", "company": "AI Horde",
                "keyless": True, "live": True, "uncensored": True,
            })
    except Exception:
        pass
    for p in ALL_PROVIDERS:
        live = key_is_set(p)
        for m in p.models:
            short = m.split("/")[-1].split(":")[0]
            out.append({
                "id": f"{p.name}:{m}", "label": short,
                "provider": p.name, "company": p.company,
                "keyless": p.keyless, "live": live,
                "uncensored": p.uncensored,
            })
    # Pollinations fallback entry (existing behavior, kept for tool calls).
    out.append({
        "id": "pollinations:openai", "label": "gpt-oss-20b",
        "provider": "pollinations", "company": "Pollinations",
        "keyless": True, "live": True, "uncensored": False,
    })
    return out


def _health_penalty(provider_name: str, model: str,
                    by_leg: dict[tuple[str, str], dict]) -> int:
    """Demotion penalty (0 = keep, 1 = bad stats, 2 = stale) for one leg.

    A leg is only demoted on real evidence: >=3 attempts with under 50%
    success, or no data within ROUTER_STALE_HOURS. Thin data and unknown
    legs stay in static order.
    """
    row = by_leg.get((provider_name, model))
    if not row:
        return 0
    if row.get("stale"):
        return 2
    try:
        attempts = int(row.get("attempts") or 0)
        rate = float(row.get("success_rate") or 0.0)
    except (TypeError, ValueError):
        return 0
    return 1 if attempts >= 3 and rate < 0.5 else 0


def _health_order_enabled() -> bool:
    return os.environ.get("ROUTER_HEALTH_ORDER", "1").lower() not in (
        "0", "false", "no", "off")


def omni_chain(preferred_id: str = "",
               stats: list[dict] | None = None) -> list[tuple[ProviderDef | None, str]]:
    """Ordered fallback chain of (provider_def, model) tuples.

    Order: explicit pick first, then uncensored-keyed, then keyless
    (LLM7, Kilo), then remaining keyed providers with keys set.
    Horde and Pollinations are handled by models.chat's existing legs;
    this chain covers everything in between.
    Each entry is tried with a short timeout; failures move to the next.

    stats (router_stats.snapshot() rows) enables health-aware ordering:
    within a tier, legs with poor recent stats (>=3 attempts, <50%
    success) or stale data are tried last. Tier order itself never
    changes — the uncensored-first promise holds regardless of stats.
    Set ROUTER_HEALTH_ORDER=0 to force static order.
    """
    tiers: list[tuple[int, ProviderDef, str]] = []
    seen: set[str] = set()

    def push(tier: int, p: ProviderDef, model: str):
        k = f"{p.name}/{model}"
        if k not in seen:
            seen.add(k)
            tiers.append((tier, p, model))

    # 0) Explicit pick from the UI.
    if preferred_id:
        prov_name, model = parse_model_id(preferred_id)
        if prov_name:
            p = get_provider(prov_name)
            if p and key_is_set(p):
                push(0, p, model)
    # 1) Uncensored keyed models first (Dolphin/Hermes carry the load when
    #    keys exist; they answer in seconds, not Horde-queue minutes).
    for p in KEYED_PROVIDERS:
        if p.uncensored and key_is_set(p):
            for m in p.models:
                push(1, p, m)
    # 2) Keyless reliability backstop.
    for p in KEYLESS_PROVIDERS:
        for m in p.models:
            push(2, p, m)
    # 3) Remaining keyed providers with keys set.
    for p in KEYED_PROVIDERS:
        if not p.uncensored and key_is_set(p):
            for m in p.models:
                push(3, p, m)

    if stats and _health_order_enabled():
        by_leg = {(r.get("provider", ""), r.get("model", "")): r
                  for r in stats}
        # Stable sort: within (tier, penalty) the static relative order is
        # preserved, so good legs never shuffle around.
        tiers.sort(key=lambda t: (
            t[0], _health_penalty(t[1].name, t[2], by_leg)))
    return [(p, m) for _, p, m in tiers]
