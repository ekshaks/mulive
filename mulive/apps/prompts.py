"""Prompt-file loading and tolerant JSON reading, shared by all apps.

A bundle's ``prompts.yml`` maps a prompt id to two blocks::

    coach_best:
      instructions: |
        You are a patient chess coach ...
      request: |
        Position: {fen}

``load_prompt_instructions`` reads the system half, ``load_prompt_request`` the
user half. ``extract_json_object`` reads a JSON object out of a model reply that
may be fenced or padded with prose.
"""

import json
import re
from pathlib import Path
from typing import Any

import yaml


def load_prompts(prompts_path: Path) -> dict[str, Any]:
    """Load a whole prompts file.

    Args:
        prompts_path: Path to a bundle's ``prompts.yml``.

    Returns:
        The parsed mapping, or an empty dict when the file is empty.
    """
    with open(prompts_path) as handle:
        return yaml.safe_load(handle) or {}


def load_prompt_instructions(prompts_path: Path, prompt_id: str) -> str:
    """Read ``prompts[prompt_id].instructions``.

    Args:
        prompts_path: Path to a bundle's ``prompts.yml``.
        prompt_id: Top-level key naming the prompt.

    Returns:
        The stripped instructions text.

    Raises:
        ValueError: When the prompt or its ``instructions`` block is missing.
    """
    return _prompt_section(prompts_path, prompt_id, "instructions")


def load_prompt_request(prompts_path: Path, prompt_id: str) -> str:
    """Read ``prompts[prompt_id].request``.

    Args:
        prompts_path: Path to a bundle's ``prompts.yml``.
        prompt_id: Top-level key naming the prompt.

    Returns:
        The stripped request template.

    Raises:
        ValueError: When the prompt or its ``request`` block is missing.
    """
    return _prompt_section(prompts_path, prompt_id, "request")


def _prompt_section(prompts_path: Path, prompt_id: str, section: str) -> str:
    prompts = load_prompts(prompts_path)
    text = ((prompts.get(prompt_id) or {}).get(section) or "").strip()
    if not text:
        raise ValueError(f"Missing prompt text: {prompt_id}.{section}")
    return text


def extract_json_object(text: str, *, strict: bool = False) -> dict:
    """Pull the first JSON object out of a model response.

    Handles ```json fences and objects embedded in prose.

    Args:
        text: Raw model reply.
        strict: When True, raise instead of returning an empty dict.

    Returns:
        The parsed object, or ``{}`` when nothing usable was found and
        ``strict`` is False.

    Raises:
        ValueError: In strict mode when no JSON object is present.
        json.JSONDecodeError: In strict mode when the candidate is malformed.
    """
    candidate = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    elif not candidate.startswith("{"):
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            if strict:
                raise ValueError("Model response did not contain JSON")
            return {}
        candidate = match.group(0)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        if strict:
            raise
        return {}
    if not isinstance(value, dict):
        if strict:
            raise ValueError("Model response must be a JSON object")
        return {}
    return value
