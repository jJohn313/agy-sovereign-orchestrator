"""
Gemini 4 Argon Frontier Model Adapter for AGY Sovereign Orchestrator.
Calls Google Generative Language API using API key from ~/.config/gemini/api_key.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

# Import AgentResponse and ToolCall from orchestrator if possible
try:
    from orchestrator import AgentResponse, ToolCall
except ImportError:
    class ToolCall:
        def __init__(self, name: str, arguments: Dict[str, Any], call_id: Optional[str] = None):
            self.name = name
            self.arguments = arguments
            self.call_id = call_id

    class AgentResponse:
        def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None):
            self.content = content or ""
            self.tool_calls = tool_calls or []

API_KEY_PATH = os.path.expanduser("~/.config/gemini/api_key")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent"


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(API_KEY_PATH):
        with open(API_KEY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise ValueError(f"Gemini API key not found in GEMINI_API_KEY or {API_KEY_PATH}")


def call_gemini_argon(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 8192,
) -> AgentResponse:
    """
    Adapter callable for orchestrator.run_agent_loop(call_frontier_model=call_gemini_argon).
    Translates serialized messages and tool schemas to Gemini generateContent format.
    """
    api_key = get_api_key()
    url = f"{ENDPOINT}?key={api_key}"

    # Convert messages to Gemini contents format
    gemini_contents = []
    system_instruction_text = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_instruction_text += f"{content}\n"
            continue

        gemini_role = "user" if role == "user" else "model"
        gemini_contents.append({
            "role": gemini_role,
            "parts": [{"text": str(content)}]
        })

    payload: Dict[str, Any] = {
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
    }

    if system_instruction_text.strip():
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction_text.strip()}]
        }

    # Format tools if provided
    if tools:
        func_decls = []
        for t in tools:
            # Map standard JSON schema to Gemini function_declaration
            func = t.get("function", t)
            decl = {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            }
            func_decls.append(decl)
        if func_decls:
            payload["tools"] = [{"function_declarations": func_decls}]

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini Argon API HTTP {e.code}: {err}") from e

    # Parse candidate
    candidates = data.get("candidates", [])
    if not candidates:
        return AgentResponse(content="No response from Gemini 4 Argon.")

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])

    text_parts = []
    tool_calls = []

    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(ToolCall(
                name=fc.get("name", ""),
                arguments=fc.get("args", {}),
                call_id=fc.get("id", None)
            ))

    # Log usage to stderr
    usage = data.get("usageMetadata", {})
    sys.stderr.write(
        f"[Frontier Argon] Prompt: {usage.get('promptTokenCount', 0)}, "
        f"Output: {usage.get('candidatesTokenCount', 0)}\n"
    )

    return AgentResponse(
        content="".join(text_parts),
        tool_calls=tool_calls
    )
