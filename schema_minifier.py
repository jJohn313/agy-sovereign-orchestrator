from typing import Dict, Any, Optional, Set

EXEMPT_TOOLS: Set[str] = {
    "edit_file",
    "patch",
    "run_command",
    "sem-find",
    "execute_command",
    "symbol_lookup",
    "write_file",
}

def truncate_description(desc: Optional[str], max_words: int = 12) -> str:
    """Trim verbose parameter/tool descriptions down to max_words."""
    if not desc:
        return ""
    words = desc.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."

def minify_tool_schema(
    tool_def: Dict[str, Any],
    exempt_tools: Optional[Set[str]] = None,
    is_exempt: bool = False,
) -> Dict[str, Any]:
    """
    Minify tool schema by removing unnecessary metadata and truncating descriptions.
    Retains: name, type, properties, required, enum, items, default.
    Strips: markdownDescription, examples, example, deprecated, $comment, $schema, title, etc.
    Truncates: description (top level and nested properties) unless tool is exempt.
    """
    if not isinstance(tool_def, dict):
        return tool_def

    if exempt_tools is None:
        exempt_tools = EXEMPT_TOOLS

    tool_name = tool_def.get("name")
    if tool_name and tool_name in exempt_tools:
        is_exempt = True

    minified: Dict[str, Any] = {}

    # Keys we want to explicitly retain in any object (including tools and parameters)
    # Plus 'name' for the top-level tool definition.
    retained_keys = {"name", "type", "properties", "required", "enum", "items", "default"}

    for key, value in tool_def.items():
        if key in retained_keys:
            if key == "properties" and isinstance(value, dict):
                minified_props = {}
                for prop_name, prop_val in value.items():
                    minified_props[prop_name] = minify_tool_schema(
                        prop_val, exempt_tools=exempt_tools, is_exempt=is_exempt
                    )
                minified[key] = minified_props
            elif key == "items" and isinstance(value, dict):
                minified[key] = minify_tool_schema(
                    value, exempt_tools=exempt_tools, is_exempt=is_exempt
                )
            else:
                minified[key] = value
        elif key == "description":
            if isinstance(value, str):
                if is_exempt:
                    minified[key] = value.strip()
                else:
                    truncated = truncate_description(value)
                    if truncated:
                        minified[key] = truncated
        elif key == "parameters" and isinstance(value, dict):
            # Special case for top-level tool 'parameters' which holds the schema
            minified[key] = minify_tool_schema(
                value, exempt_tools=exempt_tools, is_exempt=is_exempt
            )

    return minified

    return minified
