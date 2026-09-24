from typing import Dict, Any, Optional

def truncate_description(desc: Optional[str], max_words: int = 12) -> str:
    """Trim verbose parameter/tool descriptions down to max_words."""
    if not desc:
        return ""
    words = desc.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."

def minify_tool_schema(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minify tool schema by removing unnecessary metadata and truncating descriptions.
    Retains: name, type, properties, required, enum, items, default.
    Strips: markdownDescription, examples, example, deprecated, $comment, $schema, title, etc.
    Truncates: description (top level and nested properties).
    """
    if not isinstance(tool_def, dict):
        return tool_def

    minified: Dict[str, Any] = {}

    # Keys we want to explicitly retain in any object (including tools and parameters)
    # Plus 'name' for the top-level tool definition.
    retained_keys = {"name", "type", "properties", "required", "enum", "items", "default"}

    for key, value in tool_def.items():
        if key in retained_keys:
            if key == "properties" and isinstance(value, dict):
                minified_props = {}
                for prop_name, prop_val in value.items():
                    minified_props[prop_name] = minify_tool_schema(prop_val)
                minified[key] = minified_props
            elif key == "items" and isinstance(value, dict):
                minified[key] = minify_tool_schema(value)
            else:
                minified[key] = value
        elif key == "description":
            if isinstance(value, str):
                truncated = truncate_description(value)
                if truncated:
                    minified[key] = truncated
            else:
                # keep as is if it's not a string? Or drop? Assuming description is string.
                pass
        elif key == "parameters" and isinstance(value, dict):
            # Special case for top-level tool 'parameters' which holds the schema
            minified[key] = minify_tool_schema(value)

    return minified
