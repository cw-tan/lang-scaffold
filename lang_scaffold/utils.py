from typing import Optional, Type, get_args

from pydantic import BaseModel


def _inner_model(annotation) -> Optional[Type[BaseModel]]:
    """Find a nested pydantic model in an annotation (direct, Optional, list, ...)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _inner_model(arg)
        if found:
            return found
    return None


def schema_summary(model: Type[BaseModel], indent: int = 0) -> str:
    """Concise, flat field listing of a Pydantic model for prompts.

    Each field renders as ``- name (required|optional): description``; nested
    models are inlined with indentation -- no $ref indirection like
    ``model_json_schema()``, which weaker models handle less reliably.
    """
    pad = "  " * indent
    lines = []
    for name, f in model.model_fields.items():
        req = "required" if f.is_required() else "optional"
        lines.append(f"{pad}- {name} ({req}): {f.description or ''}")
        nested = _inner_model(f.annotation)
        if nested:
            lines.append(schema_summary(nested, indent + 1))
    return "\n".join(lines)
