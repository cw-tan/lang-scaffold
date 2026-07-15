from typing import Type, get_args

from pydantic import BaseModel


def _model_branches(annotation) -> list[Type[BaseModel]]:
    """All pydantic-model branches in an annotation (direct, Optional, list, Union).
    More than one means a union the model must choose among.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    out = []
    for arg in get_args(annotation):
        out += _model_branches(arg)
    return out


def schema_summary(model: Type[BaseModel], indent: int = 0) -> str:
    """Concise, flat field listing of a Pydantic model for prompts.

    Each field renders as ``- name (required|optional): description``; a single
    nested model is inlined with indentation, and a union of models is inlined
    under a ``one of:`` heading naming every variant -- no $ref indirection like
    ``model_json_schema()``, which weaker models handle less reliably.
    """
    pad = "  " * indent
    lines = []
    for name, f in model.model_fields.items():
        req = "required" if f.is_required() else "optional"
        lines.append(f"{pad}- {name} ({req}): {f.description or ''}")
        branches = _model_branches(f.annotation)
        if len(branches) == 1:
            lines.append(schema_summary(branches[0], indent + 1))
        elif len(branches) > 1:  # union: name the choice, then each variant's shape
            lines.append(f"{pad}  one of:")
            for b in branches:
                lines.append(f"{pad}    {b.__name__}:")
                lines.append(schema_summary(b, indent + 3))
    return "\n".join(lines)
