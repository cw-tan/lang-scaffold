from collections import defaultdict
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import Field, create_model

_REASON = "reason"


def with_rationale(tool: BaseTool) -> BaseTool:
    """Add a required ``reason`` field so the model must justify each call.
    Preserves the tool's other config; ``reason`` is stripped before it runs.
    """
    base = tool.args_schema
    # reason first, then the original fields -- nudges "justify, then act"
    fields = {
        _REASON: (
            str,
            Field(
                description="one sentence stating what you expect this call to reveal "
                "and how that advances answering the user's question"
            ),
        )
    }
    fields.update((n, (f.annotation, f)) for n, f in base.model_fields.items())
    schema = create_model(f"{base.__name__}WithReason", **fields)

    # augment the existing tool -- model_copy keeps metadata, tags, etc.
    # only the schema and the callable change, so nothing has to be re-propagated by hand
    updates = {"args_schema": schema}
    if (func := getattr(tool, "func", None)) is not None:

        def run(**kwargs):
            kwargs.pop(_REASON, None)  # the model's justification, not for the tool
            return func(**kwargs)

        updates["func"] = run
    if (coro := getattr(tool, "coroutine", None)) is not None:

        async def arun(**kwargs):
            kwargs.pop(_REASON, None)
            return await coro(**kwargs)

        updates["coroutine"] = arun
    return tool.model_copy(update=updates)


def describe(spec):
    """Attach a user-readable, per-call description to a tool (str template or callable)."""

    def wrap(tool: BaseTool) -> BaseTool:
        tool.metadata = {**(tool.metadata or {}), "describe": spec}
        return tool

    return wrap


def describe_call(tool: BaseTool, args: dict) -> Optional[str]:
    """Render a tool call for the user, or None if the tool opts out of display."""
    spec = (tool.metadata or {}).get("describe")
    if spec is None:
        return None
    # defaultdict(str) -> missing template keys render as '' instead of KeyError-ing
    return spec(args) if callable(spec) else spec.format_map(defaultdict(str, args))
