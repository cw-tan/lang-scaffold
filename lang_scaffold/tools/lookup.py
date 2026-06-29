from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model


def build_lookup_tool(
    data: dict[str, Any],
    name: str,
    description: str,
    key_description: str = "which document to retrieve",
) -> BaseTool:
    """Make a retrieval tool over ``data``.
    The model picks a key (typed as a ``Literal`` of ``data``'s keys) and gets the matching value back.
    """
    if not data:
        raise ValueError("data must be non-empty")
    key_type = Literal[tuple(data)]  # enum of the keys, straight from the data
    args = create_model(
        f"{name}_args", key=(key_type, Field(description=key_description))
    )
    return StructuredTool.from_function(
        func=lambda key: str(data[key]),
        name=name,
        description=description,
        args_schema=args,
    )
