import types
from typing import Any, Literal, Optional, Type, Union, get_args, get_origin

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError, create_model

from lang_scaffold.utils import schema_summary


class ExtractionState(BaseModel):
    """State for the extraction loop graph.

    The client owns the multi-turn loop: it sets a new ``user_input`` each turn,
    passes the rest of the state back in, and decides when to stop.
    Only ``user_input`` is required to seed a conversation; everything else defaults.
    """

    user_input: str = Field(description="New user input for this turn")
    filled: dict[str, Any] = Field(
        default_factory=dict, description="Accumulated raw fields"
    )
    messages: list[BaseMessage] = Field(
        default_factory=list, description="Running conversation transcript"
    )
    result: Optional[BaseModel] = Field(
        default=None, description="Validated target model; None until complete"
    )
    agent_message: str = Field(
        default="", description="Follow-up question to the user (empty unless asking)"
    )


class _Routing(BaseModel):
    """Router decision taken when an extraction fails validation."""

    decision: Literal["ask", "retry"] = Field(
        description=(
            "'retry' if you misread the conversation and can extract correctly on "
            "another pass; 'ask' if the user must supply or correct information."
        )
    )
    message: str = Field(
        description="The message to the user when asking, or a fix-it note when retrying."
    )


def _relax(annotation: Any) -> Any:
    """Recursively relax an annotation so nested models become optional too."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _make_optional(annotation)
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):  # Optional[Model], Model | None, ...
        return Union[tuple(_relax(a) for a in get_args(annotation))]
    if origin is list:  # list[Model]
        return list[_relax(get_args(annotation)[0])]
    return annotation


def _make_optional(model: Type[BaseModel]) -> Type[BaseModel]:
    """Build an all-optional variant of ``model`` for extraction.

    Required fields would otherwise force the LLM to invent values to satisfy
    the structured-output schema. Making every field optional (default None)
    lets it return null for what it doesn't know, while field descriptions are
    preserved so the schema stays informative. Nested models are relaxed
    recursively (so partial nested objects are allowed); validators are not
    copied -- validity is checked against the strict ``model`` instead.
    """
    fields = {
        name: (
            Optional[_relax(f.annotation)],
            Field(default=None, description=f.description),
        )
        for name, f in model.model_fields.items()
    }
    return create_model(f"{model.__name__}Partial", **fields)


def build_extraction_loop(
    llm: BaseLanguageModel,
    model: Type[BaseModel],
    context_prompt: str,
    retry_cap: int = 6,
) -> Any:
    """
    Build a LangGraph that fills a Pydantic model from conversation, one turn
    per invocation.

    Each turn extracts into ``model`` and validates it. On a validation failure
    an LLM router decides whether to silently re-extract (it misread the input)
    or to ask the user (information is missing or genuinely invalid), re-extracting
    up to ``retry_cap`` times before falling back to asking. The client drives the
    multi-turn loop, re-invoking with the returned state until ``result`` is not
    None (complete) or it decides to give up.

    Args:
        llm: Initialized language model with structured output support.
        model: Pydantic model to extract into. A field is treated as REQUIRED
            (asked for until provided) iff it has NO default; any default --
            including ``Optional[X] = None`` -- makes it optional. Requiredness
            is governed by the default alone, NOT the type: ``Optional[X]``
            without a default is still required (Pydantic v2). Field- and
            model-level validators are enforced too -- a present-but-invalid
            value is handled like a missing one.
        context_prompt: System context for the LLM, used for both extraction and
            routing/phrasing.
        retry_cap: Max self-correction re-extractions per turn before asking the
            user (default 3).

    Returns:
        Compiled graph ready to invoke. Note ``invoke`` returns a plain dict;
        wrap it back into ``ExtractionState`` if you want attribute access.
    """
    # include_raw=True: malformed output surfaces as parsing_error instead of
    # raising, so we can retry it; API/transport errors still raise (propagate)
    structured_llm = llm.with_structured_output(_make_optional(model), include_raw=True)
    router_llm = llm.with_structured_output(_Routing)
    schema = schema_summary(model)
    extract_system = (
        f"{context_prompt}\n\n"
        f"Your task is to fill in this schema from what the user provides:\n{schema}\n\n"
        "Extract only information the user has explicitly provided. For any field the "
        "user has not given, leave it null -- do NOT guess, infer, fabricate to satisfy "
        "'required', or fill in placeholders such as 'unknown', 'N/A', or 'none'."
    )

    def extract_node(state: ExtractionState) -> dict:
        history = [*state.messages, HumanMessage(content=state.user_input)]
        filled = dict(state.filled)
        feedback = ""
        errors = ""
        for _ in range(1 + retry_cap):
            system = extract_system
            if feedback:  # a prior attempt was rejected -- steer the re-extraction
                system += (
                    f"\n\nYour previous extraction was invalid ({feedback}). Re-read "
                    "the conversation and correct it; do not invent or alter values."
                )
            out = structured_llm.invoke([SystemMessage(content=system), *history])
            # parse failure = output doesn't fit the types-only partial schema.
            # always the LLM's fault (the user can't fix a malformed tool call),
            # so the action is never in doubt -> retry directly, skip the router.
            if out["parsing_error"]:
                feedback = "the output was not valid structured data"
                continue
            filled = {**filled, **out["parsed"].model_dump(exclude_none=True)}

            # well-typed data that fails the strict model (required field or validator) is ambiguous
            # (the LLM misread, or the user's data is genuinely bad/missing) so the router decides retry vs ask
            try:
                result = model(**filled)
                return {
                    "filled": filled,
                    "result": result,
                    "messages": history,
                    "agent_message": "",
                }
            except ValidationError as e:
                errors = "; ".join(
                    f"{'.'.join(str(x) for x in err['loc']) or '(model)'}: {err['msg']}"
                    for err in e.errors()
                )
                routing = router_llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                f"{context_prompt}\n\n"
                                f"The task is to fill in this schema:\n{schema}\n\n"
                                "An extraction attempt failed validation. Choose:\n"
                                "- 'retry': you misread the conversation and can fix "
                                "it by re-extracting.\n"
                                "- 'ask': the user must supply or correct information "
                                "(you cannot fix it by re-reading).\n"
                                "Never fabricate values to pass validation.\n"
                                "When you 'ask', write a brief, friendly message that "
                                "requests the missing/invalid info and briefly explains "
                                "why it is needed or why the value was rejected.\n\n"
                                f"Extracted so far: {filled}\n"
                                f"Validation errors: {errors}"
                            )
                        ),
                        *history,
                    ]
                )
                if routing.decision == "ask":
                    return {
                        "filled": filled,
                        "result": None,
                        "messages": [*history, AIMessage(content=routing.message)],
                        "agent_message": routing.message,
                    }
                feedback = routing.message

        # retries exhausted -> ask the user, surfacing the last failure detail
        msg = f"I'm still having trouble: {errors or feedback}. Could you help me correct that?"
        return {
            "filled": filled,
            "result": None,
            "messages": [*history, AIMessage(content=msg)],
            "agent_message": msg,
        }

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", END)
    return graph.compile()
