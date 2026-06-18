from typing import Any, Literal, Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError, create_model


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
        description=(
            "If 'ask', a brief friendly message asking the user for the "
            "missing/invalid information. If 'retry', a short note on what to fix."
        )
    )


def _make_optional(model: Type[BaseModel]) -> Type[BaseModel]:
    """Build an all-optional variant of ``model`` for extraction.

    Required fields would otherwise force the LLM to invent values to satisfy
    the structured-output schema. Making every field optional (default None)
    lets it return null for what it doesn't know, while field descriptions are
    preserved so the schema stays informative. Validators are intentionally not
    copied -- validity is checked against the strict ``model`` instead.
    """
    fields = {
        name: (
            Optional[f.annotation],
            Field(default=None, description=f.description),
        )
        for name, f in model.model_fields.items()
    }
    return create_model(f"{model.__name__}Partial", **fields)


def build_extraction_loop(
    llm: BaseLanguageModel,
    model: Type[BaseModel],
    context_prompt: str,
    retry_cap: int = 3,
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
    structured_llm = llm.with_structured_output(_make_optional(model))
    router_llm = llm.with_structured_output(_Routing)
    extract_system = (
        f"{context_prompt}\n\n"
        "Extract only information the user has explicitly provided. For any field "
        "the user has not given, leave it null -- do NOT guess, infer, or fill in "
        "placeholders such as 'unknown', 'N/A', or 'none'."
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
            extracted = structured_llm.invoke(
                [SystemMessage(content=system), *history]
            ).model_dump(exclude_none=True)
            filled = {**filled, **extracted}

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
                                "An extraction attempt failed validation. Choose:\n"
                                "- 'retry': you misread the conversation and can fix "
                                "it by re-extracting.\n"
                                "- 'ask': the user must supply or correct information "
                                "(you cannot fix it by re-reading).\n"
                                "Never fabricate values to pass validation.\n\n"
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

        # retries exhausted -> ask the user, surfacing the last errors
        msg = f"I'm still having trouble: {errors}. Could you help me correct that?"
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
