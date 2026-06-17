from typing import Any, Optional, Type

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
    missing: list[str] = Field(
        default_factory=list, description="Required fields still unfilled or invalid"
    )
    result: Optional[BaseModel] = Field(
        default=None, description="Validated target model; None until complete"
    )
    agent_message: str = Field(
        default="", description="Follow-up question to the user (empty unless asking)"
    )


def _make_optional(model: Type[BaseModel]) -> Type[BaseModel]:
    """Build an all-optional variant of ``model`` for extraction.

    Required fields would otherwise force the LLM to invent values to satisfy
    the structured-output schema. Making every field optional (default None)
    lets it return null for what it doesn't know, while field descriptions are
    preserved so the schema stays informative.
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
    llm_followup: bool = True,
) -> Any:
    """
    Build a LangGraph that fills a Pydantic model from conversation, one turn per invocation.

    Each invocation runs a single extract -> decide cycle: it merges newly
    extracted fields into ``filled``, then either validates the complete target
    model (stored in ``result``) or generates a conversational follow-up asking
    for the missing fields. The client drives the multi-turn loop, re-invoking
    with the returned state until ``result`` is not None (complete) or it decides
    to give up — the timeout/turn budget is entirely the client's call.

    Args:
        llm: Initialized language model with structured output support.
        model: Pydantic model to extract into. A field is treated as REQUIRED
            (asked for until provided) iff it has NO default; any default --
            including ``Optional[X] = None`` -- makes it optional, so it is
            never requested and just takes its default when absent.
            Requiredness is governed by the default alone, NOT by the type:
            ``Optional[X]`` without a default is still required (Pydantic v2).
            Use ``X = <value>`` for an optional with a real default and
            ``Optional[X] = None`` for an optional that may be unknown.
        context_prompt: System context for the LLM, used both for extraction
            and for phrasing follow-up questions.
        llm_followup: If True (default), an LLM call phrases a friendly,
            contextual question for the missing fields. If False, skip that call
            and use a terse auto-generated message -- cheaper, and what
            non-interactive/batch clients want (they can ignore the message).

    Returns:
        Compiled graph ready to invoke. Note ``invoke`` returns a plain dict;
        wrap it back into ``ExtractionState`` if you want attribute access.
    """
    structured_llm = llm.with_structured_output(_make_optional(model))
    descriptions = {n: f.description for n, f in model.model_fields.items()}
    extract_system = SystemMessage(
        content=(
            f"{context_prompt}\n\n"
            "Extract only information the user has explicitly provided. For any "
            "field the user has not given, leave it null -- do NOT guess, infer, "
            "or fill in placeholders such as 'unknown', 'N/A', or 'none'."
        )
    )

    def extract_node(state: ExtractionState) -> dict:
        """Extract from the transcript, then validate against the target model."""
        # append this turn's input, then extract from the whole conversation;
        # a failed extraction call propagates -- the client can catch and retry
        history = [*state.messages, HumanMessage(content=state.user_input)]
        extracted_model = structured_llm.invoke([extract_system, *history])
        extracted = extracted_model.model_dump(
            exclude_none=True
        )  # keep only found fields

        filled = {**state.filled, **extracted}

        # completeness == the strict target model validates; the validation
        # error doubles as the list of fields still missing or invalid
        try:
            result = model(**filled)
            missing = []
        except ValidationError as e:
            result = None
            missing = sorted({str(err["loc"][0]) for err in e.errors() if err["loc"]})

        return {
            "filled": filled,
            "missing": missing,
            "result": result,
            "messages": history,
        }

    def decide_and_message_node(state: ExtractionState) -> dict:
        """Clear the question when complete, else ask for what's still missing."""
        if not state.missing:
            return {"agent_message": ""}

        # describe each missing field (name + description) -- used by both paths
        need_lines = "\n".join(
            f"- {n}: {descriptions[n]}" if descriptions.get(n) else f"- {n}"
            for n in state.missing
        )
        auto_message = f"Please provide the following:\n{need_lines}"

        if llm_followup:
            # steer via the system message so the transcript stays clean
            # alternating turns; the model replies to the user's last human turn
            have = ", ".join(state.filled) or "nothing yet"
            ask_prompt = SystemMessage(
                content=(
                    f"{context_prompt}\n\n"
                    f"So far you have collected: {have}.\n"
                    f"You still require:\n{need_lines}\n\n"
                    "Reply with a message asking the user for the still-missing information, "
                    "with brief context for why it is needed. "
                    "If the user is reluctant or declines, politely but firmly explain it is required to proceed and ask again. "
                    "Do not give up, do not claim the task is complete, and do not move on until every required field is provided. "
                )
            )
            try:
                agent_message = llm.invoke([ask_prompt, *state.messages]).content
            except Exception:
                agent_message = auto_message
        else:
            agent_message = auto_message  # cheap path: no extra LLM call

        # store the question so the next turn's transcript includes it
        return {
            "agent_message": agent_message,
            "messages": [*state.messages, AIMessage(content=agent_message)],
        }

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.add_node("decide", decide_and_message_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "decide")
    graph.add_edge("decide", END)

    return graph.compile()
