import os
from typing import Optional

import pytest
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from lang_scaffold import ExtractionState, build_extraction_loop
from lang_scaffold.monitor import TracedGraph


# Skip if API key not set
pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_API_KEY"),
    reason="LLM_API_KEY not set",
)


class PersonInfo(BaseModel):
    """Example model for extraction tests."""

    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    age: Optional[int] = Field(default=None, description="Age in years")


@pytest.fixture
def llm():
    """Initialize Anthropic LLM from env vars."""
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    return ChatAnthropic(api_key=api_key, model=model)


def test_single_turn_extraction(llm):
    """All required info given up front -> complete, validated model returned."""
    graph = build_extraction_loop(
        llm=llm,
        model=PersonInfo,
        context_prompt="Extract personal information from the user input.",
    )
    graph = TracedGraph(graph, verbose=True)

    result = ExtractionState(
        **graph.invoke(
            ExtractionState(
                user_input="My name is Alice Johnson and my email is alice@example.com"
            )
        )
    )

    assert isinstance(result.result, PersonInfo)  # complete: validated target model
    assert result.result.name and result.result.email
    assert result.agent_message == ""  # nothing missing, so no question


def test_multi_turn_extraction(llm):
    """Info arrives across turns; state accumulates and the agent asks for gaps."""
    graph = build_extraction_loop(
        llm=llm,
        model=PersonInfo,
        context_prompt="Extract personal information from the user input.",
    )
    graph = TracedGraph(graph, verbose=True)

    # first turn: only the name -> email still missing, agent should ask
    result1 = ExtractionState(
        **graph.invoke(ExtractionState(user_input="My name is Bob Smith"))
    )

    assert result1.result is None  # not complete -> no validated model yet
    assert result1.agent_message  # a follow-up question asking for the email

    # second turn: client feeds new input + prior state back in
    result1.user_input = "My email is bob@example.com"
    result2 = ExtractionState(**graph.invoke(result1))

    # both fields now present; transcript grew across turns
    assert isinstance(result2.result, PersonInfo)
    assert result2.result.name and result2.result.email
    assert len(result2.messages) > len(result1.messages)
