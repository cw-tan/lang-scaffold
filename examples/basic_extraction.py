#!/usr/bin/env python3
"""Minimal example showing the client/library split.

The library gives a one-turn graph. The CLIENT owns the loop below: it invokes
the graph, does the I/O when the graph asks a question, threads the reply back
in, and owns the turn budget. Everything inside `graph.invoke` is the library.

Toggle INTERACTIVE:
  True  -> chat in the terminal; LLM I/O logged to llm.jsonl (inspect separately)
  False -> scripted replies, with PromptLogger printing the raw LLM I/O to stdout
"""

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from lang_scaffold import ExtractionState, build_extraction_loop
from lang_scaffold.monitor import PromptLogger

INTERACTIVE = True


class PersonInfo(BaseModel):
    """Fields we want to collect from the user."""

    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number")
    age: Optional[int] = Field(default=None, description="Age in years")
    country: str = Field(default="USA", description="Country of residence")


def main():
    # === set up LLM ===
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("Error: LLM_API_KEY not set")
        return
    model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    llm = ChatAnthropic(api_key=api_key, model=model)

    # === build graph ===
    graph = build_extraction_loop(
        llm=llm,
        model=PersonInfo,
        context_prompt="You are collecting a person's contact details.",
    )

    # === seed the first turn ===
    if INTERACTIVE:
        # LLM I/O goes to a JSONL file
        config = {"callbacks": [PromptLogger("llm.jsonl")]}
        print("Hi! Let's set up your contact details. Tell me about yourself.")
        state = ExtractionState(user_input=input("\n> "))
    else:
        config = {"callbacks": [PromptLogger()]}  # show the raw LLM I/O instead
        scripted = iter(["my email is alice@example.com and my phone is 555-0123"])
        state = ExtractionState(user_input="Hi, I'm Alice Johnson")

    # === extraction loop (client owns it + the budget) ===
    for _ in range(10):
        state = ExtractionState(**graph.invoke(state, config=config))

        if state.result is not None:  # complete once the model validates
            break

        # still incomplete: show the question and get the next reply
        if INTERACTIVE:
            print(f"\n{state.agent_message}")
            state.user_input = input("\n> ")
        else:
            state.user_input = next(scripted, "I don't know")

    # === result ===
    print(f"\n{'=' * 60}\n  RESULT\n{'=' * 60}")
    if state.result is not None:
        print(f"result model : {state.result!r}")  # validated PersonInfo
    else:
        print(f"partial fill : {state.filled} (still missing: {state.missing})")


if __name__ == "__main__":
    main()
