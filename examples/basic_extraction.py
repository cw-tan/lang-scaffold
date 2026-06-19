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

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, field_validator, model_validator

from lang_scaffold import ExtractionState, build_extraction_loop
from lang_scaffold.monitor import PromptLogger

INTERACTIVE = True


class ContactInfo(BaseModel):
    """Nested contact details."""

    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number")

    @field_validator("email")
    @classmethod
    def _dotcom(cls, v: str) -> str:
        # field-level validation
        assert v.endswith(".com"), "email must end with .com"
        return v


class PersonInfo(BaseModel):
    """Fields we want to collect from the user."""

    name: str = Field(..., description="Full name")
    contact: ContactInfo = Field(..., description="Contact information")
    age: Optional[int] = Field(default=None, description="Age in years")
    country: str = Field(default="USA", description="Country of residence")

    @model_validator(mode="after")
    def _phone_matches_country(self):
        # cross-field (post) validation reaching into the nested model
        expected = 10 if self.country == "USA" else 8
        digits = sum(c.isdigit() for c in self.contact.phone)
        assert digits == expected, f"a {self.country} phone number needs {expected} digits"
        return self


def main():
    # === set up LLM (OpenAI-compatible endpoint via LLM_BASE_URL, no defaults) ===
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider="openai",
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )

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
        print(f"partial fill : {state.filled}")


if __name__ == "__main__":
    main()
