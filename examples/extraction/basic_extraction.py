#!/usr/bin/env python3
"""Minimal interactive example showing the client/library split.

The library gives a one-turn graph. The CLIENT owns the loop below: it invokes
the graph, does the I/O when the graph asks a question or proposes a result,
threads the reply back in, and owns the turn budget. Everything inside
`graph.invoke` is the library.

A complete extraction comes back as a PROPOSAL (`state.proposed`) for the user to
confirm before it is finalized -- so a valid-but-unintended value can be caught.
Chat in the terminal; LLM I/O is logged to llm.jsonl (inspect with show_log.py).
"""

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, field_validator, model_validator

from lang_scaffold import ExtractionState, build_extraction_loop
from lang_scaffold.cli import ask, confirm_or_correct, say, thinking
from lang_scaffold.monitor import PromptLogger


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
    # === set up LLM (provider/endpoint/model all from env, no defaults) ===
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,  # unset/empty -> provider default
        api_key=os.environ["LLM_API_KEY"],
    )

    # === build graph ===
    graph = build_extraction_loop(
        llm=llm,
        model=PersonInfo,
        context_prompt="You are collecting a person's contact details.",
    )

    # === seed the first turn (LLM I/O logged to llm.jsonl, inspect with show_log.py) ===
    config = {"callbacks": [PromptLogger("llm.jsonl")]}
    say("I'll collect your contact details. Tell me about yourself.")
    state = ExtractionState(user_input=ask())

    # === extraction loop (client owns it + the budget) ===
    for _ in range(10):
        with thinking("thinking...", "extracting...", "checking..."):
            result = graph.invoke(state, config=config)
        state = ExtractionState(**result)

        if state.result is not None:  # confirmed -> done
            break

        if state.proposed is not None:
            # complete proposal -> explicit accept/reject (never inferred from free text)
            accepted, reason = confirm_or_correct(state.agent_message)
            if accepted:
                state.confirmed = True  # next loop finalizes it
            else:
                state.user_input = reason  # correction -> re-extract
        else:
            # still collecting -> answer the question
            say(state.agent_message)
            state.user_input = ask()

    # === result ===
    print(f"\n{'=' * 60}\n  RESULT\n{'=' * 60}")
    if state.result is not None:
        print(f"result model : {state.result!r}")  # validated PersonInfo
    else:
        print(f"partial fill : {state.filled}")


if __name__ == "__main__":
    main()
