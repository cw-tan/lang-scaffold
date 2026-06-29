"""Unit tests for tool-call observability helpers -- pure, no LLM."""

import pytest
from langchain_core.tools import tool

from lang_scaffold.tools.observability import describe, describe_call


@pytest.fixture
def greet():
    # a fresh, generic tool per test (describe() mutates the tool's metadata)
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hi {name}"

    return greet


def test_string_template_renders(greet):
    described = describe("greeting {name}")(greet)
    assert describe_call(described, {"name": "Ada"}) == "greeting Ada"


def test_callable_spec_renders(greet):
    described = describe(lambda a: f"greeting {a['name'].upper()}")(greet)
    assert describe_call(described, {"name": "Ada"}) == "greeting ADA"


def test_callable_returns_none_hides_call(greet):
    described = describe(lambda a: None)(greet)
    assert describe_call(described, {"name": "Ada"}) is None


def test_undecorated_is_hidden(greet):
    assert describe_call(greet, {"name": "Ada"}) is None


def test_missing_key_degrades_to_empty(greet):
    described = describe("greeting {name}")(greet)
    assert describe_call(described, {}) == "greeting "
