#!/usr/bin/env python3
"""Interactive read-only filesystem agent.

Binds lang_scaffold's explore tools to a model and lets it answer questions about
a directory by exploring on its own -- a minimal ReAct loop you can watch step by
step. Each tool call the model makes is printed (dimmed) so you can see HOW it
arrives at an answer, not just the answer.

Run from anywhere; defaults to exploring the repo root (pass a path to override):
    python explore_agent.py [root]

Slightly-challenging prompts to try -- each needs more than one tool:
  - "Which .py file here is largest, and what does its module docstring say?"
  - "Where is the `_cap` helper used, and what is it guarding against?"
  - "Does grep shell out to ripgrep, or search in pure Python? Prove it from the code."
  - "What env vars does the extraction example require, and are they set right now?"
"""

import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lang_scaffold.cli import ask, say, thinking
from lang_scaffold.tools.explore import EXPLORE_TOOLS

SYSTEM = (
    "You are a filesystem investigator with read-only tools: list_dir, read_file, "
    "glob, grep, path_info, get_env, which, tree. Investigate with the tools before "
    "answering -- never guess at file contents. Chain tools as needed, and cite the "
    "exact paths and line numbers that back up your answer."
)

MAX_STEPS = 12  # cap tool-calling rounds so a confused model can't loop forever
_DIM, _RESET = "\033[2m", "\033[0m"


def run_agent(llm, tools_by_name: dict, question: str) -> str:
    """Drive one ReAct loop: model calls tools until it returns a final answer."""
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
    for _ in range(MAX_STEPS):
        with thinking("thinking...", "exploring...", spinner_color="cyan", timed=True):
            ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content
        for tc in ai.tool_calls:
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            preview = result if len(result) <= 300 else result[:300] + " ..."
            print(f"{_DIM}  [{tc['name']}({tc['args']}) ->\n    {preview}]{_RESET}")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return "(gave up: hit the step cap without a final answer)"


def main():
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    os.chdir(root)  # tools resolve relative paths against cwd

    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,  # unset/empty -> provider default
        api_key=os.environ["LLM_API_KEY"],
    ).bind_tools(EXPLORE_TOOLS)
    tools_by_name = {t.name: t for t in EXPLORE_TOOLS}

    print(__doc__)
    print(f"Exploring: {os.getcwd()}  (empty line to quit)\n")
    while True:
        try:
            question = ask("ask> ", color="cyan").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print()
        say(run_agent(llm, tools_by_name, question))


if __name__ == "__main__":
    main()
