import argparse
import os
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.text import Text

from lang_scaffold.cli import (
    ThinkingSpinner,
    ask,
    confirm_or_correct,
    diff_text,
    note,
    say,
)
from lang_scaffold.monitor import ToolMonitor
from lang_scaffold.tools.edit import build_edit_tools
from lang_scaffold.tools.explore import ReadTracker, build_explore_tools, confine
from lang_scaffold.tools.observability import describe_call, with_rationale

# how to drive the tools is the tools' own business -- their contracts are enforced by
# the read tracker and stated in their descriptions, not restated here
SYSTEM = (
    "You are a coding agent working in the current directory. Make the smallest change "
    "that does the job, in the idiom of the surrounding code. Cite the paths and lines "
    "you relied on, and say plainly what you changed and what you did not verify."
)

# tools that touch the filesystem pause for approval; reads run freely
GATED = ("write_file", "edit_file")

BASE = "."  # everything the agent can reach, and what previews resolve against
_resolve, _ = confine(BASE)

# one sqlite file per conversation, so the filesystem is the conversation list;
# thread_id is then a constant -- the file already identifies the conversation
THREAD = "main"
CONV_DIR = Path(__file__).parent / "conversations"


def parse_args():
    p = argparse.ArgumentParser(description="Coding agent.")
    p.add_argument("--resume", metavar="DB", help="continue a prior conversation")
    return p.parse_args()


def new_conversation_path() -> Path:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return CONV_DIR / f"coding-{stamp}.db"


def build_tools() -> list:
    """Explore + edit tools over one shared tracker, so writes see what reads recorded."""
    tracker = ReadTracker()
    tools = build_explore_tools(BASE, tracker) + build_edit_tools(BASE, tracker)
    return [with_rationale(t) for t in tools]


def approval_prompt(tools: list):
    """Describe a gated call for review, as ToolMonitor renders any other call.

    Stays a plain string: the description travels in graph state and is checkpointed,
    so styling belongs to the terminal client (see ``_proposal``), not the payload.
    """
    by_name = {t.name: t for t in tools}

    def describe_request(tool_call, state, runtime) -> str:
        args = tool_call["args"]
        head = describe_call(by_name[tool_call["name"]], args) or tool_call["name"]
        return f"{head}. Purpose: {args['reason']}" if args.get("reason") else head

    return describe_request


def _current(path: str) -> str:
    """Contents a write would replace, so the diff shows what is lost, not just added."""
    p, err = _resolve(path)
    if err or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def _proposal(req: dict) -> Text:
    """The request's one-liner over a colored diff of what it would write.

    Rebuilt from the request's own args -- approving a bare path is meaningless.
    """
    args = req["args"]
    if req["name"] == "edit_file":
        before, after = args.get("old_string", ""), args.get("new_string", "")
    else:
        before, after = _current(args.get("path", "")), args.get("content", "")
    # indented to sit in ToolMonitor's column, leaving the -/+ signs in a gutter
    head = Text(f"  {req.get('description') or req['name']}\n")
    return head + diff_text(before, after)


def _rejection(req: dict, reason: str) -> str:
    """State that nothing happened, then relay why.

    A bare reason replaces the middleware's default text, leaving the model an
    ``error`` result with no indication the call was vetoed rather than failed -- it
    then retries the edit and reports success for a file it never touched.
    """
    target = req["args"].get("path", "the file")
    said = f' They said: "{reason}".' if reason else ""  # quoted: their words, not ours
    return (
        f"The user rejected this {req['name']} call. It was NOT executed and {target} "
        f"is unchanged on disk.{said} Do not repeat this call unless they ask again, "
        "and do not tell them the change was made."
    )


def collect_decisions(requests: list) -> list:
    """Accept or reject each pending action, the same way an extraction is confirmed."""
    decisions = []
    for req in requests:
        accepted, reason = confirm_or_correct(_proposal(req))
        if accepted:
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": _rejection(req, reason)})
    return decisions


def run_turn(agent, payload, config):
    """Drive one turn to completion, pausing at each approval interrupt."""
    result = agent.invoke(payload, config)
    while interrupts := result.get("__interrupt__"):
        requests = interrupts[0].value["action_requests"]
        decisions = collect_decisions(requests)
        result = agent.invoke(Command(resume={"decisions": decisions}), config)
    return result


def main():
    args = parse_args()
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ["LLM_API_KEY"],
    )
    path = Path(args.resume) if args.resume else new_conversation_path()
    tools = build_tools()  # confined to cwd
    approval = {
        name: {
            "allowed_decisions": ["approve", "reject"],
            "description": approval_prompt(tools),
        }
        for name in GATED
    }
    # the sqlite connection must stay open for the whole session, so wrap the loop
    with SqliteSaver.from_conn_string(str(path)) as cp:
        agent = create_agent(
            llm,
            tools,
            system_prompt=SYSTEM,
            middleware=[
                ModelCallLimitMiddleware(run_limit=40),  # edit/verify loops are chatty
                HumanInTheLoopMiddleware(approval),
            ],
            checkpointer=cp,
        )
        config = {
            "configurable": {"thread_id": THREAD},
            "callbacks": [ToolMonitor(tools, render=note), ThinkingSpinner()],
        }
        prior = agent.get_state(config).values.get("messages", [])
        note(f"{path}" + (f"  (resumed, {len(prior)} messages)" if prior else ""))

        while True:
            try:
                task = ask("task> ", color="cyan").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task:
                break
            # send only the new turn; prior state loads from the file
            result = run_turn(agent, {"messages": [HumanMessage(task)]}, config)
            say(result["messages"][-1].content)


if __name__ == "__main__":
    main()
