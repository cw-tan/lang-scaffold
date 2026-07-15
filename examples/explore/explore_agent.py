import argparse
import os
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from lang_scaffold.cli import ThinkingSpinner, ask, note, say
from lang_scaffold.monitor import ToolMonitor
from lang_scaffold.tools.explore import build_explore_tools
from lang_scaffold.tools.observability import with_rationale

SYSTEM = (
    "You are a filesystem investigator with read-only tools. Investigate before "
    "answering -- never guess -- and cite the paths and lines that back your answer."
)

# one sqlite file per conversation, so the filesystem is the conversation list;
# thread_id is then a constant -- the file already identifies the conversation
THREAD = "main"
CONV_DIR = Path(__file__).parent / "conversations"


def parse_args():
    p = argparse.ArgumentParser(description="Filesystem explore agent.")
    p.add_argument("--resume", metavar="DB", help="continue a prior conversation")
    return p.parse_args()


def new_conversation_path() -> Path:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return CONV_DIR / f"explore-{stamp}.db"


def main():
    args = parse_args()
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ["LLM_API_KEY"],
    )
    path = Path(args.resume) if args.resume else new_conversation_path()
    tools = [with_rationale(t) for t in build_explore_tools(".")]  # confined to cwd
    # the sqlite connection must stay open for the whole session, so wrap the loop
    with SqliteSaver.from_conn_string(str(path)) as cp:
        agent = create_agent(
            llm,
            tools,
            system_prompt=SYSTEM,
            middleware=[ModelCallLimitMiddleware(run_limit=12)],
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
                question = ask("ask> ", color="cyan").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            # send only the new turn; prior state loads from the file
            result = agent.invoke({"messages": [HumanMessage(question)]}, config)
            say(result["messages"][-1].content)


if __name__ == "__main__":
    main()
