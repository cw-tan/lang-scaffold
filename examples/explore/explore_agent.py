import os

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from lang_scaffold.cli import ThinkingSpinner, ask, note, say
from lang_scaffold.monitor import ToolMonitor
from lang_scaffold.tools.explore import EXPLORE_TOOLS
from lang_scaffold.tools.observability import with_rationale

SYSTEM = (
    "You are a filesystem investigator with read-only tools. Investigate before "
    "answering -- never guess -- and cite the paths and lines that back your answer."
)


def main():
    tools = [with_rationale(t) for t in EXPLORE_TOOLS]
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ["LLM_API_KEY"],
    )
    agent = create_agent(
        llm,
        tools,
        system_prompt=SYSTEM,
        middleware=[ModelCallLimitMiddleware(run_limit=12)],
    )
    config = {"callbacks": [ToolMonitor(tools, render=note), ThinkingSpinner()]}

    messages = []  # transcript persists across questions
    while True:
        try:
            question = ask("ask> ", color="cyan").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        messages.append(HumanMessage(question))
        messages = agent.invoke({"messages": messages}, config=config)["messages"]
        say(messages[-1].content)


if __name__ == "__main__":
    main()
