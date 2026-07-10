import os

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from lang_scaffold import ExtractionState, build_extraction_loop
from lang_scaffold.cli import ThinkingSpinner, ask, confirm_or_correct, note, say
from lang_scaffold.monitor import ToolMonitor
from lang_scaffold.tools.explore import build_explore_tools
from lang_scaffold.tools.lookup import build_lookup_tool
from lang_scaffold.tools.observability import describe, with_rationale

SYSTEM = "You are a helpful assistant with tools. Use them when relevant and investigate before answering rather than guessing."


class Pet(BaseModel):
    name: str = Field(..., description="Pet's name")
    species: str = Field(..., description="Species, e.g. dog or cat")


class PersonInfo(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    pets: list[Pet] = Field(
        ..., description="The person's pets (name and species each)"
    )


# a small "pet records" corpus the lookup tool retrieves from, keyed by person
PET_RECORDS = {
    "john": "Two dogs named Bing and Bong, and a cat named Russel.",
    "oliver": "A parrot named Kiwi.",
    "stacy": "Three cats: Mittens, Shadow, and Luna.",
}


def build_collect_tool(llm, tools):
    # the extraction loop, packaged as one client-side tool: it owns its own
    # multi-turn dialogue (private transcript) and hands back only the validated
    # result. it gets the given tools too, so it can look things up while
    # collecting (those tool calls surface through the agent's ToolMonitor).
    graph = build_extraction_loop(
        llm=llm,
        model=PersonInfo,
        context_prompt="You are collecting a person's contact details and pets.",
        tools=tools,
    )

    @tool
    def collect_personal_info() -> str:
        """Collect the user's personal details and pets. Use when the user wants to
        provide or record their profile (name, email, pets)."""
        state = ExtractionState(user_input=ask("tell me about yourself: "))
        for _ in range(10):
            state = ExtractionState(**graph.invoke(state))
            if state.result is not None:
                return state.result.model_dump_json()
            if state.proposed is not None:
                accepted, reason = confirm_or_correct(state.agent_message)
                if accepted:
                    state.confirmed = True
                else:
                    state.user_input = reason
            else:
                say(state.agent_message)
                state.user_input = ask()
        return "the user did not complete their profile"

    return collect_personal_info


def main():
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ["LLM_API_KEY"],
    )
    explore = [with_rationale(t) for t in build_explore_tools(".")]  # roam cwd only
    # retrieval tool over the pet-records dict; describe() so it shows in ToolMonitor
    records = describe(lambda a: f"look up {a['key']}'s pets")(
        build_lookup_tool(
            PET_RECORDS, "pet_records", "Look up a person's pets by name."
        )
    )
    # the extraction loop fills `pets` by looking the person up in pet_records
    tools = explore + [records, build_collect_tool(llm, [records])]
    agent = create_agent(
        llm,
        tools,
        system_prompt=SYSTEM,
        middleware=[ModelCallLimitMiddleware(run_limit=12)],
    )
    # ThinkingSpinner spins around every model call (agent loop + nested extraction);
    # render=note so tool lines print above that spinner, not over it
    config = {"callbacks": [ToolMonitor(tools, render=note), ThinkingSpinner()]}

    messages = []  # transcript persists across questions
    while True:
        try:
            question = ask("ask> ", color="cyan").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        messages.append(HumanMessage(content=question))
        messages = agent.invoke({"messages": messages}, config=config)["messages"]
        say(messages[-1].content)


if __name__ == "__main__":
    main()
