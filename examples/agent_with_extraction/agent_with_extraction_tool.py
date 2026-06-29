import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from lang_scaffold import ExtractionState, build_extraction_loop
from lang_scaffold.cli import ask, confirm_or_correct, note, say, thinking
from lang_scaffold.monitor import ToolMonitor
from lang_scaffold.tools.explore import EXPLORE_TOOLS
from lang_scaffold.tools.observability import with_rationale

SYSTEM = (
    "You are a helpful assistant with tools. Use them when relevant and investigate before answering rather than guessing."
)

MAX_STEPS = 12


class Pet(BaseModel):
    name: str = Field(..., description="Pet's name")
    species: str = Field(..., description="Species, e.g. dog or cat")


class PersonInfo(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    pets: list[Pet] = Field(
        ..., description="The person's pets (name and species each)"
    )


def build_collect_tool(llm, tools):
    # the extraction loop, packaged as one client-side tool: it owns its own
    # multi-turn dialogue (private transcript) and hands back only the validated
    # result. it gets the explore tools too, so it can find info on disk while
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
            with thinking("extracting...", "checking...", timed=True):
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


def run_agent(llm, tools_by_name, messages, config):
    # generic ReAct loop -- collect_personal_info is just another tool to it
    for _ in range(MAX_STEPS):
        with thinking("thinking...", spinner_color="cyan", timed=True):
            ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content
        for tc in ai.tool_calls:
            result = tools_by_name[tc["name"]].invoke(tc["args"], config=config)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return "(gave up: hit the step cap without a final answer)"


def main():
    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ["LLM_API_KEY"],
    )
    # one set of explore tools, shared by the agent and the extraction tool's gather
    explore = [with_rationale(t) for t in EXPLORE_TOOLS]
    tools = explore + [build_collect_tool(llm, explore)]
    agent = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    # render=note so tool lines print above the extraction tool's spinner, not over it
    config = {"callbacks": [ToolMonitor(tools, render=note)]}

    messages = [SystemMessage(content=SYSTEM)]  # transcript persists across questions
    while True:
        try:
            question = ask("ask> ", color="cyan").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        messages.append(HumanMessage(content=question))
        print()
        say(run_agent(agent, tools_by_name, messages, config))


if __name__ == "__main__":
    main()
