from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler


class PromptLogger(BaseCallbackHandler):
    """Logs raw LLM input prompts and output completions (the LLM-I/O layer).

    Attach per run via ``config={"callbacks": [PromptLogger()]}`` on ``invoke``,
    or bind to the model at construction. LangGraph propagates it down to the
    model calls inside the nodes, so it observes every call -- including internal
    ones (e.g. structured-output extraction) that produce no user-facing message.
    """

    def __init__(self, log_fn: Callable[[str], None] = print):
        self.log = log_fn

    def _section(self, title: str) -> None:
        self.log(f"\n{'-' * 60}\n  {title}\n{'-' * 60}")

    def on_chat_model_start(
        self, serialized: dict, messages: list, **kwargs: Any
    ) -> None:
        # messages is a list of message-lists, one per generation request
        for batch in messages:
            self._section("LLM INPUT")
            for m in batch:
                self.log(f"[{m.type}] {m.content}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # structured-output calls answer with tool calls (args carry the fields)
        # and leave the text empty, so surface both
        for gens in response.generations:
            for g in gens:
                self._section("LLM OUTPUT")
                tool_calls = getattr(getattr(g, "message", None), "tool_calls", None)
                if g.text:
                    self.log(g.text)
                for tc in tool_calls or []:
                    self.log(f"[tool_call] {tc.get('name')}: {tc.get('args')}")
                if not g.text and not tool_calls:
                    self.log("(empty)")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.log(f"\n[LLM ERROR] {type(error).__name__}: {error}")
