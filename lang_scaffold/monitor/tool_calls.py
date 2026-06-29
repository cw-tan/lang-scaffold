from langchain_core.callbacks import BaseCallbackHandler

from lang_scaffold.tools.observability import describe_call


def _print_dim(line: str) -> None:
    print(f"\033[2m{line}\033[0m")  # ANSI dim


class ToolMonitor(BaseCallbackHandler):
    """Surface each tool call as a user-readable line (rendered by ``describe_call``).

    Attach via ``config={"callbacks": [ToolMonitor(tools)]}`` to show tool use
    wherever it runs -- a manual loop or inside a graph/extractor -- with no library
    hooks. Calls a tool opts out of (``describe_call`` -> None) are skipped. ``render``
    receives each line to emit (default: dim to stdout).
    """

    def __init__(self, tools: list, render=_print_dim):
        self._by_name = {t.name: t for t in tools}
        self._render = render

    def on_tool_start(self, serialized: dict, input_str: str, *, inputs=None, **kwargs):
        tool = self._by_name.get((serialized or {}).get("name"))
        if tool is None:
            return
        args = inputs or {}
        desc = describe_call(tool, args)
        if not desc:
            return
        reason = args.get("reason")  # present when the tool is with_rationale-wrapped
        self._render(f"  {desc}. Purpose: {reason}" if reason else f"  {desc}")
