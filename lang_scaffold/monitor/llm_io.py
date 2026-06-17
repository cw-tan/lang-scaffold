import json
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from rich.align import Align
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule

_console = Console()
_ROLE = {  # role -> (display label, color)
    "system": ("System", "dim"),
    "human": ("Human", "cyan"),
    "ai": ("AI", "green"),
}


def _bubble(m: dict):
    """Render an input message: Human bubble right, AI left, system full-width."""
    label, color = _ROLE.get(m["role"], (m["role"].title(), "white"))
    if m["role"] == "system":
        return Panel(
            escape(m["content"]), title=label, title_align="left", border_style=color
        )
    align = "right" if m["role"] == "human" else "left"
    panel = Panel(
        escape(m["content"]),
        title=label,
        title_align=align,
        border_style=color,
        expand=False,
    )
    return Align.right(panel) if m["role"] == "human" else Align.left(panel)


def _render(record: dict, n: int) -> None:
    """Pretty-print one record as a numbered rich panel: chat input, then output."""
    bubbles = [_bubble(m) for m in record["input"]]
    if "error" in record:
        out = f"[bold red]Error[/]\n{escape(record['error'])}"
    else:
        parts = []
        for o in record["output"]:
            if o["text"]:
                parts.append(f"[bold green]Reply[/]\n{escape(o['text'])}")
            for tc in o["tool_calls"]:
                parts.append(
                    f"[bold yellow]Tool {tc['name']}[/]\n{escape(str(tc['args']))}"
                )
            if not o["text"] and not o["tool_calls"]:
                parts.append("[dim](empty)[/]")
        out = "\n\n".join(parts)
    body = Group(*bubbles, Rule(style="dim"), out)
    _console.print(Panel(body, title=f"[bold]LLM Call #{n}[/]", title_align="center"))


class PromptLogger(BaseCallbackHandler):
    """Logs raw LLM input/output, one record per call (the LLM-I/O layer).

    Attach per run via ``config={"callbacks": [PromptLogger()]}`` on ``invoke``, or bind to the model.

    Args:
        path: If None, pretty-print each call to stdout. If given, append one
            JSON record per call to this JSONL file; read it back with
            :meth:`load_and_print`.
    """

    def __init__(self, path: Optional[str] = None):
        self._file = open(path, "a") if path else None
        self._pending: dict = {}  # run_id -> input messages, awaiting the output
        self._n = 0  # call counter for stdout numbering

    def on_chat_model_start(
        self, serialized: dict, messages: list, **kwargs: Any
    ) -> None:
        self._pending[kwargs["run_id"]] = [
            {"role": m.type, "content": m.content} for batch in messages for m in batch
        ]

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        output = [
            {"text": g.text, "tool_calls": g.message.tool_calls}
            for gens in response.generations
            for g in gens
        ]
        self._emit({"input": self._pending.pop(kwargs["run_id"]), "output": output})

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._emit(
            {
                "input": self._pending.pop(kwargs["run_id"]),
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def _emit(self, record: dict) -> None:
        if self._file:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()  # flush so a tail/parser sees records live
        else:
            self._n += 1
            _render(record, self._n)

    @staticmethod
    def load_and_print(path: str) -> None:
        """Load a JSONL log file and pretty-print each call."""
        with open(path) as f:
            for n, line in enumerate(f, 1):
                _render(json.loads(line), n)
