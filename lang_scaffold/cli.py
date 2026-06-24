"""Optional client-side terminal helpers (spinners, selectable prompts).

These are conveniences for building interactive CLI clients; the core library
never imports them, so the no-I/O boundary stays intact. Output uses ``rich``
(already a dependency); keypress input uses ``termios`` (Unix, imported lazily so
the spinner still works elsewhere). Everything degrades to plain behavior off a
TTY (piped input, logs).
"""

import contextlib
import itertools
import sys
import threading
import time

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

_console = Console()


def say(text: str) -> None:
    """Render an assistant turn as Markdown (so **bold**, lists, etc. format)."""
    _console.print()
    _console.print(Markdown(text))


def ask(prompt: str = "❯ ", color: str = "cyan") -> str:
    """Read a user turn at a styled prompt."""
    return _console.input(f"\n[bold {color}]{prompt}[/]")


def _format_elapsed(seconds: float) -> str:
    """Human-readable duration: ``<1 sec``, ``3 secs``, or ``1 min 5 secs``."""
    if seconds < 1:
        return "<1 sec"
    if seconds < 60:
        return f"{seconds:.0f} secs"
    m, sec = divmod(round(seconds), 60)
    return f"{m} min {sec} secs"


@contextlib.contextmanager
def thinking(
    *phrases: str,
    interval: float = 1.5,
    spinner: str = "dots",
    spinner_color: str = "cyan",
    timed: bool = False,
):
    """Animated status spinner that cycles through phrases on a timer.

    with thinking("thinking...", "deciding...", "making plans..."):
        result = blocking_call()

    A single phrase shows a static label; multiple phrases rotate every
    ``interval`` seconds. ``spinner_color`` colors the icon (any rich style, e.g.
    "red" or "bold magenta"); phrase strings may carry rich markup to color the
    text. ``timed=True`` prints ``(thought for X)`` once the block exits cleanly.
    Yields the rich Status so the caller can also ``.update(...)``. No-op
    animation off a TTY; for anything fancier use rich's ``console.status``.
    """
    phrases = phrases or ("working...",)
    stop = threading.Event()
    start = time.monotonic()
    elapsed = None
    with _console.status(
        phrases[0], spinner=spinner, spinner_style=spinner_color
    ) as status:

        def _rotate():
            cycle = itertools.cycle(phrases)
            next(cycle)  # phrases[0] is already showing
            while not stop.wait(interval):
                status.update(next(cycle))

        if len(phrases) > 1:
            threading.Thread(target=_rotate, daemon=True).start()
        try:
            yield status
            elapsed = time.monotonic() - start  # set only on clean exit
        finally:
            stop.set()
    if timed and elapsed is not None:
        _console.print(f"[dim](thought for {_format_elapsed(elapsed)})[/]")


@contextlib.contextmanager
def _raw_mode():
    """Put the terminal in cbreak mode (one keypress, no echo, Ctrl-C still works)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key() -> str:
    """Read one logical keypress, normalized to: up, down, enter, or other."""
    ch = sys.stdin.read(1)
    if ch == "\x03":  # Ctrl-C (cbreak keeps signals off this raw read)
        raise KeyboardInterrupt
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x1b":  # arrow keys arrive as ESC [ A/B (or ESC O A/B)
        seq = sys.stdin.read(2)
        return {"[A": "up", "OA": "up", "[B": "down", "OB": "down"}.get(seq, "other")
    return {"k": "up", "j": "down"}.get(ch, "other")


def select(
    prompt: str, options: list[str], default: int = 0, color: str = "cyan"
) -> int:
    """Arrow-key menu (↑/↓ or j/k to move, Enter to choose); returns the index.

    A highlighted, navigable list rendered with rich.Live -- the menu collapses to
    the chosen line on selection. Falls back to a numbered line-read off a TTY.
    """
    if not sys.stdin.isatty():
        for i, opt in enumerate(options):
            _console.print(f"  {i + 1}. {opt}")
        raw = sys.stdin.readline().strip()
        return int(raw) - 1 if raw.isdigit() else default

    idx = default

    def view() -> Group:
        rows = [Text(prompt, style="bold")]
        for i, opt in enumerate(options):
            selected = i == idx
            rows.append(
                Text(
                    f" {'❯' if selected else ' '} {opt}",
                    style=f"bold {color}" if selected else "dim",
                )
            )
        return Group(*rows)

    with (
        _raw_mode(),
        Live(view(), console=_console, auto_refresh=False, transient=True) as live,
    ):
        while (key := _read_key()) != "enter":
            if key == "up":
                idx = (idx - 1) % len(options)
            elif key == "down":
                idx = (idx + 1) % len(options)
            live.update(view(), refresh=True)

    _console.print(f"[{color}]❯[/] {options[idx]}")
    return idx


def confirm_or_correct(proposal: str, color: str = "cyan") -> tuple[bool, str]:
    """Accept/reject a proposal via a selectable menu; returns ``(accepted, reason)``.

    Reject prompts for a reason on a separate step, so the decision and the
    correction are never conflated and a correction can't be misread as accept.
    """
    _console.print(proposal)
    if (
        select("Use this?", ["Yes, looks good", "No, let me correct it"], color=color)
        == 0
    ):
        return True, ""
    return False, _console.input("[dim]what should change?[/] ").strip()
