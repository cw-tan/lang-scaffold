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

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

_console = Console()

_DEFAULT_GRADIENT = ("bright_cyan", "cyan", "bright_blue", "blue", "bright_magenta")


def _gradient(lines: list[str], colors: tuple[str, ...]) -> Text:
    """Color lines top-to-bottom across ``colors`` for a vertical gradient."""
    text = Text(justify="center")
    last = max(len(lines) - 1, 1)
    for i, line in enumerate(lines):
        text.append(
            line + "\n", style=f"bold {colors[round(i / last * (len(colors) - 1))]}"
        )
    return text


def banner(
    text: str, subtitle: str = "", *, colors: tuple[str, ...] = _DEFAULT_GRADIENT
) -> None:
    """Welcome panel: render ``text`` centered with a vertical color gradient in a
    rounded box. ``text`` may be a plain title or pre-made multi-line ASCII art;
    ``colors`` is the top-to-bottom gradient ramp (a single-color tuple = flat).
    """
    body = _gradient(text.strip("\n").splitlines(), colors)
    if subtitle:
        body.append("\n" + subtitle, style="dim")
    _console.print(
        Panel(
            Align.center(body),
            box=box.ROUNDED,
            border_style=colors[0],
            padding=(1, 4),
            expand=False,
        )
    )


def say(text: str) -> None:
    """Render an assistant turn as Markdown (so **bold**, lists, etc. format)."""
    _console.print()
    _console.print(Markdown(text))


def ask(prompt: str = "❯ ") -> str:
    """Read a user turn at a styled prompt."""
    return _console.input(f"\n[bold cyan]{prompt}[/]")


@contextlib.contextmanager
def thinking(
    *phrases: str,
    interval: float = 1.5,
    spinner: str = "dots",
    spinner_color: str = "cyan",
):
    """Animated status spinner that cycles through phrases on a timer.

    with thinking("thinking...", "deciding...", "making plans..."):
        result = blocking_call()

    A single phrase shows a static label; multiple phrases rotate every
    ``interval`` seconds. ``spinner_color`` colors the icon (any rich style, e.g.
    "red" or "bold magenta"); phrase strings may carry rich markup to color the
    text. Yields the rich Status so the caller can also ``.update(...)``. No-op
    animation off a TTY; for anything fancier use rich's ``console.status``.
    """
    phrases = phrases or ("working...",)
    stop = threading.Event()
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
        finally:
            stop.set()


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


def select(prompt: str, options: list[str], *, default: int = 0) -> int:
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
                    style="bold cyan" if selected else "dim",
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

    _console.print(f"[cyan]❯[/] {options[idx]}")
    return idx


def confirm_or_correct(proposal: str) -> tuple[bool, str]:
    """Accept/reject a proposal via a selectable menu; returns ``(accepted, reason)``.

    Reject prompts for a reason on a separate step, so the decision and the
    correction are never conflated and a correction can't be misread as accept.
    """
    _console.print(proposal)
    if select("Use this?", ["Yes, looks good", "No, let me correct it"]) == 0:
        return True, ""
    return False, _console.input("[dim]what should change?[/] ").strip()
