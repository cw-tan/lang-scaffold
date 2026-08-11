import difflib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from lang_scaffold.tools.explore import ReadTracker, cap, confine, note_base
from lang_scaffold.tools.observability import describe

_MAX_DIFF_LINES = 200
_MAX_OUTPUT_LINES = 300
_TIMEOUT = 60  # seconds, default for run_command


def _decode(stream) -> str:
    """Subprocess output as text, whichever form the exception carried it in."""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream or ""


def _read_text(p: Path) -> tuple[Optional[str], Optional[str]]:
    """(text, None), or (None, error) for anything that isn't UTF-8 text."""
    try:
        return p.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, f"error: {p} is not a UTF-8 text file (binary?)"
    except OSError as e:
        return None, f"error: cannot read {p}: {e}"


def _atomic_write(p: Path, text: str) -> None:
    """Write via a temp file in the same dir, then rename -- an interrupted write
    leaves the original intact rather than a truncated file. Mode is preserved.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = p.stat().st_mode if p.exists() else None
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, p)  # atomic within a filesystem
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _diff(before: str, after: str) -> str:
    """Compact unified diff of a single file's contents."""
    body = list(
        difflib.unified_diff(before.splitlines(), after.splitlines(), n=1, lineterm="")
    )
    return cap(body[2:], _MAX_DIFF_LINES)  # [2:] drops the ---/+++ header


def _match_lines(text: str, sub: str) -> list[int]:
    """1-based line numbers where each occurrence of ``sub`` starts."""
    out, i = [], text.find(sub)
    while i != -1:
        out.append(text.count("\n", 0, i) + 1)
        i = text.find(sub, i + 1)
    return out


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _near_miss(text: str, old: str) -> str:
    """Diagnose a failed exact match -- nearly always whitespace, so say so."""
    target = _squash(old)
    if not target:
        return ""
    hits = [i + 1 for i, ln in enumerate(text.splitlines()) if target in _squash(ln)]
    if hits or target in _squash(text):
        where = f" at line(s) {', '.join(map(str, hits))}" if hits else ""
        return (
            f" The text is present{where} but with different whitespace -- copy it "
            "verbatim from read_file output (minus the line-number prefix), keeping "
            "the original indentation."
        )
    return " Re-read the file; its contents differ from what you assumed."


def _newline(before: Optional[str], text: str) -> str:
    """Honor the file's trailing-newline convention (new files get one)."""
    if not text or text.endswith("\n"):
        return text
    return text + "\n" if before is None or before.endswith("\n") else text


def build_edit_tools(
    base: Optional[str] = None, tracker: Optional[ReadTracker] = None
) -> list:
    """Build the file-mutating tools (``write_file``, ``edit_file``).

    ``base`` confines every path exactly as in ``build_explore_tools``. Sharing that
    builder's ``tracker`` is what enforces read-before-write; without one the tools
    still work, but nothing stops the model from overwriting a file it never read.
    """
    _resolve, _base = confine(base)

    def _guard(p: Path, whole: bool) -> Optional[str]:
        return tracker.check(p, whole=whole) if tracker is not None else None

    @describe("write {path}")
    @tool(parse_docstring=True)
    def write_file(path: str, content: str) -> str:
        """Create a file, or replace an existing file's contents entirely.

        A file that already exists must have been read in full first. Prefer
        ``edit_file`` for changing part of a file.

        Args:
            path: File to write.
            content: Full contents of the file.
        """
        p, err = _resolve(path)
        if err:
            return err
        if p.is_dir():
            return f"error: {path} is a directory"
        before = None
        if p.is_file():
            if err := _guard(p, whole=True):
                return err
            before, err = _read_text(p)
            if err:
                return err
        text = _newline(before, content)
        try:
            _atomic_write(p, text)
        except OSError as e:
            return f"error: cannot write {path}: {e}"
        if tracker is not None:
            tracker.record(p)  # the agent authored it, so it has seen it
        n = len(text.splitlines())
        if before is None:
            return f"created {path} ({n} lines)"
        if before == text:
            return f"{path} unchanged"
        return f"wrote {path} ({n} lines)\n{_diff(before, text)}"

    @describe("edit {path}")
    @tool(parse_docstring=True)
    def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace an exact string in a file; fails unless it occurs exactly once.

        Copy ``old_string`` verbatim from ``read_file`` output (minus the line-number
        prefix) and widen it with surrounding lines until it is unique in the file.

        Args:
            path: File to edit.
            old_string: Exact text to replace, including indentation.
            new_string: Text to put in its place.
            replace_all: Replace every occurrence instead of requiring exactly one.
        """
        p, err = _resolve(path)
        if err:
            return err
        if not p.is_file():
            return f"error: not a file: {path}"
        if not old_string:
            return "error: old_string is empty -- use write_file to create a file"
        if old_string == new_string:
            return "error: old_string and new_string are identical"
        if err := _guard(p, whole=False):
            return err
        before, err = _read_text(p)
        if err:
            return err
        count = before.count(old_string)
        if count == 0:
            return f"error: old_string not found in {path}.{_near_miss(before, old_string)}"
        if count > 1 and not replace_all:
            lines = ", ".join(map(str, _match_lines(before, old_string)))
            return (
                f"error: old_string occurs {count} times in {path} (lines {lines}) -- "
                "add surrounding lines to target one of them, or pass replace_all=True"
            )
        after = before.replace(old_string, new_string)
        try:
            _atomic_write(p, after)
        except OSError as e:
            return f"error: cannot write {path}: {e}"
        if tracker is not None:
            tracker.record(p)
        occurrences = "1 occurrence" if count == 1 else f"all {count} occurrences"
        return f"edited {path} ({occurrences})\n{_diff(before, after)}"

    return note_base([write_file, edit_file], _base)


def build_shell_tools(base: Optional[str] = None) -> list:
    """Build the command runner (``run_command``).

    Kept apart from the edit tools because a shell voids path confinement: ``base``
    only sets the working directory, it cannot keep a command inside it.
    """
    _base = Path(base).resolve() if base else None

    @describe("run: {command}")
    @tool(parse_docstring=True)
    def run_command(command: str, timeout: int = _TIMEOUT) -> str:
        """Run a shell command and return its exit code with its combined output.

        Args:
            command: Command line to run, e.g. ``pytest -q``.
            timeout: Seconds to wait before killing the command.
        """
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=_base,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            partial = _decode(e.stdout) + _decode(e.stderr)
            tail = (
                f"\n{cap(partial.splitlines(), _MAX_OUTPUT_LINES)}" if partial else ""
            )
            return f"error: killed after {timeout}s timeout{tail}"
        except OSError as e:
            return f"error: cannot run command: {e}"
        output = (proc.stdout + proc.stderr).rstrip()
        body = cap(output.splitlines(), _MAX_OUTPUT_LINES) if output else "(no output)"
        return f"exit {proc.returncode}\n{body}"

    if _base is not None:
        run_command.description += f"  Runs in {_base}, which it is free to leave."
    return [run_command]
