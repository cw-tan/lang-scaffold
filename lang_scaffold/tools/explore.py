"""Read-only filesystem and environment exploration tools.

Plain LangChain tools the client binds to any model -- the agent can then explore the filesystem on its own.
Every tool is read-only (no write, edit, delete, or shell exec), so they are all safe to auto-allow.

NOTE: these tools are NOT sandboxed -- they can read any path the host process can, including secrets exposed via ``get_env``.
Root confinement (refusing to escape a base directory via ``..``/symlinks) is intentionally left out;
wrap the tools in the client if you need it.
"""

import os
import re
import shutil
import stat
import time
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from lang_scaffold.tools.observability import describe

# output caps so a single call can never flood the agent's context
_MAX_LINES = 1000
_MAX_MATCHES = 200
_TIMEOUT = 5.0  # seconds; tree-walking tools bail with a partial result past this

# directories pruned from recursive walks (grep/tree) -- noise, rarely useful
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}


def _cap(lines: list[str]) -> str:
    """Join lines, truncating past the cap with a marker."""
    if len(lines) > _MAX_LINES:
        extra = len(lines) - _MAX_LINES
        return "\n".join(lines[:_MAX_LINES]) + f"\n... [{extra} more lines truncated]"
    return "\n".join(lines)


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024


def _walk_files(base: Path, pattern: Optional[str], deadline: Optional[float] = None):
    """Yield files under base, pruning heavy dirs and filtering by glob pattern.
    Stops walking once ``deadline`` (a ``time.monotonic`` value) passes.
    """
    for root, dirs, files in os.walk(base):
        if deadline and time.monotonic() > deadline:
            return  # caller detects the timeout via its own deadline check
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if pattern and not fnmatch(name, pattern):
                continue
            yield Path(root) / name


@describe(lambda a: f"list directory {a.get('path', '.')}")
@tool(parse_docstring=True)
def list_dir(path: str = ".", show_hidden: bool = False) -> str:
    """List the entries of a directory (dirs first, then files).

    Args:
        path: Directory to list.
        show_hidden: Include dotfiles when true.
    """
    p = Path(path)
    if not p.is_dir():
        return f"error: not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    lines = []
    for e in entries:
        if not show_hidden and e.name.startswith("."):
            continue
        if e.is_symlink():
            lines.append(f"{e.name} -> {os.readlink(e)}")
        elif e.is_dir():
            lines.append(f"{e.name}/")
        else:
            lines.append(f"{e.name}  ({_human_size(e.stat().st_size)})")
    header = f"{p} ({len(lines)} entries)"
    return f"{header}\n{_cap(lines)}" if lines else f"{header}\n(empty)"


@describe("inspect {path}")
@tool(parse_docstring=True)
def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a UTF-8 text file as numbered lines.

    Args:
        path: File to read.
        offset: 0-based line to start from.
        limit: Maximum number of lines to return.
    """
    p = Path(path)
    if not p.is_file():
        return f"error: not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: {path} is not a UTF-8 text file (binary?)"
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return f"{path} (empty file)"
    offset = max(0, offset)
    window = lines[offset : offset + limit]
    last = offset + len(window)
    width = len(str(last))
    body = "\n".join(f"{offset + k + 1:>{width}}| {ln}" for k, ln in enumerate(window))
    return f"{path} (lines {offset + 1}-{last} of {total})\n{body}"


@describe("find files matching {pattern!r}")
@tool(parse_docstring=True)
def glob(pattern: str, root: str = ".") -> str:
    """Find files and directories whose name matches a glob, newest first.

    Recurses from ``root`` (heavy dirs pruned), matching the basename of ``pattern``;
    returns a partial result if the walk exceeds the time budget.

    Args:
        pattern: Glob pattern, e.g. ``**/*.py``; only the basename part is matched.
        root: Directory to search under.
    """
    base = Path(root)
    if not base.is_dir():
        return f"error: not a directory: {root}"
    # match basenames; recursion + pruning are implicit
    name = pattern.rsplit("/", 1)[-1]
    deadline = time.monotonic() + _TIMEOUT
    matches: list[Path] = []
    timed_out = False
    for r, dirs, files in os.walk(base):
        if time.monotonic() > deadline:
            timed_out = True
            break
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        matches.extend(Path(r) / n for n in (*dirs, *files) if fnmatch(n, name))
        if len(matches) >= _MAX_MATCHES:
            break
    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    flag = "  [stopped: timeout]" if timed_out else ""
    if not matches:
        return f"no matches for {pattern!r} under {root}{flag}"
    return (
        f"{len(matches)} matches for {pattern!r} under {root} (newest first){flag}:\n"
        + _cap([str(p) for p in matches])
    )


@describe("search for {pattern!r}")
@tool(parse_docstring=True)
def grep(
    pattern: str,
    path: str = ".",
    glob: Optional[str] = None,
    context: int = 0,
    ignore_case: bool = False,
) -> str:
    """Search file contents for a regex, returning ``file:line: text`` matches.

    Args:
        pattern: Regular expression to search for.
        path: File or directory to search (directories are searched recursively).
        glob: Only search files whose name matches this glob, e.g. ``*.py``.
        context: Number of context lines to show around each match.
        ignore_case: Case-insensitive search when true.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"error: invalid regex {pattern!r}: {e}"
    base = Path(path)
    deadline = time.monotonic() + _TIMEOUT
    files = [base] if base.is_file() else _walk_files(base, glob, deadline)
    out: list[str] = []
    count = 0
    timed_out = False
    for f in files:
        if time.monotonic() > deadline:
            timed_out = True
            break
        try:
            flines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # skip binary / unreadable
        for i, line in enumerate(flines):
            if rx.search(line):
                lo, hi = max(0, i - context), min(len(flines), i + context + 1)
                for j in range(lo, hi):
                    sep = ":" if j == i else "-"
                    out.append(f"{f}:{j + 1}{sep} {flines[j]}")
                if context:
                    out.append("--")
                count += 1
                if count >= _MAX_MATCHES:
                    out.append(f"... [stopped at {_MAX_MATCHES} matches]")
                    return f"{count} matches for {pattern!r}:\n" + "\n".join(out)
    if not count:
        tail = "  [stopped: timeout]" if timed_out else ""
        return f"no matches for {pattern!r} in {path}{tail}"
    tail = "\n... [stopped: timeout]" if timed_out else ""
    return f"{count} matches for {pattern!r}:\n" + "\n".join(out) + tail


@describe("check metadata of {path}")
@tool(parse_docstring=True)
def path_info(path: str) -> str:
    """Report what a path is: kind, size, permissions, and mtime.

    Args:
        path: Path to inspect.
    """
    p = Path(path)
    try:
        st = p.lstat()  # lstat so symlinks report as symlinks
    except OSError:
        return f"path does not exist: {path}"
    mode = oct(stat.S_IMODE(st.st_mode))[2:]
    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    if stat.S_ISLNK(st.st_mode):
        status = "ok" if p.exists() else "broken"
        return f"{path}: kind=symlink target={os.readlink(p)} ({status}) mode={mode} mtime={mtime}"
    kind = "dir" if p.is_dir() else "file"
    return (
        f"{path}: kind={kind} size={_human_size(st.st_size)} mode={mode} mtime={mtime}"
    )


@describe(
    lambda a: (
        f"read env var {a['name']}"
        if a.get("name")
        else "read all environment variables"
    )
)
@tool(parse_docstring=True)
def get_env(name: Optional[str] = None) -> str:
    """Read environment variables -- one named var, or all of them.

    Args:
        name: Variable to read; omit to list every variable.
    """
    if name is None:
        return _cap([f"{k}={v}" for k, v in sorted(os.environ.items())])
    return f"{name}={os.environ[name]}" if name in os.environ else f"{name} is not set"


@describe("locate the {name!r} executable")
@tool(parse_docstring=True)
def which(name: str) -> str:
    """Locate an executable on PATH (is a tool installed, and where).

    Args:
        name: Executable name, e.g. ``python`` or ``git``.
    """
    found = shutil.which(name)
    return found if found else f"{name} not found on PATH"


@describe(lambda a: f"show the tree under {a.get('path', '.')}")
@tool(parse_docstring=True)
def tree(path: str = ".", depth: int = 2) -> str:
    """Print a directory tree to a given depth (dotfiles and heavy dirs pruned).

    Args:
        path: Root directory.
        depth: How many levels deep to descend.
    """
    base = Path(path)
    if not base.is_dir():
        return f"error: not a directory: {path}"
    lines = [f"{base}/"]
    _tree(base, depth, "  ", lines)
    return _cap(lines)


def _tree(d: Path, depth: int, prefix: str, lines: list[str]) -> None:
    if depth <= 0:
        return
    try:
        entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except OSError:
        return
    for e in entries:
        if e.name.startswith(".") or e.name in _SKIP_DIRS:
            continue
        if len(lines) >= _MAX_LINES:
            return
        if e.is_dir():
            lines.append(f"{prefix}{e.name}/")
            _tree(e, depth - 1, prefix + "  ", lines)
        else:
            lines.append(f"{prefix}{e.name}")


# bundle for one-line binding: llm.bind_tools(EXPLORE_TOOLS)
EXPLORE_TOOLS = [list_dir, read_file, glob, grep, path_info, get_env, which, tree]
