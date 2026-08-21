import shutil
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from lang_scaffold.tools.explore import confine, note_base
from lang_scaffold.tools.observability import describe


def _files(p: Path) -> str:
    """A tree's file count, parenthesized; nothing for a single file."""
    if p.is_file():
        return ""
    return f" ({sum(1 for f in p.rglob('*') if f.is_file())} files)"


def _pair(resolve, src: str, dst: str, verb: str):
    """Resolve a ``src``/``dst`` pair -> ``(src, dst, None)`` or ``(None, None, error)``.

    Refusing an existing ``dst`` is what keeps these tools non-destructive; the
    containment check is what stops a directory being moved or copied into itself.
    A ``src`` that is ``base`` itself needs no separate check -- confinement puts every
    ``dst`` under ``base``, so containment already catches it.
    """
    s, err = resolve(src)
    if err:
        return None, None, err
    d, err = resolve(dst)
    if err:
        return None, None, err
    if not s.exists():
        return None, None, f"error: path does not exist: {src}"
    sr, dr = s.resolve(), d.resolve()
    # before the exists check below, which would otherwise claim a no-op is occupied
    if sr == dr:
        return None, None, f"error: {src} and {dst} are the same path"
    if d.exists() or d.is_symlink():
        return (
            None,
            None,
            f"error: {dst} already exists -- this tool never writes over an existing "
            f"path; {verb} to a new path, or change {dst} in place with edit_file",
        )
    if s.is_dir() and dr.is_relative_to(sr):
        return (
            None,
            None,
            f"error: {dst} is inside {src} -- cannot {verb} a directory into itself",
        )
    return s, d, None


def build_fs_tools(base: Optional[str] = None) -> list:
    """Build the path-structure tools (``make_dir``, ``move_path``, ``copy_path``).

    ``base`` confines every path exactly as in ``build_explore_tools``. Kept apart from
    ``build_edit_tools``, and needing no ``ReadTracker``: these tools relocate paths
    without touching contents and refuse an existing destination, so there is nothing
    for a read-before-write guard to protect.
    """
    _resolve, _base = confine(base)

    @describe("create directory {path}")
    @tool(parse_docstring=True)
    def make_dir(path: str) -> str:
        """Create a directory, including any missing parent directories.

        Only needed for a directory that must exist while empty -- ``write_file``
        already creates the parents of the file it writes.

        Args:
            path: Directory to create.
        """
        p, err = _resolve(path)
        if err:
            return err
        if p.is_dir():
            return f"{path}/ already exists"
        if p.exists() or p.is_symlink():
            return f"error: {path} exists and is not a directory"
        try:
            p.mkdir(parents=True)
        except OSError as e:
            return f"error: cannot create {path}: {e}"
        return f"created {path}/"

    @describe("move {src} to {dst}")
    @tool(parse_docstring=True)
    def move_path(src: str, dst: str) -> str:
        """Move or rename a file or directory. Fails if ``dst`` already exists.

        Missing parent directories of ``dst`` are created.

        Args:
            src: File or directory to move.
            dst: Path to move it to; must not exist.
        """
        s, d, err = _pair(_resolve, src, dst, "move")
        if err:
            return err
        kind, count = ("directory" if s.is_dir() else "file"), _files(s)
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))  # falls back to copy+remove across filesystems
        except (OSError, shutil.Error) as e:
            return f"error: cannot move {src} to {dst}: {e}"
        return f"moved {kind} {src} -> {dst}{count}"

    @describe("copy {src} to {dst}")
    @tool(parse_docstring=True)
    def copy_path(src: str, dst: str) -> str:
        """Copy a file or directory tree. Fails if ``dst`` already exists.

        Missing parent directories of ``dst`` are created.

        Args:
            src: File or directory to copy.
            dst: Path to copy it to; must not exist.
        """
        s, d, err = _pair(_resolve, src, dst, "copy")
        if err:
            return err
        is_dir = s.is_dir()
        kind, count = ("directory" if is_dir else "file"), _files(s)
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            if is_dir:
                shutil.copytree(s, d, symlinks=True)  # links stay links, never inlined
            else:
                shutil.copy2(s, d)  # preserves mode and mtime
        except (OSError, shutil.Error) as e:
            # a tree copy is not atomic, and cleaning up would mean deleting -- which
            # these tools never do, so say what was left behind instead
            partial = f"; {dst} may be incomplete" if is_dir else ""
            return f"error: cannot copy {src} to {dst}: {e}{partial}"
        return f"copied {kind} {src} -> {dst}{count}"

    return note_base([make_dir, move_path, copy_path], _base)
