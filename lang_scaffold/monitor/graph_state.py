from typing import Any, Callable, Optional

import json

from pydantic import BaseModel


class TracedGraph:
    """Wraps a compiled LangGraph to log state transitions (the graph-state layer).

    Generic: works with any compiled graph regardless of its state schema
    (dict / TypedDict / pydantic / dataclass). Sees only graph state in/out, not
    the LLM prompts inside the nodes -- use PromptLogger for those.
    """

    def __init__(
        self,
        graph: Any,
        verbose: bool = True,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            graph: Compiled StateGraph from langgraph.
            verbose: Whether to log output.
            log_fn: Custom logging function (defaults to print).
        """
        self.graph = graph
        self.verbose = verbose
        self.log_fn = log_fn or print

    @staticmethod
    def _as_dict(state: Any) -> dict:
        """Normalize any LangGraph state to a shallow dict for logging."""
        if isinstance(state, dict):
            return state
        if isinstance(state, BaseModel):
            return dict(state)  # pydantic v2: shallow, keeps nested objects intact
        return dict(getattr(state, "__dict__", {}))  # dataclass / plain object

    def _format_state(self, state: Any) -> str:
        """Format state for readable output."""
        try:
            return json.dumps(self._as_dict(state), indent=2, default=str)
        except Exception:
            return str(state)

    def invoke(self, input_state: Any, **kwargs) -> dict:
        """Invoke graph with tracing."""
        if self.verbose:
            self._log_section("INPUT STATE")
            self._log(self._format_state(input_state))

        try:
            output_state = self.graph.invoke(input_state, **kwargs)
            if self.verbose:
                self._log_section("OUTPUT STATE")
                self._log(self._format_state(output_state))
                self._log_diff(input_state, output_state)
            return output_state
        except Exception as e:
            if self.verbose:
                self._log_section("ERROR")
                self._log(f"{type(e).__name__}: {e}")
            raise

    def stream(self, input_state: Any, **kwargs):
        """Stream graph execution with tracing."""
        if self.verbose:
            self._log_section("STREAM START")
            self._log(self._format_state(input_state))

        step = 0
        for output in self.graph.stream(input_state, **kwargs):
            if self.verbose:
                step += 1
                self._log_section(f"STEP {step}")
                self._log(self._format_state(output))
            yield output

        if self.verbose:
            self._log_section("STREAM END")

    def _log_section(self, title: str) -> None:
        """Log a section header."""
        self._log(f"\n{'=' * 60}")
        self._log(f"  {title}")
        self._log(f"{'=' * 60}")

    def _log(self, message: str) -> None:
        """Log a message."""
        if self.verbose:
            self.log_fn(message)

    def _log_diff(self, before: Any, after: Any) -> None:
        """Log what changed between states."""
        before, after = self._as_dict(before), self._as_dict(after)
        changed, added, removed = {}, {}, {}
        for key in set(list(before.keys()) + list(after.keys())):
            before_val = before.get(key)
            after_val = after.get(key)
            if key not in before:
                added[key] = after_val
            elif key not in after:
                removed[key] = before_val
            elif before_val != after_val:
                changed[key] = {"before": before_val, "after": after_val}

        if changed or added or removed:
            self._log_section("STATE CHANGES")
            if changed:
                self._log("CHANGED:")
                for key, diff in changed.items():
                    self._log(f"  {key}:")
                    self._log(f"    before: {diff['before']}")
                    self._log(f"    after: {diff['after']}")
            if added:
                self._log("ADDED:")
                for key, val in added.items():
                    self._log(f"  {key}: {val}")
            if removed:
                self._log("REMOVED:")
                for key, val in removed.items():
                    self._log(f"  {key}: {val}")
