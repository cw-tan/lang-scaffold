from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver


def load_messages(path, thread="main"):
    """Load a conversation's stored messages straight from its checkpoint file."""
    with SqliteSaver.from_conn_string(str(path)) as cp:
        tup = cp.get_tuple({"configurable": {"thread_id": thread}})
        if tup is None:
            return []
        return list(tup.checkpoint["channel_values"].get("messages", []))


def _format_transcript(messages, tool_chars=200):
    """Format stored message objects into a compact text transcript.

    Long tool outputs are shortened so a single file read can't bloat the request;
    this trims the LLM's *input*, not the summary length.
    """
    lines = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if m.type == "human":
            lines.append(f"User: {content}")
        elif m.type == "ai":
            if content.strip():
                lines.append(f"Assistant: {content}")
            for tc in getattr(m, "tool_calls", None) or []:
                lines.append(f"Assistant: (called {tc['name']}: {tc['args']})")
        elif m.type == "tool":
            if len(content) > tool_chars:
                content = content[:tool_chars] + " ...[truncated]"
            lines.append(f"Tool: {content}")
    return "\n".join(lines)


def summarize_conversation(llm, path, thread="main", max_words=100):
    """Summarize a stored conversation in at most ~``max_words`` words (soft limit)."""
    messages = load_messages(path, thread)
    if not messages:
        return "(no conversation found)"
    transcript = _format_transcript(messages)
    return llm.invoke(
        [
            SystemMessage(
                f"Summarize this conversation for the user in at most {max_words} words."
            ),
            HumanMessage(transcript),
        ]
    ).content
