import argparse
import os

from langchain.chat_models import init_chat_model

from lang_scaffold.summary import load_messages, summarize_conversation

THREAD = "main"  # matches coding_agent.py's constant thread


def main():
    p = argparse.ArgumentParser(description="Summarize a stored agent conversation.")
    p.add_argument("db", help="conversation .db file")
    p.add_argument("--max-words", type=int, default=100, help="soft length limit")
    args = p.parse_args()

    llm = init_chat_model(
        os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ["LLM_API_KEY"],
    )
    n = len(load_messages(args.db, thread=THREAD))
    print(f"{args.db}: {n} messages\n")
    print(summarize_conversation(llm, args.db, thread=THREAD, max_words=args.max_words))


if __name__ == "__main__":
    main()
