"""Command-line demo of the HR Policy Assistant.

Run with:  python main.py
"""

import sys

# The Windows console defaults to cp1252, which can't print characters like
# the non-breaking hyphen the LLM sometimes produces.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hr_assistant.pipeline import ask , build_hr_assistant
from hr_assistant.logger import get_logger
logger = get_logger(__name__)

def main():
    logger.info("=== CLI run started ===")
    agent = build_hr_assistant()

    demo_questions = [
        "How many paid annual leave days do I get?",
        "What is the notice period during probation?",
        "Can I work from home every day?",
    ]

    for question in demo_questions:
        print("=" * 60)
        print("QUESTION:", question)
        print("-" * 60)
        answer = ask(agent, question)
        print("ANSWER:", answer)
        print("=" * 60)
        print()
        
        logger.info("=== CLI run finished ===")


if __name__ == "__main__":
    main()