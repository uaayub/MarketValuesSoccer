"""
Player Query Agent
------------------
A small AI-augmented tool that lets you ask natural-language questions
about a soccer player dataset. Instead of hardcoding logic, it uses the
Claude API to translate your question into pandas code, executes that
code against the dataframe, and returns the result.

BIG PICTURE / INTERVIEW PITCH:
This is a two-hop pipeline: natural language -> code -> execution -> natural
language. It's a lightweight example of "agentic" behavior: the LLM decides
*what action to take* (what pandas code to run), and the program executes
that action and reports back the outcome. The schema summary below acts as
a "map" that lets Claude write accurate code without ever touching the real
27k-row dataset directly.

Setup:
    pip install anthropic pandas
    export ANTHROPIC_API_KEY="your-key-here"

Usage:
    python player_query_agent.py
"""

import os
import re
import sys
import pandas as pd
from anthropic import Anthropic

CSV_PATH = "player_attributes.csv"
MODEL = "claude-sonnet-5"


def load_data(path: str) -> pd.DataFrame:
    # Just reads the CSV into a DataFrame. Kept as its own function for
    # clarity/reuse rather than because it needs to be complicated.
    df = pd.read_csv(path)
    return df


def build_schema_summary(df: pd.DataFrame) -> str:
    """
    WHY THIS EXISTS: sending all 27,290 rows to Claude on every question
    would blow past reasonable token limits and cost. Instead, this builds
    a compact text description of each column (name, type, and range or
    sample values) ONCE at startup. Claude never sees the actual data --
    it only sees this description, then writes code that operates on the
    real dataframe locally. This is the same pattern used in production
    RAG/data-agent systems: give the model a map, not the whole territory.
    """
    lines = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == bool:
            lines.append(f"- {col}: boolean")
        elif pd.api.types.is_numeric_dtype(dtype):
            lines.append(
                f"- {col}: {dtype} (min={df[col].min():.2f}, max={df[col].max():.2f})"
            )
        else:
            sample = df[col].dropna().unique()[:3]
            lines.append(f"- {col}: string (e.g. {list(sample)})")
    return "\n".join(lines)


def ask_claude_for_code(client: Anthropic, schema: str, question: str) -> str:
    """
    API CALL #1 of 2: question -> pandas code.

    This call is deliberately narrow-scoped -- the system prompt forces
    Claude to output ONLY a line of code, nothing else. Splitting "write
    code" and "explain the answer" (see explain_result below) into two
    separate calls keeps each prompt focused on one job, which makes the
    output more reliable than asking one call to do both at once.
    """
    system_prompt = f"""You are a pandas code generator. You will be given
the schema of a dataframe called `df` and a natural language question.

Respond with ONLY a single line of valid Python/pandas code that computes
the answer and assigns it to a variable called `result`. No explanation,
no markdown, no backticks — just the code.

Dataframe schema:
{schema}
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    code = response.content[0].text.strip()
    # Defensive cleanup: models don't always perfectly follow formatting
    # instructions, so strip markdown fences if Claude adds them anyway.
    code = re.sub(r"^```(?:python)?|```$", "", code, flags=re.MULTILINE).strip()
    return code


def run_generated_code(df: pd.DataFrame, code: str):
    """
    Executes the Claude-generated code with Python's exec().

    SECURITY NOTE (bring this up proactively in the interview -- it shows
    judgment, not just glue code): the namespace passed to exec() is
    restricted to only `df` and `pd`, and `__builtins__` is blanked out,
    so the generated code can't read files, import modules, or make
    network/system calls. That said, this is a DEMO-LEVEL guardrail, not
    production-safe -- a sufficiently adversarial prompt could still find
    edge cases. In a real system I'd run generated code in an isolated
    subprocess/sandbox with no filesystem or network access, or avoid
    arbitrary code execution entirely by constraining the model to a
    fixed set of allowed operations.
    """
    local_vars = {"df": df, "pd": pd}
    exec(code, {"__builtins__": {}}, local_vars)
    return local_vars.get("result")


def explain_result(client: Anthropic, question: str, code: str, result) -> str:
    """
    API CALL #2 of 2: raw result -> plain English.

    Takes the original question, the code that ran, and the raw pandas
    output (which might just be a bare number or name) and asks Claude to
    phrase it as a natural sentence. This is the step that turns
    "Erling Haaland" into "The player with the highest market value under
    23 is Erling Haaland."
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Code run: {code}\n"
                    f"Raw result: {result}\n\n"
                    "Answer the question in one or two plain-English sentences "
                    "based on the raw result. Don't mention the code."
                ),
            }
        ],
    )
    return response.content[0].text.strip()


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY before running this script.")
        sys.exit(1)

    print("Loading player_attributes.csv...")
    df = load_data(CSV_PATH)
    # Schema is built ONCE here, not per-question -- rebuilding it every
    # loop iteration would waste tokens on something that never changes.
    schema = build_schema_summary(df)
    client = Anthropic()

    print(f"Loaded {len(df):,} players. Ask a question (or 'quit' to exit).\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        try:
            code = ask_claude_for_code(client, schema, question)
            print(f"  [generated code] {code}")
            result = run_generated_code(df, code)
            answer = explain_result(client, question, code, result)
            print(f"  {answer}\n")
        except Exception as e:
            # try/except here means one bad question (e.g. Claude generates
            # code that errors) doesn't crash the whole session -- it just
            # prints the error and loops back for another question.
            # IMPROVEMENT I'D MAKE: catch the exception, send the error
            # message back to Claude, and ask it to self-correct the code.
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()