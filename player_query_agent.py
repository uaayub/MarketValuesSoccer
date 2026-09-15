"""
Player Query Agent
------------------
A small AI-augmented tool that lets you ask natural-language questions
about a soccer player dataset. Instead of hardcoding logic, it uses the
Claude API to translate your question into pandas code, executes that
code against the dataframe, and returns the result.

This is a lightweight example of "agentic" behavior: the LLM decides
*what action to take* (what pandas code to run), and the program
executes that action and reports back the outcome.

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
MODEL = "claude-sonnet-4-6"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_schema_summary(df: pd.DataFrame) -> str:
    """Give the model a compact description of the dataframe instead of
    the whole dataset, so it can write accurate pandas code without us
    sending 27k rows through the API."""
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
    """Ask Claude to translate a natural language question into a single
    line of pandas code that operates on a dataframe called `df`."""
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
    # Strip accidental markdown fences if the model adds them anyway
    code = re.sub(r"^```(?:python)?|```$", "", code, flags=re.MULTILINE).strip()
    return code


def run_generated_code(df: pd.DataFrame, code: str):
    """Execute the generated code in a restricted namespace and return
    the resulting value. This is a demo, so the sandboxing here is
    minimal — in a production system you'd want much stricter execution
    controls (e.g. a subprocess with no filesystem/network access)."""
    local_vars = {"df": df, "pd": pd}
    exec(code, {"__builtins__": {}}, local_vars)
    return local_vars.get("result")


def explain_result(client: Anthropic, question: str, code: str, result) -> str:
    """Ask Claude to turn the raw pandas output into a plain-English answer."""
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
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()