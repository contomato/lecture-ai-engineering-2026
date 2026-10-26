"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --continue "もう少し詳しく"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールを使わない「素の」呼び出し。通常実行では前の質問を会話履歴に含めない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
--continue を指定すると、同じターミナルの前回の実行から会話を続ける。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def terminal_history_path(tty_name: str | None = None) -> Path | None:
    if tty_name is None:
        if not sys.stdin.isatty():
            return None
        tty_name = os.ttyname(sys.stdin.fileno())

    # 現時点では同一ターミナルの前回の実行だけを維持し、ターミナルをまたぐ会話管理はしない。
    terminal_id = hashlib.sha256(tty_name.encode("utf-8")).hexdigest()[:16]
    cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "aie2026" / f"ask-{terminal_id}.json"


def load_history(path: Path) -> list[types.Content]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("会話履歴の形式が正しくありません")
        return [types.Content.model_validate(item) for item in data]
    except ValueError as e:
        raise ValueError(f"会話履歴を読み込めません: {e}") from e


def save_history(path: Path, history: list[types.Content]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary_path = Path(file.name)
            json.dump(
                [content.model_dump(mode="json", exclude_none=True) for content in history],
                file,
                ensure_ascii=False,
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ask(
    question: str, history: list[types.Content] | None = None
) -> tuple[str, list[types.Content]]:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL, history=history or [])
    response = chat.send_message(question)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None

    answer = (
        f"{response.text}\n\n"
        f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}\n"
        f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}"
    )
    return answer, chat.get_history()


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_conversation",
        action="store_true",
        help="同じターミナルの前回の会話を続ける",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        history_path = terminal_history_path()
        if args.continue_conversation and history_path is None:
            parser.error("--continue は対話的なターミナルで実行してください")
        history = (
            load_history(history_path)
            if args.continue_conversation and history_path is not None
            else []
        )
        answer, updated_history = ask(args.question, history)
        if history_path is not None:
            save_history(history_path, updated_history)
        print(answer)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except (OSError, ValueError) as e:
        raise SystemExit(f"会話履歴を扱えません: {e}") from e


if __name__ == "__main__":
    main()
