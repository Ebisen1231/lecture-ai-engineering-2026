"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""


import argparse
import csv
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# 比較する質問
QUESTIONS = [
    "LoRAとは何ですか？",
    "LLMの量子化とは何ですか？",
    DEFAULT_QUESTION,
]


def measure(response, elapsed: float) -> dict:
    """応答時間とトークン数を取得する"""
    usage = response.usage_metadata

    return {
        "time_sec": round(elapsed, 3),
        "input_tokens": usage.prompt_token_count if usage else None,
        "output_tokens": usage.candidates_token_count if usage else None,
        "thinking_tokens": usage.thoughts_token_count if usage else None,
        "total_tokens": usage.total_token_count if usage else None,
    }


def chat_mode(client, question: str) -> None:
    """会話履歴を保持する通常モード"""
    chat = client.chats.create(model=MODEL)

    print(f"[model] {MODEL}")

    while True:
        if question:
            try:
                start = time.perf_counter()
                response = chat.send_message(question)
                elapsed = time.perf_counter() - start
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                raise SystemExit(1)

            print(f"\n{response.text or '(回答なし)'}")
            print(f"\n[計測結果] {measure(response, elapsed)}")

        try:
            question = input("\n次の質問 (exitで終了) > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() == "exit":
            break


def compare_mode(client, models: list[str], repeat: int) -> None:
    """モデルと質問の組み合わせを比較する"""
    results = []

    for model in models:
        for question in QUESTIONS:
            for trial in range(1, repeat + 1):
                print(f"\n[モデル] {model}")
                print(f"[質問] {question}")
                print(f"[試行] {trial}")

                start = time.perf_counter()

                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=question,
                    )
                except errors.APIError as e:
                    print(api_error_message(e.code, e.message))
                    continue

                elapsed = time.perf_counter() - start
                stats = measure(response, elapsed)

                results.append({
                    "model": model,
                    "question": question,
                    "trial": trial,
                    **stats,
                    "answer": response.text or "",
                })

                print(f"[応答時間] {stats['time_sec']} 秒")
                print(f"[入力] {stats['input_tokens']} tokens")
                print(f"[出力] {stats['output_tokens']} tokens")
                print(f"[推論] {stats['thinking_tokens']} tokens")
                print(f"[合計] {stats['total_tokens']} tokens")

    # CSVへ保存
    if results:
        with open(
            "comparison.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print("\n比較結果を comparison.csv に保存しました。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--models", nargs="+", default=[MODEL])
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat は1以上にしてください")

    client = genai.Client(api_key=get_api_key())

    if args.compare:
        compare_mode(client, args.models, args.repeat)
    else:
        chat_mode(client, args.question)


if __name__ == "__main__":
    main()
