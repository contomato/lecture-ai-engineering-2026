"""素のLLM CLIの会話履歴テスト（LLMもAPIキーも使わない）。"""

import sys
from types import SimpleNamespace

from google.genai import types

from app import ask


def test_history_round_trip(tmp_path):
    path = tmp_path / "session" / "history.json"
    history = [
        types.Content(role="user", parts=[types.Part(text="LoRAとは？")]),
        types.Content(role="model", parts=[types.Part(text="追加学習の手法です。")]),
    ]

    assert ask.load_history(path) == []
    ask.save_history(path, history)
    assert ask.load_history(path) == history


def test_history_is_scoped_to_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first = ask.terminal_history_path("/dev/pts/1")
    same = ask.terminal_history_path("/dev/pts/1")
    other = ask.terminal_history_path("/dev/pts/2")

    assert first == same
    assert first != other
    assert first.parent == tmp_path / "aie2026"


def test_main_continues_history_with_short_and_long_options(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "history.json"
    calls = []

    def fake_ask(question, history):
        calls.append((question, history))
        updated_history = [
            *history,
            types.Content(role="user", parts=[types.Part(text=question)]),
        ]
        return f"回答: {question}", updated_history

    monkeypatch.setattr(ask, "terminal_history_path", lambda: path)
    monkeypatch.setattr(ask, "ask", fake_ask)

    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    ask.main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["app.ask", "--continue", "2つ目"])
    ask.main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["app.ask", "-c", "3つ目"])
    ask.main()

    assert calls[0] == ("最初の質問", [])
    assert calls[1][0] == "2つ目"
    assert [item.parts[0].text for item in calls[1][1]] == ["最初の質問"]
    assert calls[2][0] == "3つ目"
    assert [item.parts[0].text for item in calls[2][1]] == ["最初の質問", "2つ目"]
    assert "回答: 3つ目" in capsys.readouterr().out


def test_ask_sends_and_returns_conversation_history(monkeypatch):
    previous = types.Content(role="user", parts=[types.Part(text="最初の質問")])

    class FakeChat:
        def __init__(self, history):
            self.history = list(history)

        def send_message(self, question):
            assert question == "続きの質問"
            user_message = types.Content(
                role="user", parts=[types.Part(text=question)]
            )
            model_message = types.Content(
                role="model", parts=[types.Part(text="回答")]
            )
            self.history.extend([user_message, model_message])
            return SimpleNamespace(text="回答", usage_metadata=None)

        def get_history(self):
            return self.history

    class FakeChats:
        def create(self, *, model, history):
            assert model == ask.MODEL
            assert history == [previous]
            return FakeChat(history)

    class FakeClient:
        chats = FakeChats()

    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    answer, updated_history = ask.ask("続きの質問", [previous])

    assert "回答" in answer
    assert updated_history[0] == previous
    assert updated_history[1].parts[0].text == "続きの質問"
    assert updated_history[2].parts[0].text == "回答"
