from src.main import greeting


def test_greeting() -> None:
    assert greeting("World") == "Hello, World!"
