"""Simple module for greeting users."""

from __future__ import annotations


def greeting(name: str) -> str:
    """Return a friendly greeting for the given ``name``."""
    return f"Hello, {name}!"


def main() -> None:
    """Print a greeting for the default user."""
    print(greeting("world"))


if __name__ == "__main__":
    main()
