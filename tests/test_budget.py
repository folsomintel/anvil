from __future__ import annotations

from harness import Budget, build_registry
from harness.prompt import build_system_prompt

WORDS = ["def", "clamp", "value", "return", "assert", "import"]


def words(count: int) -> str:
    return " ".join(WORDS[index % len(WORDS)] for index in range(count))


def message(role: str, text: str) -> dict[str, str]:
    return {"role": role, "content": text}


def test_counter_is_pluggable():
    budget = Budget(count=lambda text: len(text.split()))
    assert budget.tokens([message("user", "a b c")]) == 3


def test_pinned_turns_survive_trimming():
    budget = Budget(window=64, reserve_for_output=16)
    messages = [
        message("system", "tools: read_file"),
        message("user", "task: fix clamp"),
        *[message("tool", words(30)) for _ in range(6)],
    ]
    fitted, dropped = budget.fit(messages)
    assert dropped > 0
    assert fitted[0]["content"] == "tools: read_file"
    assert fitted[1]["content"] == "task: fix clamp"
    assert budget.fits(fitted)


def test_the_newest_turn_is_kept():
    budget = Budget(window=64, reserve_for_output=16)
    messages = [
        message("system", "s"),
        message("user", "t"),
        *[message("tool", words(20)) for _ in range(5)],
        message("tool", "newest"),
    ]
    fitted, _ = budget.fit(messages)
    assert fitted[-1]["content"] == "newest"


def test_dropping_leaves_a_note():
    budget = Budget(window=80, reserve_for_output=16)
    messages = [
        message("system", "s"),
        message("user", "t"),
        *[message("tool", words(20)) for _ in range(5)],
    ]
    fitted, dropped = budget.fit(messages)
    assert dropped > 0
    assert any("earlier turns dropped" in m["content"] for m in fitted)


def test_short_transcripts_are_untouched():
    budget = Budget(window=512)
    messages = [message("system", "s"), message("user", "t"), message("tool", "r")]
    fitted, dropped = budget.fit(messages)
    assert dropped == 0
    assert fitted == messages


def test_clip_respects_the_ceiling():
    budget = Budget()
    text = words(200)
    clipped = budget.clip(text, 20)
    assert budget.count(clipped) <= 20
    assert text.startswith(clipped)


def test_clip_leaves_short_text_alone():
    budget = Budget()
    assert budget.clip("short", 50) == "short"


def test_compact_prompt_fits_a_512_window():
    """the whole point of compact mode: the prompt must not eat the window."""
    budget = Budget(window=512)
    compact = build_system_prompt(build_registry(), compact=True)
    verbose = build_system_prompt(build_registry(), compact=False)
    assert budget.count(compact) < 120
    assert budget.count(verbose) > budget.count(compact) * 3
