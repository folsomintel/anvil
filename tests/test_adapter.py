from __future__ import annotations

from harness import GenerativeModel, flatten

CALL = "<tool>\nname: run_tests\n</tool>"


def test_flatten_has_no_role_tags():
    messages = [
        {"role": "system", "content": "tools:"},
        {"role": "user", "content": "task: fix"},
        {"role": "assistant", "content": CALL},
    ]
    assert flatten(messages) == "tools:\ntask: fix\n" + CALL


def test_prompt_is_the_flattened_transcript():
    seen: dict[str, object] = {}

    def generate(prompt: str, max_new_tokens: int, stop: list[str]) -> str:
        seen["prompt"] = prompt
        seen["stop"] = stop
        return CALL

    model = GenerativeModel(generate=generate)
    model.step([{"role": "user", "content": "task: fix"}])
    assert seen["prompt"] == "task: fix\n"
    assert seen["stop"] == ["</tool>"]


def test_closing_tag_is_restored_when_the_stop_ate_it():
    model = GenerativeModel(generate=lambda *a, **k: "<tool>\nname: run_tests\n")
    assert model.step([]) == CALL


def test_output_after_the_closing_tag_is_discarded():
    noisy = CALL + "\nand then i will read the file"
    model = GenerativeModel(generate=lambda *a, **k: noisy)
    assert model.step([]) == CALL


def test_junk_without_a_block_is_passed_through_for_the_parser_to_reject():
    model = GenerativeModel(generate=lambda *a, **k: "i think the bug is in clamp")
    assert model.step([]) == "i think the bug is in clamp"


def test_stop_can_be_disabled():
    seen: dict[str, object] = {}

    def generate(prompt: str, max_new_tokens: int, stop: list[str]) -> str:
        seen["stop"] = stop
        return CALL

    GenerativeModel(generate=generate, stop_supported=False).step([])
    assert seen["stop"] == []
