"""Verification tests for Module 3: Prompt Ownership.

Proves: rendering fails loudly with attribution when a variable is
missing, and the registry enforces additive versioning (no silent
overwrites, "latest" resolves correctly, unknown lookups fail clearly).
"""

import pytest

from agent_harness.prompts.template import PromptRegistry, PromptTemplate


def make_template(version: str) -> PromptTemplate:
    return PromptTemplate(
        name="greet",
        version=version,
        system_tmpl="You are a helpful assistant.",
        user_tmpl="Say hello to $name in $language.",
    )


def test_render_fills_in_variables() -> None:
    tmpl = make_template("v1")

    rendered = tmpl.render(name="Sam", language="French")

    assert rendered.user == "Say hello to Sam in French."
    assert rendered.prompt_id == "greet@v1"


def test_render_raises_with_prompt_attribution_on_missing_variable() -> None:
    tmpl = make_template("v1")

    with pytest.raises(ValueError, match="greet@v1"):
        tmpl.render(name="Sam")  # missing 'language'


def test_registry_get_specific_version() -> None:
    registry = PromptRegistry()
    registry.register(make_template("v1"))
    registry.register(make_template("v2"))

    assert registry.get("greet", "v1").version == "v1"
    assert registry.get("greet", "v2").version == "v2"


def test_registry_latest_is_most_recently_registered() -> None:
    registry = PromptRegistry()
    registry.register(make_template("v1"))
    registry.register(make_template("v2"))

    assert registry.get("greet").version == "v2"
    assert registry.get("greet", "latest").version == "v2"


def test_registry_rejects_duplicate_version() -> None:
    registry = PromptRegistry()
    registry.register(make_template("v1"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_template("v1"))


def test_registry_unknown_name_raises() -> None:
    registry = PromptRegistry()

    with pytest.raises(KeyError, match="No prompt registered"):
        registry.get("does_not_exist")


def test_registry_unknown_version_raises() -> None:
    registry = PromptRegistry()
    registry.register(make_template("v1"))

    with pytest.raises(KeyError, match="no version 'v2'"):
        registry.get("greet", "v2")
