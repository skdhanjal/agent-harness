"""Prompt Ownership: prompts as versioned, first-class code artifacts.

Untracked f-strings scattered through business logic make prompts
undebuggable and easy to silently drift. This module makes every prompt:
1. A named, versioned artifact -- never edited in place, only added to.
2. Strict about its variables -- a missing one fails loudly, with the
   exact prompt name/version attached, instead of silently blanking.
3. Split into system/user roles explicitly, since providers treat them
   with different instruction priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    prompt_id: str


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system_tmpl: str
    user_tmpl: str

    def render(self, **kwargs: str) -> RenderedPrompt:
        try:
            system = Template(self.system_tmpl).substitute(**kwargs)
            user = Template(self.user_tmpl).substitute(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing prompt variable {e} for '{self.name}@{self.version}'") from e
        return RenderedPrompt(system=system, user=user, prompt_id=f"{self.name}@{self.version}")


class PromptRegistry:
    """Holds every version of every prompt. Versions are additive, never overwritten.

    "latest" means most-recently-registered, not alphabetically/numerically
    greatest -- this sidesteps the "v9 vs v10" string-sorting trap entirely.
    """

    def __init__(self) -> None:
        self._prompts: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, tmpl: PromptTemplate) -> None:
        versions = self._prompts.setdefault(tmpl.name, {})
        if tmpl.version in versions:
            raise ValueError(
                f"Prompt '{tmpl.name}@{tmpl.version}' already registered -- "
                "bump the version instead of overwriting it"
            )
        versions[tmpl.version] = tmpl

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        if name not in self._prompts:
            raise KeyError(f"No prompt registered under name '{name}'")
        versions = self._prompts[name]
        if version == "latest":
            return next(reversed(versions.values()))
        if version not in versions:
            raise KeyError(f"Prompt '{name}' has no version '{version}'")
        return versions[version]
