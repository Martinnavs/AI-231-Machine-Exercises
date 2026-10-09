"""Experimental rejection grammar for incomplete Option B commands.

This module is intentionally separate from :mod:`vcm.optionb.grammar` while
the decoder policy is being validated.  It adds no accepted commands and does
not alter CTC training targets.  Instead, it identifies proper whole-word
prefixes such as ``"color"`` that can compete with completed commands during
inference and cause an explicit rejection.

Once the policy has been validated against the cluster's held-out Option B
data, this metadata can be moved into the shared ``Grammar`` representation.
"""

from __future__ import annotations

from dataclasses import dataclass

# `derive_incomplete_prefixes` moved to `common.grammar_core` (it is
# grammar-generic, per the spec's "preferred merge"); re-exported here so the
# module's public API -- and every existing import of it -- is unchanged.
from me2_voicegen.common.grammar_core import Grammar, derive_incomplete_prefixes

from .grammar import OPTIONB_GRAMMAR


@dataclass(frozen=True)
class IncompletePrefixRejectionGrammar:
    """Inference-only rejection metadata layered over a command grammar."""

    command_grammar: Grammar
    incomplete_prefixes: frozenset[str]

    @classmethod
    def from_command_grammar(cls, grammar: Grammar) -> "IncompletePrefixRejectionGrammar":
        rejection_grammar = cls(
            command_grammar=grammar,
            incomplete_prefixes=derive_incomplete_prefixes(grammar),
        )
        rejection_grammar.validate()
        return rejection_grammar

    def is_incomplete(self, text: str) -> bool:
        """Whether ``text`` is a designated incomplete command prefix."""
        return text in self.incomplete_prefixes

    def completions_for(self, prefix: str) -> tuple[str, ...]:
        """Accepted commands reached by adding whole words to ``prefix``."""
        word_prefix = f"{prefix} "
        return tuple(
            sorted(
                text
                for text, _, _ in self.command_grammar.all_phrases()
                if text.startswith(word_prefix)
            )
        )

    def validate(self) -> None:
        """Raise if rejection metadata changes command-grammar semantics."""
        accepted = {text for text, _, _ in self.command_grammar.all_phrases()}
        overlap = accepted.intersection(self.incomplete_prefixes)
        if overlap:
            raise ValueError(f"accepted commands marked incomplete: {sorted(overlap)!r}")

        for prefix in self.incomplete_prefixes:
            if not prefix or not self.completions_for(prefix):
                raise ValueError(f"not a proper whole-word command prefix: {prefix!r}")

            node = self.command_grammar.root
            for character in prefix:
                node = node.children.get(character)
                if node is None:
                    raise ValueError(f"prefix is not reachable in command trie: {prefix!r}")
            if node.terminal is not None:
                raise ValueError(f"incomplete prefix has a terminal intent: {prefix!r}")


OPTIONB_INCOMPLETE_PREFIX_GRAMMAR = IncompletePrefixRejectionGrammar.from_command_grammar(
    OPTIONB_GRAMMAR
)
