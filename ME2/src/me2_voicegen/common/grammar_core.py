"""Dataset-agnostic grammar combinator/trie machinery, shared by `vcm` and
`optionb` (and any future grammar-based dataset). Holds only the reusable
building blocks -- none of the VCM- or Option B-specific rules, vocabularies,
or intent taxonomies, which live in each dataset's own `grammar.py`.

- `literal`, `opt`, `alt`, `seq`, `slot` build "alternative lists"
  (`list[tuple[tuple[str, ...], dict[str, str]]]`, aliased `_Alt`): a list of
  (word-sequence, slots-so-far) options. Small enough for any grammar in this
  repo's current scale to fully enumerate at import time rather than build a
  generalized NFA/WFST.
- `with_intent` stamps a caller-chosen intent label onto every alternative in
  a finished component, turning each (words, slots) pair into a (words,
  slots, intent) triple ready for trie insertion. It is deliberately generic
  over what "intent" means -- callers decide their own taxonomy and how
  fine-grained (per-rule vs per-alternative) to stamp it.
- `_vocab` turns a flat list of value strings into (word-tuple, value) pairs
  for use with `slot`.
- `TrieNode`/`_insert`/`compile_grammar`/`Grammar` compile a `rules` dict
  (rule name -> list of (words, slots, intent) triples) into a single shared
  character-level trie, keyed by each alternative's full text joined with
  single spaces. `Grammar.accepts` walks that trie for direct string
  acceptance (used by `optionb`); a decoder built against the same trie can
  instead do CTC prefix beam search over it (used by `vcm.decoder`) --
  either usage is representation-neutral from this module's point of view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Grammar combinators. All operate on "alternative lists":
#   list[tuple[tuple[str, ...], dict[str, str]]]
# i.e. a list of (word-sequence, slots-so-far) options. `with_intent` is the
# only combinator that introduces the third (intent) element, applied once
# per finished CMD_* alternative (see module docstring).
# ---------------------------------------------------------------------------

_Alt = list  # alias for readability: list[tuple[tuple[str, ...], dict]]

EMPTY: list = [((), {})]


def literal(*words: str) -> list:
    return [(tuple(words), {})]


def opt(component: list) -> list:
    """`(...)?` / `[...]` -- the component, or nothing."""
    return list(component) + list(EMPTY)


def alt(*components: list) -> list:
    """`a | b | ...`"""
    out: list = []
    for c in components:
        out.extend(c)
    return out


def seq(*components: list) -> list:
    """Concatenation: cartesian product across components, merging slots."""
    results: list = [((), {})]
    for comp in components:
        new_results: list = []
        for words_a, slots_a in results:
            for words_b, slots_b in comp:
                merged_slots = {**slots_a, **slots_b}
                new_results.append((words_a + words_b, merged_slots))
        results = new_results
    return results


def slot(name: str, values: list[tuple[tuple[str, ...], str]]) -> list:
    """A slot vocabulary reference: each value contributes one alternative,
    recording `slots[name] = value` (the vocab's own value, not the words)."""
    return [(words, {name: value}) for words, value in values]


def with_intent(intent: str, component: list) -> list:
    """Stamp `intent` onto every (words, slots) alternative in `component`,
    finishing it into a (words, slots, intent) triple ready for trie
    insertion."""
    return [(words, slots_, intent) for words, slots_ in component]


def _vocab(*texts: str) -> list[tuple[tuple[str, ...], str]]:
    return [(tuple(t.split()), t) for t in texts]


# ---------------------------------------------------------------------------
# Character-trie compilation.
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        # list of (intent, slots) -- a list (not a single value) in case two
        # distinct alternatives ever produce the exact same text; empty
        # grammar here never hits that, but decoding must not silently drop
        # a collision if the grammar changes later.
        self.terminal: list[tuple[str, dict]] | None = None


def _insert(root: TrieNode, text: str, intent: str, slots: dict) -> None:
    node = root
    for ch in text:
        node = node.children.setdefault(ch, TrieNode())
    if node.terminal is None:
        node.terminal = []
    node.terminal.append((intent, dict(slots)))


@dataclass
class Grammar:
    name: str
    rules: dict[str, list[tuple[tuple[str, ...], dict, str]]]
    root: TrieNode = field(repr=False)
    # Inference-only rejection metadata (docs/INCOMPLETE-GRAMMAR-REJECTION.md):
    # designated whole-word prefixes such as "color" that may compete with
    # completed commands during decoding. Defaults to empty, preserving every
    # existing constructor/call site; grammars opt in explicitly (e.g.
    # OPTIONB_GRAMMAR via `dataclasses.replace` in vcm.optionb.grammar).
    # Never makes a prefix accepted: `accepts`/trie semantics are untouched.
    incomplete_prefixes: frozenset[str] = field(default_factory=frozenset)

    def accepts(self, text: str) -> list[tuple[str, dict]] | None:
        """Direct string acceptance check (no CTC/decoder involved): walk
        `text` (already-normalized, single-spaced, lowercase) through the
        trie and return the terminal (intent, slots) list if `text` is
        exactly a complete accepted phrase, else None."""
        node = self.root
        for ch in text:
            node = node.children.get(ch)
            if node is None:
                return None
        return node.terminal

    def all_phrases(self) -> Iterable[tuple[str, str, dict]]:
        """(text, intent, slots) for every alternative across every rule --
        for test enumeration/coverage, not used by the decoder."""
        for alternatives in self.rules.values():
            for words, slots_, intent in alternatives:
                yield " ".join(words), intent, slots_


def derive_incomplete_prefixes(grammar: Grammar) -> frozenset[str]:
    """Return non-command, proper whole-word prefixes of accepted phrases.

    Grammar-generic (moved here from `vcm.optionb.incomplete_prefix_grammar`
    per docs/INCOMPLETE-GRAMMAR-REJECTION.md's prefix-derivation spec). An
    accepted command is never returned, even when it is a strict prefix of
    another command. This preserves commands such as `"pause"` and `"time"`
    while identifying incomplete slot-bearing phrases such as `"color"`.
    Only whole-word prefixes belong in the set: character fragments such as
    `"colo"` do not."""
    accepted = {text for text, _, _ in grammar.all_phrases()}
    prefixes: set[str] = set()

    for phrase in accepted:
        words = phrase.split()
        for word_count in range(1, len(words)):
            prefix = " ".join(words[:word_count])
            if prefix not in accepted:
                prefixes.add(prefix)

    return frozenset(prefixes)


def compile_grammar(name: str, rules: dict[str, list[tuple[tuple[str, ...], dict, str]]]) -> Grammar:
    root = TrieNode()
    for alternatives in rules.values():
        for words, slots_, intent in alternatives:
            _insert(root, " ".join(words), intent, slots_)
    return Grammar(name=name, rules=rules, root=root)
