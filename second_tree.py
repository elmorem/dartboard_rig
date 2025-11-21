from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

PREPOSITIONS = {"into", "to", "from", "with", "in", "on", "for", "of"}
VERB_HINTS = {"translate", "create", "compile", "test", "define", "build"}


@dataclass
class Node:
    """A single word in the Engrish Parse Summary Tree."""

    word: str
    children: List["Node"] = field(default_factory=list)

    def add_child(self, child: "Node") -> None:
        self.children.append(child)


def _is_identifier(token: str) -> bool:
    """Heuristic: uppercase or alphanumeric tokens behave like identifiers (X, M, obj1)."""
    return token and token[0].isupper()


def _choose_head(tokens: List[str]) -> int:
    """
    Pick the most general head word from a noun phrase.
    Prefer the rightmost non-identifier token; fall back to the last token.
    """
    for idx in range(len(tokens) - 1, -1, -1):
        if not _is_identifier(tokens[idx]):
            return idx
    return len(tokens) - 1


def build_noun_phrase(tokens: List[str]) -> Optional[Node]:
    """Construct a noun-phrase subtree where specificity increases with depth."""
    if not tokens:
        return None

    head_idx = _choose_head(tokens)
    head_word = tokens[head_idx]
    head = Node(head_word)

    # Modifiers before the head become a chain, most general closest to head.
    modifiers = tokens[:head_idx]
    current = head
    for word in reversed(modifiers):
        node = Node(word)
        current.add_child(node)
        current = node

    # Remaining tokens after the head attach as direct children (more specific items).
    for word in tokens[head_idx + 1 :]:
        head.add_child(Node(word))

    return head


def parse_engrish_line(line: str) -> Node:
    """
    Parse a single Engrish pseudocode line into a Parse Summary Tree (PST).

    - If the line appears to be a noun phrase only, the noun phrase root becomes the tree root.
    - Otherwise the top-level verb becomes the root.
    - Noun phrases attach as children, with most general terms nearer the root.
    - Prepositional phrases attach under their preposition nodes.
    """
    tokens = [t for t in line.strip().split() if t]
    if not tokens:
        raise ValueError("Cannot parse empty line")

    # Noun-phrase-only line (no prepositions, first token not a known verb).
    if (
        len(tokens) > 1
        and not any(tok.lower() in PREPOSITIONS for tok in tokens)
        and tokens[0].lower() not in VERB_HINTS
    ):
        noun_root = build_noun_phrase(tokens)
        if noun_root is None:
            raise ValueError("Unable to parse noun phrase")
        return noun_root

    root = Node(tokens[0])
    rest = tokens[1:]

    def flush_phrase(chunk: List[str]) -> None:
        if chunk:
            np_node = build_noun_phrase(chunk)
            if np_node:
                root.add_child(np_node)

    current_chunk: List[str] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.lower() in PREPOSITIONS:
            flush_phrase(current_chunk)
            current_chunk = []
            prep = Node(token)
            # Collect tokens until next preposition or end.
            i += 1
            phrase: List[str] = []
            while i < len(rest) and rest[i].lower() not in PREPOSITIONS:
                phrase.append(rest[i])
                i += 1
            np_node = build_noun_phrase(phrase)
            if np_node:
                prep.add_child(np_node)
            root.add_child(prep)
            continue
        current_chunk.append(token)
        i += 1

    flush_phrase(current_chunk)
    return root


def pretty(node: Node, prefix: str = "", is_last: bool = True) -> str:
    """Utility for debugging: ASCII representation of the PST."""
    connector = "└── " if is_last else "├── "
    lines = [f"{prefix}{connector}{node.word}"]
    children = node.children
    for idx, child in enumerate(children):
        last = idx == len(children) - 1
        extension = "    " if is_last else "│   "
        lines.append(pretty(child, prefix + extension, last))
    return "\n".join(lines)


if __name__ == "__main__":
    sample = "brew wyrmtongue elixir vial V with obsidian dust"
    tree = parse_engrish_line(sample)
    print(f"Parsed PST for: {sample}")
    print(pretty(tree))
