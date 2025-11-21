from __future__ import annotations

"""
Binary summarization tree with an exponentially weighted fetch.

Leaf nodes hold full episode text. Internal nodes hold concise summaries of
their two children. The fetch function surfaces a frontier of nodes that
heavily compresses the distant past while revealing more detail for recent
episodes.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

EPISODES = [
    "Planning: Held an in-depth meeting with the entire team to brainstorm requirements for a new user registration API endpoint...",
    "Setup: Cloned the project repository from GitHub and verified branch integrity to ensure a clean starting point before any development...",
    "Coding: Created a dedicated file 'user_registration.py' in the 'api' directory to house the new registration logic...",
    "Review: Opened a detailed pull request that included inline code comments, step-by-step walkthroughs, and links to associated documentation...",
    "Testing: Deployed the new changes to the staging environment after confirming that all associated configuration files were properly updated...",
    "Documentation: Updated the API documentation with full, detailed descriptions about the new registration endpoint and its usage...",
    "Deployment: Rolled out the API changes to the production environment following strict best practices that ensured zero downtime...",
    "Maintenance: Performed follow-up optimizations on the API endpoint to reduce latency and remove redundant code...",
]


def trimmed(text: str, max_words: int = 12) -> str:
    """Return the first few words of a string, adding ellipsis when truncated."""
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def heading(text: str) -> str:
    """Derive a short topic-style heading from an episode."""
    if ":" in text:
        return text.split(":", 1)[0].strip().lower()
    return text.split(".", 1)[0].strip().split()[0].lower()


def summarize_headings(headings: List[str]) -> str:
    """
    Summarize a list of headings. Larger spans get an 'Early project phases'
    prefix to reflect heavy compression of the past.
    """
    if len(headings) >= 4:
        return "Early project phases: " + ", ".join(headings)
    if len(headings) == 3:
        return ", ".join(headings)
    if len(headings) == 2:
        return " | ".join(headings)
    return headings[0]


@dataclass
class Node:
    summary: str
    headings: List[str]
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    start_idx: int = 0
    end_idx: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def span(self) -> int:
        return self.end_idx - self.start_idx + 1

    @property
    def label(self) -> str:
        if self.span == 1:
            return f"E{self.start_idx + 1}"
        return f"E{self.start_idx + 1}-E{self.end_idx + 1}"


def merge_nodes(left: Node, right: Node) -> Node:
    headings = left.headings + right.headings
    return Node(
        summary=summarize_headings(headings),
        headings=headings,
        left=left,
        right=right,
        start_idx=left.start_idx,
        end_idx=right.end_idx,
    )


def build_tree(episodes: Iterable[str]) -> Node:
    """
    Build a binary tree where leaves are episodes and internal nodes hold
    summaries derived from their children's headings.
    """
    leaves = [
        Node(summary=episode, headings=[heading(episode)], start_idx=i, end_idx=i)
        for i, episode in enumerate(episodes)
    ]
    nodes = leaves
    while len(nodes) > 1:
        merged: List[Node] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else None
            if right is None:
                merged.append(left)
                continue
            merged.append(merge_nodes(left, right))
        nodes = merged
    return nodes[0]


def _recency_score(node: Node, bias: float) -> float:
    """
    Higher score means the node is more likely to remain detailed.
    Older, wider spans get the smallest scores, so they are merged first.
    """
    return (bias**node.end_idx) / node.span


def collect_leaves(node: Optional[Node]) -> List[Node]:
    if node is None:
        return []
    if node.is_leaf:
        return [node]
    return collect_leaves(node.left) + collect_leaves(node.right)


def exponentially_weighted_fetch(
    root: Node, num_nodes: int, recency_bias: float = 1.25
) -> List[Node]:
    """
    Return a frontier of `num_nodes` that compresses earlier history and shows
    more detail for recent episodes.

    Strategy: start from the leaf list (most detailed), then iteratively merge
    the oldest adjacent pair until the frontier size matches `num_nodes`.
    Merging is chosen by smallest recency score so older spans collapse first,
    giving an exponential-like tilt toward recency.
    """
    if num_nodes <= 0:
        return []

    leaves = collect_leaves(root)
    if num_nodes >= len(leaves):
        return leaves

    frontier: List[Node] = leaves.copy()

    def merge_index() -> int:
        scores = [
            _recency_score(frontier[i + 1], recency_bias)
            for i in range(len(frontier) - 1)
        ]
        return min(range(len(scores)), key=lambda idx: scores[idx])

    while len(frontier) > num_nodes:
        idx = merge_index()
        merged = merge_nodes(frontier[idx], frontier[idx + 1])
        frontier[idx : idx + 2] = [merged]

    return frontier


def print_tree(node: Optional[Node], prefix: str = "", is_tail: bool = True) -> None:
    """Pretty-print the tree for quick inspection."""
    if node is None:
        return
    connector = "└── " if is_tail else "├── "
    node_type = "LEAF" if node.is_leaf else "INTERNAL"
    print(f"{prefix}{connector}[{node_type}] {node.label}: {trimmed(node.summary)}")
    if not node.is_leaf:
        extension = "    " if is_tail else "│   "
        print_tree(node.right, prefix + extension, False)
        print_tree(node.left, prefix + extension, True)


def assert_example_shape(root: Node) -> None:
    """
    Runtime sanity check mirroring the test suite's expected shape for the
    eight sample episodes.
    """
    assert (root.start_idx, root.end_idx) == (0, 7)
    left = root.left
    right = root.right
    assert left and right
    assert (left.start_idx, left.end_idx) == (0, 3)
    assert (right.start_idx, right.end_idx) == (4, 7)
    a1, a2 = left.left, left.right
    b1, b2 = right.left, right.right
    assert a1 and a2 and b1 and b2
    assert (a1.start_idx, a1.end_idx) == (0, 1)
    assert (a2.start_idx, a2.end_idx) == (2, 3)
    assert (b1.start_idx, b1.end_idx) == (4, 5)
    assert (b2.start_idx, b2.end_idx) == (6, 7)


def demo() -> None:
    root = build_tree(EPISODES)
    assert_example_shape(root)

    print("Full tree:")
    print_tree(root)

    print("\nfetch(4) summaries:")
    nodes = exponentially_weighted_fetch(root, num_nodes=4)
    for node in nodes:
        print(f"- [{node.label}] {trimmed(node.summary)}")

    print("\nfetch(4) as list-like output:")
    print([node.summary for node in nodes])


if __name__ == "__main__":
    demo()
