import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_tree, exponentially_weighted_fetch


@pytest.fixture
def episodes():
    return [
        "Planning: Held an in-depth meeting with the entire team to brainstorm requirements for a new user registration API endpoint...",
        "Setup: Cloned the project repository from GitHub and verified branch integrity to ensure a clean starting point before any development...",
        "Coding: Created a dedicated file 'user_registration.py' in the 'api' directory to house the new registration logic...",
        "Review: Opened a detailed pull request that included inline code comments, step-by-step walkthroughs, and links to associated documentation...",
        "Testing: Deployed the new changes to the staging environment after confirming that all associated configuration files were properly updated...",
        "Documentation: Updated the API documentation with full, detailed descriptions about the new registration endpoint and its usage...",
        "Deployment: Rolled out the API changes to the production environment following strict best practices that ensured zero downtime...",
        "Maintenance: Performed follow-up optimizations on the API endpoint to reduce latency and remove redundant code...",
    ]


def _count_leaves(node) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)


def test_build_tree_structure(episodes):
    root = build_tree(episodes)

    assert not root.is_leaf
    assert root.span == len(episodes)
    assert _count_leaves(root) == len(episodes)


def test_tree_shape_matches_example(episodes):
    """
    Assert the balanced shape shown in the reference diagram:
    - Root spans E1-E8
    - Left subtree spans E1-E4 (with grandchildren E1-E2 and E3-E4)
    - Right subtree spans E5-E8 (with grandchildren E5-E6 and E7-E8)
    - Leaves preserve episode order and text
    """
    root = build_tree(episodes)

    # Root spans all
    assert (root.start_idx, root.end_idx) == (0, 7)
    assert not root.is_leaf

    # Level 1
    left = root.left
    right = root.right
    assert left and right
    assert (left.start_idx, left.end_idx) == (0, 3)
    assert (right.start_idx, right.end_idx) == (4, 7)

    # Level 2
    a1, a2 = left.left, left.right
    b1, b2 = right.left, right.right
    assert a1 and a2 and b1 and b2
    assert (a1.start_idx, a1.end_idx) == (0, 1)
    assert (a2.start_idx, a2.end_idx) == (2, 3)
    assert (b1.start_idx, b1.end_idx) == (4, 5)
    assert (b2.start_idx, b2.end_idx) == (6, 7)

    # Leaves E1..E8 in order
    leaves = [
        a1.left,
        a1.right,
        a2.left,
        a2.right,
        b1.left,
        b1.right,
        b2.left,
        b2.right,
    ]
    assert all(leaf and leaf.is_leaf for leaf in leaves)
    texts = [leaf.summary for leaf in leaves]  # summary holds full text for leaves
    assert texts == episodes


def test_fetch_frontier_prefers_recent_detail(episodes):
    root = build_tree(episodes)
    frontier = exponentially_weighted_fetch(root, num_nodes=4)

    labels = [node.label for node in frontier]
    summaries = [node.summary for node in frontier]
    # Older history is summarized into a single early-phase node; recent events are expanded.
    assert labels == ["E1-E5", "E6", "E7", "E8"]
    assert (
        summaries[0] == "Early project phases: planning, setup, coding, review, testing"
    )
    assert summaries[1].startswith("Documentation: Updated the API documentation")
    assert summaries[2].startswith("Deployment: Rolled out the API changes")
    assert summaries[3].startswith("Maintenance: Performed follow-up optimizations")


def test_fetch_small_frontier(episodes):
    root = build_tree(episodes)
    frontier = exponentially_weighted_fetch(root, num_nodes=2)

    labels = [node.label for node in frontier]
    assert labels == ["E1-E7", "E8"]


def test_fetch_zero_nodes(episodes):
    root = build_tree(episodes)
    assert exponentially_weighted_fetch(root, num_nodes=0) == []
