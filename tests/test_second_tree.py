import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from second_tree import Node, parse_engrish_line


def test_translate_example():
    root = parse_engrish_line("translate Engrish code X into python")
    assert root.word == "translate"
    assert [c.word for c in root.children] == ["code", "into"]

    code = root.children[0]
    assert [c.word for c in code.children] == ["Engrish", "X"]

    into = root.children[1]
    assert into.word == "into"
    assert len(into.children) == 1
    assert into.children[0].word == "python"


def test_noun_phrase_only():
    root = parse_engrish_line("pyngrish function definition")
    assert root.word == "definition"
    assert root.children[0].word == "function"
    assert root.children[0].children[0].word == "pyngrish"


def test_compile_example():
    root = parse_engrish_line("compile advanced Engrish module M into python")
    assert root.word == "compile"
    module = root.children[0]
    assert module.word == "module"
    # Modifier specificity increases with depth: module -> Engrish -> advanced
    assert module.children[0].word == "Engrish"
    assert module.children[0].children[0].word == "advanced"
    assert module.children[1].word == "M"
    assert root.children[1].word == "into"
    assert root.children[1].children[0].word == "python"


def test_compound_noun_example():
    root = parse_engrish_line("test new Engrish function F")
    assert root.word == "test"
    func = root.children[0]
    assert func.word == "function"
    assert func.children[0].word == "Engrish"
    assert func.children[0].children[0].word == "new"
    assert func.children[1].word == "F"


def test_empty_line_error():
    with pytest.raises(ValueError):
        parse_engrish_line("  ")


def test_orchestrate_arcane_widget():
    root = parse_engrish_line("orchestrate arcane widget Z into starlight")
    assert root.word == "orchestrate"
    assert [c.word for c in root.children] == ["widget", "into"]

    widget = root.children[0]
    assert widget.word == "widget"
    # Modifiers increase in specificity downward: widget -> arcane
    assert widget.children[0].word == "arcane"
    # Identifier stays as a direct child of the head
    assert widget.children[1].word == "Z"

    into = root.children[1]
    assert into.word == "into"
    assert into.children[0].word == "starlight"


def test_brew_wyrmtongue_elixir():
    root = parse_engrish_line("brew wyrmtongue elixir vial V with obsidian dust")
    assert root.word == "brew"
    noun = root.children[0]
    assert noun.word == "vial"
    # Specificity chain inside noun phrase
    assert noun.children[0].word == "elixir"
    assert noun.children[0].children[0].word == "wyrmtongue"
    assert noun.children[1].word == "V"

    with_prep = root.children[1]
    assert with_prep.word == "with"
    dust_np = with_prep.children[0]
    assert dust_np.word == "dust"
    assert dust_np.children[0].word == "obsidian"
