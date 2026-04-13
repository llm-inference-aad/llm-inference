"""Tests for sub-page section splitting in PageIndex text assignment."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from src.pageindex.pageindex.utils import (
    find_title_offset,
    compute_section_offsets,
    _assign_text_single,
    add_node_text_deduped,
)


# ---------------------------------------------------------------------------
# find_title_offset
# ---------------------------------------------------------------------------

class TestFindTitleOffset:
    def test_exact_match(self):
        page = "Some preamble text. Introduction to the Topic. More text follows."
        assert find_title_offset(page, "Introduction to the Topic") == page.index("Introduction")

    def test_whitespace_variation(self):
        page = "Some text. Introduction   to\nthe   Topic. More text."
        offset = find_title_offset(page, "Introduction to the Topic")
        assert offset is not None
        assert page[offset:].startswith("Introduction")

    def test_case_insensitivity(self):
        page = "Preamble. INTRODUCTION TO THE TOPIC. Body."
        offset = find_title_offset(page, "Introduction to the Topic")
        assert offset is not None
        assert page[offset:].upper().startswith("INTRODUCTION")

    def test_special_characters(self):
        page = "Prior text. Section 3.1 (Overview) details follow."
        offset = find_title_offset(page, "Section 3.1 (Overview)")
        assert offset is not None
        assert page[offset:].startswith("Section 3.1")

    def test_not_found(self):
        page = "Some completely unrelated page content here."
        assert find_title_offset(page, "Nonexistent Section") is None

    def test_empty_title(self):
        assert find_title_offset("any page text", "") is None

    def test_whitespace_only_title(self):
        assert find_title_offset("any page text", "   ") is None

    def test_single_word_title(self):
        page = "Preamble. Methodology. Body text."
        offset = find_title_offset(page, "Methodology")
        assert offset == page.index("Methodology")

    def test_intra_word_whitespace(self):
        """PDF extraction artifact: 'REDUCING' becomes 'R EDUCING'."""
        page = "Prior text. IV. R EDUCING OVERFITTING  \nBody text."
        offset = find_title_offset(page, "IV. REDUCING OVERFITTING")
        assert offset is not None
        assert page[offset:].startswith("IV.")


# ---------------------------------------------------------------------------
# compute_section_offsets
# ---------------------------------------------------------------------------

def _make_page_list(pages):
    """Helper: convert a list of page-text strings to (text, token_count) tuples."""
    return [(p, len(p.split())) for p in pages]


class TestComputeSectionOffsets:
    """Tests for compute_all_section_offsets (via the compute_section_offsets shim).

    The current implementation annotates nodes with _split_combined,
    _split_start, and _split_end (character-level boundaries in the
    concatenated document text).
    """

    def test_shared_page_annotated(self):
        pages = _make_page_list([
            "Page 1 content.",
            "End of Section Alpha. Section B starts here. Rest of page 2.",
        ])
        siblings = [
            {"title": "Section Alpha", "start_index": 1, "end_index": 2},
            {"title": "Section B", "start_index": 2, "end_index": 2},
        ]
        compute_section_offsets(siblings, pages)
        # Both nodes should get split annotations
        assert "_split_start" in siblings[0]
        assert "_split_start" in siblings[1]
        # Section Alpha ends where Section B starts
        assert siblings[0]["_split_end"] == siblings[1]["_split_start"]

    def test_no_shared_page(self):
        pages = _make_page_list(["Section Alpha.", "Section Beta.", "Page 3."])
        siblings = [
            {"title": "Section Alpha", "start_index": 1, "end_index": 1},
            {"title": "Section Beta", "start_index": 2, "end_index": 3},
        ]
        compute_section_offsets(siblings, pages)
        # Both should still get split annotations (global title search)
        assert "_split_start" in siblings[0]
        assert "_split_start" in siblings[1]
        assert siblings[0]["_split_end"] == siblings[1]["_split_start"]

    def test_three_siblings_same_page(self):
        page_text = "Part A content. Part B content. Part C content."
        pages = _make_page_list([page_text])
        siblings = [
            {"title": "Part A", "start_index": 1, "end_index": 1},
            {"title": "Part B", "start_index": 1, "end_index": 1},
            {"title": "Part C", "start_index": 1, "end_index": 1},
        ]
        compute_section_offsets(siblings, pages)
        # A-B boundary
        assert siblings[0]["_split_end"] == find_title_offset(page_text, "Part B")
        assert siblings[1]["_split_start"] == find_title_offset(page_text, "Part B")
        # B-C boundary
        assert siblings[1]["_split_end"] == find_title_offset(page_text, "Part C")
        assert siblings[2]["_split_start"] == find_title_offset(page_text, "Part C")

    def test_title_not_found_no_annotation(self):
        pages = _make_page_list(["Page 1 only has unrelated text."])
        siblings = [
            {"title": "Phantom Section", "start_index": 1, "end_index": 1},
            {"title": "Nonexistent", "start_index": 1, "end_index": 1},
        ]
        compute_section_offsets(siblings, pages)
        # Titles not found: nodes get page-range fallback via _split_start/_split_end
        # but _split_combined is still set
        assert "_split_combined" in siblings[0]
        assert "_split_combined" in siblings[1]

    def test_nested_children(self):
        pages = _make_page_list([
            "Intro. Child Alpha text. Child Beta text.",
        ])
        parent = {
            "title": "Parent",
            "start_index": 1,
            "end_index": 1,
            "nodes": [
                {"title": "Child Alpha", "start_index": 1, "end_index": 1},
                {"title": "Child Beta", "start_index": 1, "end_index": 1},
            ],
        }
        compute_section_offsets(parent, pages)
        children = parent["nodes"]
        assert children[0]["_split_end"] == find_title_offset(pages[0][0], "Child Beta")
        assert children[1]["_split_start"] == find_title_offset(pages[0][0], "Child Beta")


# ---------------------------------------------------------------------------
# _assign_text_single  (integration with offsets)
# ---------------------------------------------------------------------------

class TestAssignTextSingleWithOffsets:
    """Tests for _assign_text_single using the current _split_* annotations."""

    def test_shared_page_split(self):
        """Sibling A gets text before the title, sibling B gets text from the title onward."""
        page_text = "Content of Section Alpha. Section B Title. Content of B."
        combined = "First page." + page_text
        alpha_start = find_title_offset(combined, "Section Alpha")
        b_start = find_title_offset(combined, "Section B Title")

        node_a = {
            "title": "Section Alpha", "start_index": 1, "end_index": 2,
            "_split_combined": combined, "_split_start": alpha_start, "_split_end": b_start,
        }
        node_b = {
            "title": "Section B Title", "start_index": 2, "end_index": 2,
            "_split_combined": combined, "_split_start": b_start, "_split_end": len(combined),
        }

        pages = _make_page_list(["First page.", page_text])
        seen = set()
        _assign_text_single(node_a, pages, seen_pages=seen)
        _assign_text_single(node_b, pages, seen_pages=seen)

        assert node_a["text"] == combined[alpha_start:b_start]
        assert node_b["text"] == combined[b_start:]

    def test_same_page_section(self):
        """A tiny section that starts and ends on the same page."""
        page_text = "Before. Target Section. After Section."
        combined = page_text
        start_off = find_title_offset(combined, "Target Section")
        end_off = find_title_offset(combined, "After Section")

        node = {
            "title": "Target Section", "start_index": 1, "end_index": 1,
            "_split_combined": combined, "_split_start": start_off, "_split_end": end_off,
        }

        seen = set()
        seen.add(1)
        _assign_text_single(node, _make_page_list([page_text]), seen_pages=seen)

        assert node["text"] == combined[start_off:end_off]

    def test_no_offsets_backward_compat(self):
        """Without split annotations, falls back to page-level text concatenation."""
        pages = _make_page_list(["Page 1.", "Page 2.", "Page 3."])
        node = {"title": "Section Alpha", "start_index": 1, "end_index": 3}

        _assign_text_single(node, pages)

        assert node["text"] == "Page 1.Page 2.Page 3."

    def test_offsets_popped_after_use(self):
        """Split annotations should not persist in the node dict."""
        page_text = "Before. Title Section. After."
        combined = page_text
        offset = find_title_offset(combined, "Title Section")
        node = {
            "title": "Title Section", "start_index": 1, "end_index": 1,
            "_split_combined": combined, "_split_start": offset, "_split_end": len(combined),
        }
        seen = set()
        seen.add(1)
        _assign_text_single(node, _make_page_list([page_text]), seen_pages=seen)

        assert "_split_combined" not in node
        assert "_split_start" not in node
        assert "_split_end" not in node


# ---------------------------------------------------------------------------
# add_node_text_deduped end-to-end with compute_section_offsets
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_no_text_loss(self):
        """All page text should be covered by the tree's leaf nodes."""
        page1 = "Section Alpha starts here on page one."
        page2 = "End of alpha. Section Beta Title. Start of section beta."
        page3 = "Page three is all section beta."
        pages = _make_page_list([page1, page2, page3])

        tree = [
            {"title": "Section Alpha", "start_index": 1, "end_index": 2},
            {"title": "Section Beta Title", "start_index": 2, "end_index": 3},
        ]

        compute_section_offsets(tree, pages)
        add_node_text_deduped(tree, pages)

        combined = tree[0]["text"] + tree[1]["text"]
        full_doc = page1 + page2 + page3
        assert combined == full_doc

    def test_three_sections_one_page(self):
        """Three sections on one page each get their slice."""
        page = "Section Alpha content. Section Beta content. Section Gamma content."
        pages = _make_page_list([page])

        tree = [
            {"title": "Section Alpha", "start_index": 1, "end_index": 1},
            {"title": "Section Beta", "start_index": 1, "end_index": 1},
            {"title": "Section Gamma", "start_index": 1, "end_index": 1},
        ]

        compute_section_offsets(tree, pages)
        add_node_text_deduped(tree, pages)

        b_start = find_title_offset(page, "Section Beta")
        c_start = find_title_offset(page, "Section Gamma")

        assert tree[0]["text"] == page[:b_start]
        assert tree[1]["text"] == page[b_start:c_start]
        assert tree[2]["text"] == page[c_start:]
        # No text lost
        assert tree[0]["text"] + tree[1]["text"] + tree[2]["text"] == page


# ---------------------------------------------------------------------------
# find_title_offset — short title hardening (Change 6)
# ---------------------------------------------------------------------------

class TestFindTitleOffsetShortTitles:
    def test_short_title_no_false_positive_in_body(self):
        """A single-char title like 'A' should NOT match a random 'A' in body text."""
        page = "This is a paragraph about A topic and some more text."
        assert find_title_offset(page, "A") is None

    def test_short_title_on_own_line(self):
        """A short title on its own line should match."""
        page = "Some preamble.\nA\nBody text follows."
        offset = find_title_offset(page, "A")
        assert offset is not None
        assert page[offset] == "A"

    def test_short_title_at_start(self):
        """A short title at the very start of text on its own line should match."""
        page = "B\nSome body text."
        offset = find_title_offset(page, "B")
        assert offset is not None
        assert page[offset] == "B"

    def test_intra_char_disabled_for_short_single_word(self):
        """Single short word (< 6 chars) should NOT use intra-char fallback.
        'M e t h' should not match 'Math' via intra-char pattern."""
        page = "Some text about M e t h o d s here."
        # "Math" is 4 chars — should not match via intra-char whitespace
        assert find_title_offset(page, "Math") is None

    def test_intra_char_works_for_long_word(self):
        """Single long word (>= 6 chars) should still use intra-char fallback."""
        page = "Prior text. R EDUCING overfitting stuff."
        offset = find_title_offset(page, "REDUCING")
        assert offset is not None

    def test_multi_word_title_still_works(self):
        """Multi-word titles should work normally."""
        page = "Prior text. Section Three Overview. Body."
        offset = find_title_offset(page, "Section Three Overview")
        assert offset is not None
        assert page[offset:].startswith("Section Three")


# ---------------------------------------------------------------------------
# validate_toc (Change 2)
# ---------------------------------------------------------------------------

# Import validate_toc — it lives in page_index module
import importlib
try:
    from src.pageindex.pageindex.page_index import validate_toc
except ImportError:
    validate_toc = None


@pytest.mark.skipif(validate_toc is None, reason="validate_toc not importable")
class TestValidateToc:
    def test_removes_duplicates(self):
        """Duplicate titles (case-insensitive) should be deduped, keeping first."""
        toc = [
            {"title": "Introduction", "physical_index": "<physical_index_1>", "structure": "1"},
            {"title": "introduction", "physical_index": "<physical_index_3>", "structure": "2"},
            {"title": "Method", "physical_index": "<physical_index_2>", "structure": "3"},
        ]
        source = "Introduction\nSome text.\nMethod\nMore text."
        result = validate_toc(toc, source)
        titles = [r["title"] for r in result]
        assert titles == ["Introduction", "Method"]

    def test_removes_unfound_titles(self):
        """Titles not found in source text should be dropped."""
        toc = [
            {"title": "Introduction", "physical_index": "<physical_index_1>", "structure": "1"},
            {"title": "Nonexistent Section", "physical_index": "<physical_index_2>", "structure": "2"},
        ]
        source = "Introduction\nBody text about something."
        result = validate_toc(toc, source)
        assert len(result) == 1
        assert result[0]["title"] == "Introduction"

    def test_preserves_valid_entries(self):
        """All valid entries should survive validation."""
        toc = [
            {"title": "Abstract", "physical_index": "<physical_index_1>", "structure": "1"},
            {"title": "Method", "physical_index": "<physical_index_2>", "structure": "2"},
            {"title": "Results", "physical_index": "<physical_index_3>", "structure": "3"},
        ]
        source = "Abstract\nSome text.\nMethod\nMore text.\nResults\nFinal text."
        result = validate_toc(toc, source)
        assert len(result) == 3

    def test_removes_none_physical_index(self):
        """Entries with physical_index=None should be removed."""
        toc = [
            {"title": "Introduction", "physical_index": "<physical_index_1>", "structure": "1"},
            {"title": "Method", "physical_index": None, "structure": "2"},
        ]
        source = "Introduction\nSome text.\nMethod\nMore text."
        result = validate_toc(toc, source)
        assert len(result) == 1
        assert result[0]["title"] == "Introduction"

    def test_empty_input(self):
        """Empty TOC should return empty list."""
        assert validate_toc([], "some text") == []

    def test_strips_physical_index_tags(self):
        """Physical index tags in source should not interfere with title matching."""
        toc = [
            {"title": "Method", "physical_index": "<physical_index_1>", "structure": "1"},
        ]
        source = "<physical_index_1>\nMethod\nBody text.\n<physical_index_1>"
        result = validate_toc(toc, source)
        assert len(result) == 1
