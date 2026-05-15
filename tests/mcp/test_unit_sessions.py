"""Unit tests for ``spikelab.mcp.sessions`` — kind tagging and decode.

The load-bearing case here is the SDK whitespace-normalization quirk: when the
Claude Agent SDK persists a first prompt, it collapses the ``]\\n`` suffix
we write into ``] ``. The decoder must accept either form so
``list_spikelab_tasks(kind=…)`` filters correctly after an SDK round-trip.
"""

from __future__ import annotations

import pytest

from spikelab.mcp.sessions import (
    TASK_KIND_PREFIX,
    TASK_KIND_SUFFIX,
    _decode_kind,
    encode_kind,
)


class TestEncodeKind:
    def test_prefix_and_suffix_present(self):
        out = encode_kind("sorter", "hello")
        assert out.startswith(TASK_KIND_PREFIX + "sorter" + TASK_KIND_SUFFIX)
        assert out.endswith("hello")

    def test_prompt_body_preserved(self):
        body = "line 1\nline 2 with [brackets] and ]closer"
        out = encode_kind("educator", body)
        assert out.endswith(body)

    @pytest.mark.parametrize(
        "kind", ["educator", "sorter", "analyzer", "developer", "map_updater"]
    )
    def test_round_trip_with_native_suffix(self, kind):
        body = "round-trip body"
        encoded = encode_kind(kind, body)
        decoded_kind, decoded_body = _decode_kind(encoded)
        assert decoded_kind == kind
        assert decoded_body == body


class TestDecodeKind:
    def test_unknown_when_no_prefix(self):
        kind, body = _decode_kind("regular prompt")
        assert kind == "unknown"
        assert body == "regular prompt"

    def test_unknown_when_empty(self):
        kind, body = _decode_kind("")
        assert kind == "unknown"
        assert body == ""

    def test_unknown_when_prefix_unterminated(self):
        # Prefix starts but no closer — defensive fallback to unknown.
        raw = TASK_KIND_PREFIX + "garbage no closer"
        kind, body = _decode_kind(raw)
        assert kind == "unknown"
        assert body == raw

    # --- The SDK-normalization cases — most important ---------------------
    # The Claude Agent SDK stores first_prompt with whitespace normalized.
    # An encoded ``[oc-spikelab-mcp:kind=sorter]\nfoo`` becomes
    # ``[oc-spikelab-mcp:kind=sorter] foo`` by the time we read it back via
    # ``list_sessions``. Both forms must decode the same way.

    def test_decodes_newline_form(self):
        raw = "[oc-spikelab-mcp:kind=sorter]\nfoo bar"
        assert _decode_kind(raw) == ("sorter", "foo bar")

    def test_decodes_space_form_after_sdk_normalization(self):
        raw = "[oc-spikelab-mcp:kind=sorter] foo bar"
        assert _decode_kind(raw) == ("sorter", "foo bar")

    def test_decodes_with_tab(self):
        raw = "[oc-spikelab-mcp:kind=educator]\tfoo"
        assert _decode_kind(raw) == ("educator", "foo")

    def test_decodes_with_no_whitespace_after_closer(self):
        # Defensive: agent SDK could in principle strip whitespace entirely.
        raw = "[oc-spikelab-mcp:kind=analyzer]foo"
        assert _decode_kind(raw) == ("analyzer", "foo")

    def test_only_one_whitespace_char_consumed(self):
        # Two trailing spaces — only the first is part of the suffix, the
        # second is real content.
        raw = "[oc-spikelab-mcp:kind=sorter]  foo"
        assert _decode_kind(raw) == ("sorter", " foo")

    def test_kind_can_contain_underscore(self):
        raw = "[oc-spikelab-mcp:kind=map_updater] x"
        kind, _ = _decode_kind(raw)
        assert kind == "map_updater"

    def test_developer_kind(self):
        # Sanity for the kind that's new to spikelab (ephys doesn't have it).
        raw = "[oc-spikelab-mcp:kind=developer]\nintegrate this script"
        kind, body = _decode_kind(raw)
        assert kind == "developer"
        assert body == "integrate this script"
