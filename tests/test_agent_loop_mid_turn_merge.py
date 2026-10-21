"""``merge_mid_turn``: what the provider is handed for a mid-turn arrival.

The loop appends one user message per arrival and labels them at the call seam,
so this pass is the only thing standing between "the reader corrected me while I
worked" and "the reader asked a second question". These pin the rules the seam
depends on: which messages it touches, which it must leave alone, and that
running it twice over the same list changes nothing.

``merge_adjacent_user`` runs at the same seam and folds two plain user messages
in a row, which a turn the model ended without a reply leaves behind.
"""

from __future__ import annotations

from raven.agent.loop._shared import _MID_TURN_HEADER, _MID_TURN_USER_KEY, merge_adjacent_user, merge_mid_turn


def _mid(text: str) -> dict:
    return {"role": "user", "content": text, _MID_TURN_USER_KEY: True}


def test_an_unmarked_list_is_handed_back_untouched() -> None:
    """The common path: every turn's call goes through here and almost none
    carries an arrival, so the same object comes back rather than a copy."""
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    assert merge_mid_turn(messages) is messages


def test_the_merged_message_keeps_the_mark() -> None:
    """A turn that is thrown away and run again seeds itself from the question
    plus whatever carries this key -- so dropping it here loses the correction
    on exactly the re-run that was meant to act on it."""
    [merged] = merge_mid_turn([_mid("only the last quarter")])

    assert merged[_MID_TURN_USER_KEY] is True
    assert merged["role"] == "user"
    assert merged["content"] == f"{_MID_TURN_HEADER}\n\nonly the last quarter"


def test_a_second_pass_adds_no_second_header() -> None:
    """The merged message keeps the mark, so it is a candidate again; the label
    is a statement about the message, and stating it twice states it wrong."""
    once = merge_mid_turn([_mid("skip 2023"), _mid("Q4 only")])

    assert merge_mid_turn(once) == once


def test_only_adjacent_arrivals_fold_together() -> None:
    """The work between two gaps is what the second correction is about."""
    merged = merge_mid_turn(
        [
            {"role": "user", "content": "summarise the report"},
            _mid("skip 2023"),
            {"role": "assistant", "content": "reading it"},
            _mid("Q4 only"),
        ]
    )

    assert [m["content"] for m in merged] == [
        "summarise the report",
        f"{_MID_TURN_HEADER}\n\nskip 2023",
        "reading it",
        f"{_MID_TURN_HEADER}\n\nQ4 only",
    ]


def test_a_stored_mid_turn_entry_is_never_labelled_again() -> None:
    """Replayed history carries the plain ``mid_turn`` of an entry already
    saved. Only the private spelling means "this arrived during the turn now
    running"; labelling the stored one would put the header on a question the
    model answered turns ago."""
    messages = [{"role": "user", "content": "only the last quarter", "mid_turn": True}]

    assert merge_mid_turn(messages) is messages


def test_two_user_messages_in_a_row_are_sent_as_one() -> None:
    """A turn the model ended without a reply leaves its question with nothing
    after it, so the next question lands right behind it. Some providers refuse
    two user messages in a row; the history keeps both, the payload folds them."""
    messages = [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]

    assert merge_adjacent_user(messages) == [{"role": "user", "content": "first\n\nsecond"}]


def test_a_body_in_parts_keeps_its_parts_when_folded() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [{"role": "user", "content": "first"}, {"role": "user", "content": [image]}]

    assert merge_adjacent_user(messages) == [{"role": "user", "content": [{"type": "text", "text": "first"}, image]}]


def test_a_list_with_no_two_user_messages_in_a_row_is_handed_back_untouched() -> None:
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    assert merge_adjacent_user(messages) is messages


def test_a_message_the_loop_marked_is_not_folded() -> None:
    """A mid-turn arrival rides its own message under its own header; the fold
    takes only plain messages and leaves the loop's own marks where they are."""
    messages = [{"role": "user", "content": "summarise the report"}, _mid("skip 2023")]

    assert merge_adjacent_user(messages) is messages
