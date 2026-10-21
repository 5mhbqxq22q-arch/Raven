"""How the agent loop reads a model reply, and the limits its retries spend.

A reply with no text and no tool call is read for one fact: whether the model
stopped while thinking (:func:`stopped_in_thinking`). One with nothing in it at
all is a model that said nothing: its turn ends without a reply and it is not
asked again, since a second ask cannot tell a chosen silence from a stall. One
that reasoned and never reached a body, or was cut at the output ceiling, spent
its budget thinking: it is asked again a rung at a time, high and then low
(:func:`lower_reasoning_effort`), and fails its turn once no rung is left.

The think-tag helpers decide what counts as text, because reasoning some
backends inline in ``content`` is not a reply. :class:`RecoveryLimits` carries
the waits the loop spends on a model call that came back an error, and
:class:`DraftGate` holds a streamed draft until a hook has decided to keep it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from raven.config import schema as config_schema
from raven.contracts.llm_provider import REASONING_EFFORT_LADDER, LLMResponse
from raven.providers.usage import token_count

# In-content thinking markers. Some models (Ollama, certain Qwen gateways) put
# the reasoning in ``content`` as <think>…</think> rather than in the structured
# reasoning_content field, so a content scan is required — checking only the
# structured fields would miss them.
# Matches an opening or a closing tag, with or without a vendor namespace
# prefix (``<mm:think>``). A turn cut off inside an inlined block arrives as
# a lone closing tag, so an opener-only pattern reads it as ordinary prose.
_THINK_TAG_RE = re.compile(r"</?(?:[a-z][\w.-]{0,15}:)?(?:think|thinking|reasoning)>", re.IGNORECASE)
_NS = r"(?:[a-z][\w.-]{0,15}:)?"
_THINK_BLOCK_RE = re.compile(
    rf"<{_NS}(think|thinking|reasoning)>[\s\S]*?</{_NS}\1>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecoveryLimits:
    """Per-turn retry budgets."""

    #: How long the loop waits, and how many times, when a model call comes back as
    #: a retryable error after the provider's own ladder gave up. The provider's
    #: ladder is seconds long -- right for a dropped connection, and too short for
    #: a gateway that serves error pages for a few minutes. One measured run had 62
    #: minutes and 23.8M tokens of work behind it when a 40-second outage ended it.
    llm_error_retry_delays: tuple[float, ...] = field(
        default_factory=lambda: config_schema.LLM_ERROR_RETRY_DELAYS_DEFAULT
    )
    #: Whether a streamed call that failed after it had already produced output is
    #: asked again. Off, the turn fails (N-TURNFAILED): a person watching the stream
    #: has seen the words and would see them twice. On, for an unattended agent whose
    #: client is a machine: one measured deck build had two hours behind it when a
    #: mid-stream "Network connection lost" ended the turn with nothing published.
    llm_retry_after_output: bool = False
    #: Base64 bytes of tool-shown pictures one request may carry before the standing
    #: image window collapses to the newest messages (``agents.defaults.
    #: imageWindowBudgetBytes``). Counted as the pictures travel, encoded, because
    #: that is what the request body and the gateway are charged for. 0 turns the
    #: standing pass off: pictures then leave only through the ladder that answers a
    #: size refusal.
    image_window_budget_bytes: int = 12_000_000


def limits_from_defaults(defaults: object) -> RecoveryLimits:
    """Build limits from an ``agents.defaults`` config object (duck-typed).

    Centralizes the config→RecoveryLimits mapping so the several AgentLoop
    construction sites don't each repeat the field plumbing.
    """
    return RecoveryLimits(
        llm_error_retry_delays=_ladder(getattr(defaults, "llm_error_retry_delays", None)),
        llm_retry_after_output=bool(getattr(defaults, "llm_retry_after_output", False)),
        image_window_budget_bytes=max(0, int(getattr(defaults, "image_window_budget_bytes", 12_000_000))),
    )


def _ladder(configured: object) -> tuple[float, ...]:
    """The outer retry ladder as configured, where an explicit `[]` means none.

    Only an absent setting takes the default: `[]` is the documented way to turn
    the ladder off, and `or` read it as absent and put the 105 seconds back.
    """
    if configured is None:
        return config_schema.LLM_ERROR_RETRY_DELAYS_DEFAULT
    return tuple(float(delay) for delay in configured)


def strip_think_blocks(text: str) -> str:
    """Remove paired think blocks. Debris is left in place for the check below.

    Matched by the same shape as ``_THINK_TAG_RE``: any of the three tag names,
    with or without a vendor namespace prefix. A single spelling here would let
    a complete ``<mm:think>...</mm:think>`` block through to the user in full,
    which is the one outcome stripping exists to prevent.

    The backreference keeps the two ends the same tag -- ``<think>x</thinking>``
    is a malformed pair, and deleting everything between two unrelated tags
    would take real content with it. The prefixes are not tied to each other:
    a backend that stamps one end and not the other still wrote one block.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def is_only_think_debris(text: str) -> bool:
    """True when the text is nothing but think tags once they are removed.

    The shape a turn cut off inside an inlined reasoning block arrives in: a
    lone closing tag with no opener, which pairs with nothing and so survives
    ``strip_think_blocks`` as an eleven-character string that reads like a real
    answer. Asked of the residue rather than of vendor spellings -- which
    prefix a backend picked is not knowable in advance.
    """
    return bool(text) and not _THINK_TAG_RE.sub("", text).strip()


def has_inline_thinking(content: str | None) -> bool:
    """True when raw content carries a <think>/<thinking>/<reasoning> marker."""
    return bool(content) and bool(_THINK_TAG_RE.search(content))


def has_thinking(response: LLMResponse) -> bool:
    """True when the response produced reasoning in any form (structured or inline)."""
    return bool(response.reasoning_content or response.thinking_blocks or has_inline_thinking(response.content))


OUTPUT_LIMIT_FAILURE = "The model reached its output limit before producing any reply."
STOPPED_THINKING_FAILURE = "The model stopped while still thinking, before producing any reply."


def stopped_in_thinking(response: LLMResponse) -> str | None:
    """Why a reply with no text and no tool call stopped while thinking, or None.

    None is a model that said nothing, and its turn ends without a reply. A
    reply cut at the output ceiling, or one that reasoned and never reached a
    body, stopped while thinking: every measured empty reply behind this module
    was reasoning that filled the budget. The reason is what the turn fails with
    once no lower rung is left to ask at. Reasoning counts in whatever form the
    provider reports it, the billed token count included, because Codex
    returns none of the text; a count is read the way the usage normalizer reads
    one, so a value that is not a token count is no sign of thinking.
    """
    if response.truncated:
        return OUTPUT_LIMIT_FAILURE
    if has_thinking(response) or (token_count((response.usage or {}).get("reasoning_tokens")) or 0) > 0:
        return STOPPED_THINKING_FAILURE
    return None


#: The efforts a reply that stopped while thinking is asked again at, strongest
#: first: the two levels nearly every reasoning backend takes as sent. The
#: ladder's floor, minimal, is not one of them -- the Codex backend refuses it
#: on every model with an HTTP 400.
_STEP_DOWN_RUNGS = ("high", "low")


def lower_reasoning_effort(current: str | None) -> str | None:
    """The rung to ask again at after a call that stopped while thinking at
    ``current``, or None when no rung is below it.

    The strongest of ``_STEP_DOWN_RUNGS`` below ``current`` on
    ``REASONING_EFFORT_LADDER``, so a step down from a named effort never asks
    for more thinking. ``None`` in means the turn named no effort and the
    backend applied its own default, which the loop cannot see, so the
    step-down walks every rung from the top, as it would from max. Where that
    default sits below high, the first step asks for more thinking rather than
    less. An effort the ladder does not carry has no rung below it.
    """
    if current is None:
        return _STEP_DOWN_RUNGS[0]
    try:
        here = REASONING_EFFORT_LADDER.index(current.strip().lower())
    except (AttributeError, ValueError):
        return None
    below = [rung for rung in _STEP_DOWN_RUNGS if REASONING_EFFORT_LADDER.index(rung) > here]
    return below[0] if below else None


class DraftGate:
    """Holds a response's deltas until something has decided to keep it.

    The decision is not in the text: a hook reading state the response does not
    carry can send the whole thing back (`HookDecision.rollback`), and a
    rollback pops history the stream has already left. A draft the loop
    discarded and a reader who saw it are not the same turn -- on the ACP lane
    `_TurnCollector` concatenates every chunk, so the rejected draft and its
    replacement arrive as one answer.
    """

    def __init__(self, deliver) -> None:
        self._deliver = deliver
        self._held = ""
        self._open = False
        self._dropped = False

    async def __call__(self, delta: str) -> None:
        if self._dropped:
            return
        if self._open:
            await self._deliver(delta)
            return
        self._held += delta

    async def release(self) -> None:
        """Deliver what is held, then let the rest through untouched."""
        if self._dropped or self._open:
            return
        self._open = True
        text = self._held
        self._held = ""
        if text:
            await self._deliver(text)

    def discard(self) -> None:
        """Drop the held draft. Nothing kept it, so nobody reads it."""
        self._held = ""
        self._dropped = True
