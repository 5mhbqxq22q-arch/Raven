"""How the agent loop reads a reply with no text and no tool call.

A model that said nothing ends its turn without a reply, and is not asked again.
A reply that stopped while thinking -- reasoning with no body, or one cut at the
output ceiling -- is asked again with less thinking allowed, high then low, and
fails its turn once no rung is left. What counts as text, and so what can make a
reply empty, is decided by the think-tag helpers tested here too.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from raven.agent.hook import AgentHook, HookDecision
from raven.agent.loop import AgentLoop
from raven.agent.loop.bundles import ToolWiring, TurnPolicy
from raven.agent.loop.recovery import (
    RecoveryLimits,
    has_inline_thinking,
    has_thinking,
    limits_from_defaults,
    lower_reasoning_effort,
)
from raven.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from raven.providers.litellm_provider import LiteLLMProvider
from raven.spine.message import ChatType, Source
from raven.spine.turn import AnswerlessTurnError, Origin, TurnRequest


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _turn(text: str = "hi") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
        text=text,
    )


async def _run_answerless(agent: AgentLoop, text: str = "hi", **kwargs) -> str:
    """Run a turn that ends with no answer, and return the failure it reports."""
    with pytest.raises(AnswerlessTurnError) as failed:
        await agent._process_message(_turn(text), session_key="s1", **kwargs)
    return str(failed.value)


def _make_agent(
    workspace: Path, provider: LLMProvider, limits: RecoveryLimits | None = None, *, max_iterations: int = 10
) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        workspace=workspace,
        model="stub",
        policy=TurnPolicy(max_iterations=max_iterations, empty_recovery=limits),
        tools=ToolWiring(restrict_to_workspace=True),
    )


# --------------------------------------------------------------------------- #
# unit: thinking detection                                                     #
# --------------------------------------------------------------------------- #


def test_has_inline_thinking():
    assert has_inline_thinking("<think>...</think>") is True
    assert has_inline_thinking("<THINKING>x") is True
    assert has_inline_thinking("plain text") is False
    assert has_inline_thinking(None) is False
    assert has_inline_thinking("") is False


def test_has_thinking():
    assert has_thinking(LLMResponse(content=None, reasoning_content="hmm")) is True
    assert has_thinking(LLMResponse(content=None, thinking_blocks=[{"x": 1}])) is True
    assert has_thinking(LLMResponse(content="<think>...</think>")) is True
    assert has_thinking(LLMResponse(content="real answer")) is False
    assert has_thinking(LLMResponse(content=None)) is False


# --------------------------------------------------------------------------- #
# unit: limits from config                                                     #
# --------------------------------------------------------------------------- #


def test_limits_from_defaults_uses_defaults_for_missing_attrs():
    assert limits_from_defaults(object()) == RecoveryLimits()


class _AlwaysEmptyProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key="test")
        self.calls = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.calls += 1
        return LLMResponse(content="", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


# ---------------------------------------------------------------------------
# Tag debris counts as empty
# ---------------------------------------------------------------------------


def test_lone_closing_think_tag_reads_as_empty() -> None:
    """A cut-off inline reasoning block leaves an opener-less closing tag.

    The paired-block substitution finds nothing to remove, so an
    eleven-character string reads as a real answer, and the tag is what the
    user sees.
    """
    for debris in ("</mm:think>", "</think>", "</thinking>", "</mm:think></mm:think>"):
        assert AgentLoop._strip_think(debris) is None, debris


def test_paired_namespaced_block_reads_as_empty() -> None:
    """A complete but empty namespaced block is debris too."""
    assert AgentLoop._strip_think("<mm:think></mm:think>") is None


def test_text_mentioning_a_tag_is_left_alone() -> None:
    """Debris detection must not eat an answer that talks about tags.

    The check asks whether anything survives removing the tags, and does not
    rewrite the text it returns.
    """
    said = "The model emits </think> at the end of its reasoning."
    assert AgentLoop._strip_think(said) == said


def test_real_content_after_a_think_block_survives() -> None:
    """The existing paired-block behaviour is unchanged."""
    assert AgentLoop._strip_think("<think>reasoning</think>the answer") == "the answer"


# ---------------------------------------------------------------------------
# Inline-thinking detection: namespaced and orphan spellings
# ---------------------------------------------------------------------------


def test_inline_thinking_detects_namespaced_and_orphan_tags() -> None:
    """The scan decides whether a turn produced reasoning at all.

    An opener-only pattern misses both shapes a truncated turn actually
    produces: a vendor-namespaced tag, and a closing tag whose opener the
    backend swallowed. Missing them reads a reply that stopped while thinking
    as a model that said nothing, which ends its turn silently instead of
    failing it.
    """
    from raven.agent.loop.recovery import has_inline_thinking

    assert has_inline_thinking("<mm:think>weighing options</mm:think>")
    assert has_inline_thinking("weighing options</think>")
    assert has_inline_thinking("<think>weighing options</think>")


def test_inline_thinking_ignores_prose_about_thinking() -> None:
    """No tag, no detection -- the word alone is not reasoning."""
    from raven.agent.loop.recovery import has_inline_thinking

    assert not has_inline_thinking("I was thinking about the reasoning behind it")
    assert not has_inline_thinking("")
    assert not has_inline_thinking(None)


def test_paired_namespaced_block_is_stripped_not_shown() -> None:
    """A complete block must not reach the user because of its prefix.

    Debris detection and the orphan split both match tags by shape, so a
    literal-only pattern here was the one place a vendor spelling still got
    through -- and it got through in the worst form: a whole reasoning block
    rendered as if it were the answer.
    """
    assert AgentLoop._strip_think("<mm:think>weighing options</mm:think>the answer") == "the answer"
    assert AgentLoop._strip_think("<thinking>weighing</thinking>the answer") == "the answer"


def test_a_mismatched_pair_is_not_treated_as_a_block() -> None:
    """Deleting between two unrelated tags would take real content with it."""
    text = "<think>weighing</thinking>the answer"

    assert AgentLoop._strip_think(text) == text


def test_one_end_namespaced_is_still_one_block() -> None:
    """A backend that stamps only one end still wrote a single block."""
    assert AgentLoop._strip_think("<mm:think>weighing</think>the answer") == "the answer"


# --------------------------------------------------------------------------- #
# the transport verdict must not call a silent model a failure                 #
# --------------------------------------------------------------------------- #
#
# The loop fails the turn on `finish_reason == "error"`, so a provider exit that
# reported an empty response as an error would fail a turn the model chose to
# end. The tests above stub `chat` directly and so never reach the provider's
# response exit, which is where such a verdict is reached -- this one goes
# through it on purpose.


class _SilentAfterToolProvider(LiteLLMProvider):
    """Calls a tool, then comes back with nothing.

    Honest usage, no text, built through the real `_parse_response` so a
    verdict at that exit is in the path.
    """

    def __init__(self) -> None:
        with (
            patch("raven.providers.litellm_provider.litellm"),
            patch.object(LiteLLMProvider, "_setup_env"),
        ):
            super().__init__(api_key="sk-test", provider_name="openrouter")
        self.calls = 0

    def get_default_model(self) -> str:
        return "stub"

    async def chat(self, messages, tools=None, model=None, **kw):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
                usage={"prompt_tokens": 900, "total_tokens": 910},
            )
        # Empty, and billed truthfully: a model that said nothing, not a
        # request that never arrived.
        return self._parse_response(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2900, completion_tokens=1, total_tokens=2901),
            ),
            sent_chars=12000,
        )


@pytest.mark.asyncio
async def test_a_silent_model_after_a_tool_ends_its_turn_without_a_failure(workspace):
    """Honest usage is a model that said nothing.

    Reported as an error instead, this turn would fail -- and the error would
    claim a transport failure, which for this response is simply untrue.
    """
    provider = _SilentAfterToolProvider()
    agent = _make_agent(workspace, provider)

    out = await agent._process_message(_turn("list the files"), session_key="s1")

    assert out is None
    assert provider.calls == 2


# --------------------------------------------------------------------------- #
# unit: which providers accept an assistant prefill                            #
# --------------------------------------------------------------------------- #


def test_base_provider_accepts_an_assistant_prefill_by_default():
    provider = _AlwaysEmptyProvider()
    assert provider.supports_assistant_prefill("some/model") is True
    assert provider.supports_assistant_prefill(None) is True


def test_litellm_provider_refuses_an_assistant_prefill_for_anthropic_models():
    """Same judgement that decides whether thinking_blocks go on the wire.

    The illegal request is exactly "trailing assistant message + thinking
    blocks", so the two must be decided by one rule or they drift apart.
    """
    provider = LiteLLMProvider(api_key="test-key", default_model="openai/gpt-4o")
    assert provider.supports_assistant_prefill("anthropic/claude-opus-5") is False
    assert provider.supports_assistant_prefill("claude-opus-4-5") is False
    assert provider.supports_assistant_prefill("openrouter/anthropic/claude-opus-5") is False


def test_litellm_provider_accepts_an_assistant_prefill_for_other_models():
    provider = LiteLLMProvider(api_key="test-key", default_model="openai/gpt-4o")
    assert provider.supports_assistant_prefill("openai/gpt-4o") is True
    assert provider.supports_assistant_prefill("deepseek/deepseek-chat") is True
    assert provider.supports_assistant_prefill(None) is True


def test_litellm_provider_answers_for_its_default_model_when_none_is_named():
    provider = LiteLLMProvider(api_key="test-key", default_model="anthropic/claude-opus-5")
    assert provider.supports_assistant_prefill() is False


def test_delegating_providers_answer_for_the_provider_they_wrap():
    """LazyProvider / PerModelProvider are the shapes the factory actually
    builds, so the answer has to come from the provider they wrap."""
    from raven.providers.base import GenerationSettings
    from raven.providers.lazy import LazyProvider
    from raven.providers.per_model_provider import PerModelProvider

    inner = LiteLLMProvider(api_key="test-key", default_model="anthropic/claude-opus-5")
    lazy = LazyProvider(factory=lambda: inner, default_model="anthropic/claude-opus-5", generation=GenerationSettings())
    assert lazy.supports_assistant_prefill("anthropic/claude-opus-5") is False
    assert lazy.supports_assistant_prefill("openai/gpt-4o") is True

    routed = PerModelProvider(models=[], fallback=inner)
    assert routed.supports_assistant_prefill("anthropic/claude-opus-5") is False
    assert routed.supports_assistant_prefill("openai/gpt-4o") is True


# --------------------------------------------------------------------------- #
# loop: the output-limit marker on the session                                 #
# --------------------------------------------------------------------------- #


class _AlwaysCutProvider(LLMProvider):
    """Every call ends on the output ceiling having produced nothing at all."""

    def __init__(self):
        super().__init__(api_key="test")
        self.calls = 0
        self.efforts: list[str | None] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.calls += 1
        self.efforts.append(reasoning_effort)
        return LLMResponse(content="", finish_reason="length")

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_a_turn_cut_at_the_ceiling_records_it_on_the_session(workspace):
    """Recorded on the session, not on a message: this turn persists no
    assistant row at all, so a message-seated record has nowhere to land."""
    agent = _make_agent(workspace, _AlwaysCutProvider(), limits=RecoveryLimits())

    await _run_answerless(agent)

    # Recorded although the turn failed: a turn cut at the ceiling is the one
    # most likely to come back with nothing, so a marker written only on the way
    # out of a successful turn would never see the case it exists for.
    session = agent.sessions.get_or_create("s1")
    assert session.metadata.get("output_limit_turn_at") == 0


@pytest.mark.asyncio
async def test_a_silent_turn_records_no_output_limit(workspace):
    agent = _make_agent(workspace, _AlwaysEmptyProvider(), limits=RecoveryLimits())

    await agent._process_message(_turn(), session_key="s1")

    # Said rather than left unsaid: a save merges over the record on disk, so
    # the marker is cleared by writing None. Its reader asks for an int.
    assert agent.sessions.get_or_create("s1").metadata.get("output_limit_turn_at") is None


@pytest.mark.asyncio
async def test_a_turn_that_was_not_cut_clears_a_marker_left_by_an_earlier_one(workspace):
    """The slot has to be turn-scoped by being written OR cleared every turn.

    Scoping it by the message index alone assumes that index only grows, and
    `Session.clear()` (what `/new` calls) and `undo_last_turn` both move it
    back without touching metadata -- so an old marker can sit at an index a
    later turn's own start satisfies, and that turn gets reported as cut.
    """
    provider = _OneReplyProvider(LLMResponse(content=None, reasoning_content="first I need to", finish_reason="stop"))
    agent = _make_agent(workspace, provider, limits=RecoveryLimits())
    agent.sessions.get_or_create("s1").metadata["output_limit_turn_at"] = 0

    # And a turn that fails clears it too: the failing turn is the likeliest one
    # to be reported as cut by a marker it did not write.
    await _run_answerless(agent)

    # Said rather than left unsaid: a save merges over the record on disk, so
    # the marker is cleared by writing None. Its reader asks for an int.
    assert agent.sessions.get_or_create("s1").metadata.get("output_limit_turn_at") is None


# --------------------------------------------------------------------------- #
# loop: an empty reply ends the turn silently                                  #
# --------------------------------------------------------------------------- #


class _EmptyAfterToolProvider(LLMProvider):
    """Runs one tool, then ends its turn with nothing to say."""

    def __init__(self):
        super().__init__(api_key="test")
        self.calls = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
            )
        return LLMResponse(content="", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


class _OneReplyProvider(LLMProvider):
    """Answers every call with the same response."""

    def __init__(self, response: LLMResponse):
        super().__init__(api_key="test")
        self.calls = 0
        self.efforts: list[str | None] = []
        self._response = response

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.calls += 1
        self.efforts.append(reasoning_effort)
        return self._response

    def get_default_model(self) -> str:
        return "stub"


class _RecordsAfterSend(AgentHook):
    def __init__(self):
        self.seen: list[str | None] = []

    async def after_send(self, ctx):
        self.seen.append(ctx.outbound_content)
        return HookDecision()


@pytest.mark.asyncio
async def test_an_empty_reply_is_not_asked_again(workspace):
    provider = _AlwaysEmptyProvider()
    agent = _make_agent(workspace, provider)

    await agent._process_message(_turn(), session_key="s1")

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_an_empty_reply_delivers_nothing(workspace):
    agent = _make_agent(workspace, _AlwaysEmptyProvider())

    out = await agent._process_message(_turn(), session_key="s1")

    assert out is None


@pytest.mark.asyncio
async def test_an_empty_reply_keeps_the_question_and_files_no_failure(workspace):
    agent = _make_agent(workspace, _AlwaysEmptyProvider())

    await agent._process_message(_turn("routine progress, nothing to add"), session_key="s1")

    rows = agent.sessions.get_or_create("s1").messages
    assert [m["role"] for m in rows] == ["user"]
    assert "routine progress, nothing to add" in str(rows[0]["content"])


@pytest.mark.asyncio
async def test_an_empty_message_with_no_reasoning_behind_it_ends_silently(workspace):
    """The measured shape: an empty final message is billed a few completion
    tokens, none of them reasoning, and that is a model that said nothing."""
    provider = _OneReplyProvider(
        LLMResponse(content="", finish_reason="stop", usage={"completion_tokens": 4, "reasoning_tokens": 0})
    )
    agent = _make_agent(workspace, provider)

    out = await agent._process_message(_turn(), session_key="s1")

    assert out is None
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_saying_nothing_after_a_tool_call_ends_the_turn_normally(workspace):
    provider = _EmptyAfterToolProvider()
    agent = _make_agent(workspace, provider)

    out = await agent._process_message(_turn("list the files"), session_key="s1")

    assert out is None
    assert provider.calls == 2
    rows = agent.sessions.get_or_create("s1").messages
    assert [m["role"] for m in rows] == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_a_reply_that_stops_in_its_reasoning_fails_once_no_lower_rung_is_left(workspace):
    provider = _OneReplyProvider(LLMResponse(content=None, reasoning_content="first I need to", finish_reason="stop"))
    agent = _make_agent(workspace, provider)

    failure = await _run_answerless(agent)

    assert provider.efforts == [None, "high", "low"]
    assert "thinking" in failure


@pytest.mark.asyncio
async def test_a_reply_that_stops_in_hidden_reasoning_fails_once_no_lower_rung_is_left(workspace):
    """Codex bills its reasoning but never returns the text, so the token
    count is the only sign the model stopped while thinking."""
    provider = _OneReplyProvider(
        LLMResponse(content="", finish_reason="stop", usage={"completion_tokens": 30, "reasoning_tokens": 26})
    )
    agent = _make_agent(workspace, provider)

    failure = await _run_answerless(agent)

    assert provider.efforts == [None, "high", "low"]
    assert "thinking" in failure


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", ["n/a", True, 2.5, -3])
async def test_a_reasoning_count_that_is_not_a_count_is_no_sign_of_thinking(workspace, spent):
    """Read the way the usage normalizer reads it: a value that is not a token
    count says nothing about thinking, and does not crash the turn either."""
    provider = _OneReplyProvider(LLMResponse(content="", finish_reason="stop", usage={"reasoning_tokens": spent}))
    agent = _make_agent(workspace, provider)

    out = await agent._process_message(_turn(), session_key="s1")

    assert out is None


@pytest.mark.asyncio
async def test_a_reply_that_is_only_an_inline_think_block_fails_once_no_lower_rung_is_left(workspace):
    provider = _OneReplyProvider(LLMResponse(content="<think>first I need to</think>", finish_reason="stop"))
    agent = _make_agent(workspace, provider)

    failure = await _run_answerless(agent)

    assert provider.efforts == [None, "high", "low"]
    assert "thinking" in failure


@pytest.mark.asyncio
async def test_an_empty_reply_cut_at_the_output_limit_fails_naming_the_limit_once_no_rung_is_left(workspace):
    """A reply cut at the ceiling is not a chosen silence: it is asked again
    with less thinking allowed, and fails if that is cut too."""
    provider = _AlwaysCutProvider()
    agent = _make_agent(workspace, provider)

    failure = await _run_answerless(agent)

    assert provider.efforts == [None, "high", "low"]
    assert "output limit" in failure


class _ThinksUntilProvider(LLMProvider):
    """Stops in its reasoning at every effort but the ones it answers at, and
    records the effort each call asked for."""

    def __init__(self, answers_at: tuple[str | None, ...] = ()):
        super().__init__(api_key="test")
        self.efforts: list[str | None] = []
        self._answers_at = answers_at

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.efforts.append(reasoning_effort)
        if reasoning_effort in self._answers_at:
            return LLMResponse(content="the answer", finish_reason="stop")
        return LLMResponse(content=None, reasoning_content="first I need to", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


def _at_effort(agent: AgentLoop, effort: str) -> AgentLoop:
    agent.set_session_policy("s1", mode=effort, reasoning_effort=effort)
    return agent


def test_a_step_down_lands_on_high_or_low_below_a_named_effort():
    assert lower_reasoning_effort("max") == "high"
    assert lower_reasoning_effort("xhigh") == "high"
    assert lower_reasoning_effort("high") == "low"
    assert lower_reasoning_effort("medium") == "low", "high is above medium, so it is no step down"
    assert lower_reasoning_effort("low") is None
    assert lower_reasoning_effort("minimal") is None
    assert lower_reasoning_effort(None) == "high", "a default the loop cannot see is stepped down from the top"
    assert lower_reasoning_effort(" HIGH ") == "low", "an effort is a config string, not a keyword"
    assert lower_reasoning_effort("enthusiastic") is None, "a value that cannot be placed has no rung below it"


@pytest.mark.asyncio
async def test_a_reply_that_stops_in_its_reasoning_is_asked_again_a_rung_lower(workspace):
    """Thinking that spent the budget before the answer began is asked for
    again with less thinking allowed, one rung at a time."""
    provider = _ThinksUntilProvider(answers_at=("low",))
    agent = _at_effort(_make_agent(workspace, provider), "max")

    out = await agent._process_message(_turn(), session_key="s1")

    assert provider.efforts == ["max", "high", "low"]
    assert out[0] == "the answer"


@pytest.mark.asyncio
async def test_a_medium_turn_steps_down_to_low(workspace):
    provider = _ThinksUntilProvider(answers_at=("low",))
    agent = _at_effort(_make_agent(workspace, provider), "medium")

    out = await agent._process_message(_turn(), session_key="s1")

    assert provider.efforts == ["medium", "low"]
    assert out[0] == "the answer"


@pytest.mark.asyncio
async def test_a_turn_already_at_the_floor_fails_without_asking_again(workspace):
    provider = _ThinksUntilProvider(answers_at=("high",))
    agent = _at_effort(_make_agent(workspace, provider), "low")

    failure = await _run_answerless(agent)

    assert provider.efforts == ["low"]
    assert "thinking" in failure


class _ScriptedEffortProvider(LLMProvider):
    """Answers each call with the next reply of a script, recording the effort
    each call asked for."""

    def __init__(self, *replies: LLMResponse):
        super().__init__(api_key="test")
        self.efforts: list[str | None] = []
        self._replies = replies

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.efforts.append(reasoning_effort)
        return self._replies[min(len(self.efforts), len(self._replies)) - 1]

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_the_lower_effort_is_spent_on_the_retry_alone(workspace):
    """A call that answers leaves the turn's own effort standing for the calls
    after it."""
    provider = _ScriptedEffortProvider(
        LLMResponse(content=None, reasoning_content="first I need to", finish_reason="stop"),
        LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
        ),
        LLMResponse(content="the answer", finish_reason="stop"),
    )
    agent = _at_effort(_make_agent(workspace, provider), "max")

    out = await agent._process_message(_turn("list the files"), session_key="s1")

    assert provider.efforts == ["max", "high", "max"]
    assert out[0] == "the answer"


@pytest.mark.asyncio
async def test_a_turn_steps_down_each_rung_once(workspace):
    """A second stall in the same turn carries on from the rung the first one
    reached rather than starting over, so a turn asks again at most once per
    rung."""
    stuck = LLMResponse(content=None, reasoning_content="first I need to", finish_reason="stop")
    provider = _ScriptedEffortProvider(
        stuck,
        LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
        ),
        stuck,
        LLMResponse(content="the answer", finish_reason="stop"),
    )
    agent = _at_effort(_make_agent(workspace, provider), "max")

    out = await agent._process_message(_turn("list the files"), session_key="s1")

    assert provider.efforts == ["max", "high", "max", "low"]
    assert out[0] == "the answer"


@pytest.mark.asyncio
async def test_running_out_of_rungs_on_the_last_iteration_still_fails_the_turn(workspace):
    """A failure that lands on the iteration cap is not a turn that ran out of
    iterations: nothing wraps it up into a reply."""
    provider = _ThinksUntilProvider()
    agent = AgentLoop(
        provider=provider,
        workspace=workspace,
        model="stub",
        policy=TurnPolicy(max_iterations=3),
        tools=ToolWiring(restrict_to_workspace=True),
    )

    failure = await _run_answerless(agent)

    assert provider.efforts == [None, "high", "low"]
    assert "thinking" in failure


@pytest.mark.asyncio
async def test_saying_nothing_on_the_last_iteration_ends_the_turn_without_a_wrap_up(workspace):
    """A turn reaches its cap by working, not by answering. A model that ends the
    turn on its last iteration has ended it: a wrap-up would put words in its
    mouth and file the turn as cut short."""
    provider = _AlwaysEmptyProvider()
    agent = _make_agent(workspace, provider, max_iterations=1)

    out = await agent._process_message(_turn(), session_key="s1")

    assert out is None
    assert provider.calls == 1
    assert [m["role"] for m in agent.sessions.get_or_create("s1").messages] == ["user"]


@pytest.mark.asyncio
async def test_a_step_down_owed_on_the_last_iteration_is_still_asked(workspace):
    """Asking again a rung lower repeats a call that did no work, so it spends no
    iteration: the step-down, and the failure once no rung is left, land on a
    turn's last iteration exactly as on its first."""
    provider = _OneReplyProvider(LLMResponse(content="", reasoning_content="Let me weigh it.", finish_reason="stop"))
    agent = _make_agent(workspace, provider, max_iterations=1)

    failure = await _run_answerless(agent)

    assert provider.efforts == [None, "high", "low"]
    assert "thinking" in failure


@pytest.mark.asyncio
async def test_a_silent_turn_still_runs_the_after_send_chain_with_no_reply(workspace):
    """The chain is also where a turn's hooks do their bookkeeping, so it runs
    on a silent turn too -- handed no reply, rather than an empty one."""
    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    hook = _RecordsAfterSend()
    agent.hooks.append(hook)

    out = await agent._process_message(_turn(), session_key="s1")

    assert hook.seen == [None]
    assert out is None


class _FilesAHold(AgentHook):
    """What the oncall flow does once its campaign has a wake pending."""

    async def after_send(self, ctx):
        from raven.acp.methods import ACP_META_OBSERVER, HOLD_TURN_META

        observers = ctx.metadata.setdefault("observers", {})
        observers.setdefault(ACP_META_OBSERVER, {})[HOLD_TURN_META] = {
            "untilMs": None,
            "why": "waiting for the next look",
        }
        return HookDecision()


@pytest.mark.asyncio
async def test_a_hold_filed_on_a_silent_turn_reaches_the_acp_client(workspace):
    from raven.acp.methods import HOLD_TURN_META, AcpMethods

    agent = _make_agent(workspace, _EmptyAfterToolProvider())
    agent.hooks.append(_FilesAHold())

    out = await agent._process_message(_turn("watch the deploy"), session_key="s1")

    assert out is None
    assert AcpMethods._turn_meta(agent.sessions, "s1", 0) == {
        HOLD_TURN_META: {"untilMs": None, "why": "waiting for the next look"}
    }


@pytest.mark.asyncio
async def test_a_hold_filed_on_a_silent_turn_that_called_no_tool_still_reaches_the_acp_client(workspace):
    """No tool ran, so the turn wrote no assistant row for its observers to ride
    on. A wait the client reads as finished is the failure the hold exists for."""
    from raven.acp.methods import HOLD_TURN_META, AcpMethods

    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    agent.hooks.append(_FilesAHold())

    out = await agent._process_message(_turn("anything new?"), session_key="s1")

    assert out is None
    assert [m["role"] for m in agent.sessions.get_or_create("s1").messages] == ["user"]
    assert AcpMethods._turn_meta(agent.sessions, "s1", 0) == {
        HOLD_TURN_META: {"untilMs": None, "why": "waiting for the next look"}
    }


class _FilesAHoldOnce(_FilesAHold):
    def __init__(self):
        self.filed = False

    async def after_send(self, ctx):
        if self.filed:
            return HookDecision()
        self.filed = True
        return await super().after_send(ctx)


@pytest.mark.asyncio
async def test_an_earlier_silent_turn_s_hold_does_not_answer_a_later_prompt(workspace):
    """A prompt answered before the loop runs (a slash command) files nothing, so
    the earlier turn's record is still there; where it was filed says it is not
    this prompt's."""
    from raven.acp.methods import AcpMethods

    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    agent.hooks.append(_FilesAHoldOnce())
    await agent._process_message(_turn("anything new?"), session_key="s1")
    assert AcpMethods._turn_meta(agent.sessions, "s1", 0) is not None, "the first turn filed a hold"
    filed_before = len(agent.sessions.get_or_create("s1").messages)

    await agent._process_message(_turn("/help"), session_key="s1")

    assert AcpMethods._turn_meta(agent.sessions, "s1", filed_before) is None


@pytest.mark.asyncio
async def test_a_hold_from_an_undone_turn_does_not_answer_the_turn_after_it(workspace):
    """Undo rewinds the messages and leaves the session's record alone, so the next
    turn starts where the undone one did. That turn files its own record, an empty
    one included, or it would read the undone turn's hold as its own."""
    from raven.acp.methods import AcpMethods

    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    agent.hooks.append(_FilesAHoldOnce())
    await agent._process_message(_turn("anything new?"), session_key="s1")
    assert AcpMethods._turn_meta(agent.sessions, "s1", 0) is not None, "the first turn filed a hold"
    session = agent.sessions.get_or_create("s1")
    session.undo_last_turn()
    agent.sessions.save(session)

    await agent._process_message(_turn("and now?"), session_key="s1")

    assert AcpMethods._turn_meta(agent.sessions, "s1", 0) is None


class _ComposesADeck(AgentHook):
    """A deck engine's outbound: whatever the reply was, the deck lines after it."""

    async def after_send(self, ctx):
        return HookDecision(modified_content=(ctx.outbound_content or "") + "\n\nPublished a deck.\nMEDIA: /out/d.pptx")


@pytest.mark.asyncio
async def test_what_a_hook_composes_on_a_silent_turn_is_the_reply(workspace):
    """The deck was built this turn; the model saying nothing about it does not
    make it undelivered. With no reply of the model's own, the separator the
    hook put in front of its lines separates nothing, so the reply starts at
    the lines -- but a reply that streamed text earlier in the turn still gets
    the tail as composed, separator included."""
    agent = _make_agent(workspace, _EmptyAfterToolProvider())
    agent.hooks.append(_ComposesADeck())
    sink: dict[str, str] = {}

    out = await agent._process_message(_turn("make the deck"), session_key="s1", hook_sink=sink)

    assert out == ("Published a deck.\nMEDIA: /out/d.pptx", [])
    assert sink["appended"] == "\n\nPublished a deck.\nMEDIA: /out/d.pptx"


class _ComposesWhitespace(AgentHook):
    async def after_send(self, ctx):
        return HookDecision(modified_content="\n\n")


@pytest.mark.asyncio
async def test_a_hook_that_composes_only_whitespace_leaves_the_turn_silent(workspace):
    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    agent.hooks.append(_ComposesWhitespace())

    out = await agent._process_message(_turn(), session_key="s1")

    assert out is None


@pytest.mark.asyncio
async def test_a_queued_nudge_waits_out_a_silent_turn(workspace):
    """A nudge rides on a reply. A turn with none would otherwise send the
    nudge alone, as a message the model never wrote."""
    from raven.agent.hook.adapters import ResponseModifierAdapter
    from raven.proactive_engine.sentinel.executor.injector import NudgeInjector

    injector = NudgeInjector()
    injector.queue("s1", "Your build finished.")
    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    agent.hooks.append(ResponseModifierAdapter(injector))

    out = await agent._process_message(_turn(), session_key="s1")

    assert out is None
    assert injector.peek("s1") == ["Your build finished."]


@pytest.mark.asyncio
async def test_a_silent_turn_has_no_reply_to_learn_from(workspace, monkeypatch):
    import raven.agent.personalizer as personalizer_mod

    learned: list[tuple] = []

    class _SpyPersonalizer:
        def __init__(self, *a, **k):
            pass

        async def classify(self, content, history=None):
            return {"needs_clarification": False}

        async def post_learn(self, *args, **kwargs):
            learned.append(args)

    monkeypatch.setattr(personalizer_mod, "Personalizer", _SpyPersonalizer)
    agent = _make_agent(workspace, _AlwaysEmptyProvider())
    agent.configure_personalization(True)

    await agent._process_message(_turn(), session_key="s1")
    await asyncio.gather(*list(agent._consolidation_tasks))

    assert learned == []


class _SilentThenAnswers(LLMProvider):
    """Says nothing on its first call and answers every one after, keeping each request."""

    def __init__(self):
        super().__init__(api_key="test")
        self.requests: list[list[dict]] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.requests.append([dict(m) for m in messages])
        if len(self.requests) == 1:
            return LLMResponse(content="", finish_reason="stop")
        return LLMResponse(content="Both, then.", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_the_call_after_a_silent_turn_carries_no_two_user_messages_in_a_row(workspace):
    """A silent turn that called no tool leaves its question with no reply after it,
    and the next question lands right behind it. Some providers refuse two user
    messages in a row, and would refuse every call for the rest of the session.
    The history keeps both; only what is sent folds them."""
    provider = _SilentThenAnswers()
    agent = _make_agent(workspace, provider)

    await agent._process_message(_turn("first question"), session_key="s1")
    await agent._process_message(_turn("second question"), session_key="s1")

    roles = [m["role"] for m in provider.requests[-1]]
    assert not any(a == b == "user" for a, b in zip(roles, roles[1:])), roles
    sent = str(provider.requests[-1])
    assert "first question" in sent and "second question" in sent
    rows = agent.sessions.get_or_create("s1").messages
    assert [m["role"] for m in rows] == ["user", "user", "assistant"]


class _RecordsRequests(LLMProvider):
    """Answers from a script, one response per call, keeping each request."""

    def __init__(self, *responses: LLMResponse):
        super().__init__(api_key="test")
        self.requests: list[list[dict]] = []
        self._responses = responses

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.requests.append([dict(m) for m in messages])
        return self._responses[min(len(self.requests), len(self._responses)) - 1]

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_the_wrap_up_after_a_silent_turn_carries_no_two_user_messages_in_a_row(workspace):
    """The wrap-up of a turn that ran out of iterations is a call of its own, on
    the turn's conversation plus one instruction, so it folds the same way."""
    provider = _RecordsRequests(
        LLMResponse(content="", finish_reason="stop"),
        LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
        ),
        LLMResponse(content="Here is what I found.", finish_reason="stop"),
    )
    agent = _make_agent(workspace, provider, max_iterations=1)

    await agent._process_message(_turn("first question"), session_key="s1")
    out = await agent._process_message(_turn("second question"), session_key="s1")

    assert out[0] == "Here is what I found.", "the second turn ran out and was wrapped up"
    roles = [m["role"] for m in provider.requests[-1]]
    assert not any(a == b == "user" for a, b in zip(roles, roles[1:])), roles
