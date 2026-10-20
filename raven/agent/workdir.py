"""Resolve the working directory a turn runs in.

Agent home (user memory, skills, transcripts) is global and separate; this
module only decides where a turn reads and writes files.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from raven.utils.paths import safe_path_segment

# Subtrees of agent home the agent must not be able to adopt as a working
# directory: it would then write artifacts over its own memory and skills.
# ``sessions`` covers the subagent history too -- it lives under each session's
# own directory there (raven/agent/subagent/history.py).
_PROTECTED_SUBTREES = ("user_memory", "skills", "sessions")

# Directories Raven keeps its own state in, directly under its home: provider
# tokens, MCP credentials, machine passwords, cron jobs, channel sessions, the
# engines' homes, installed agents and plugins. Matched under the resolved home,
# so a home not named `.raven` and one reached through a symlink are covered
# alike. Agent home (`workspace`) has its own rules above and the per-channel
# root (`tmp`) is where work happens, so neither is here, and a directory of
# the user's own beside them stays usable. A test walks raven/ for every
# directory created under the home, so a new one cannot be left off unnoticed.
_STATE_ENTRIES = frozenset(
    {
        "acp-launches",
        "agents",
        "background",
        "bridge",
        "cache",
        "credentials",
        "cron",
        "deliverables",
        "everos",
        "imports",
        "keys",
        "knowledge",
        "logs",
        "matrix-store",
        "media",
        "mochat",
        "oauth",
        "plugins",
        "runtime",
        "sandbox",
        "sentinel",
        "subagent_sessions",
        "subagents",
        "telemetry",
        "traces",
        "weixin",
        "whatsapp-auth",
    }
)

_CURRENT: ContextVar[Path | None] = ContextVar("raven_session_workdir", default=None)


def default_channel_root(agent_home: Path) -> Path:
    """Root of the per-channel working directories, a sibling of agent home.

    With the default agent home (``~/.raven/workspace``) that is
    ``~/.raven/tmp``. Derived from agent home rather than from ``$HOME`` so an
    instance started with ``--home`` keeps its scratch directories next to that
    instance instead of reaching back into the real one.

    A sibling, not a child: these directories hold whatever a chat happens to
    produce, which is not part of the agent's own memory and skills. One shared
    root also lets a single sandbox mount cover every channel.

    The one case a plain sibling gets wrong is a home directly under the
    filesystem root -- ``--home /workspace``, a common container mount, would
    make the root the system ``/tmp``: shared with every other process,
    world-writable, and cleared on reboot. There the sibling is named after the
    home instead, which cannot collide with an existing system directory.
    """
    home = Path(agent_home).expanduser()
    if home.parent == Path(home.anchor) or not home.name:
        return home.parent / f"{home.name or 'raven'}-tmp"
    return home.parent / "tmp"


class WorkdirPolicy(str, Enum):
    """How an entrypoint picks a working directory when nothing overrides it."""

    LAUNCH_DIR = "launch_dir"
    PER_CHANNEL = "per_channel"


def current() -> Path | None:
    """The working directory bound for the running turn, if any."""
    return _CURRENT.get()


@contextmanager
def bind(path: Path) -> Iterator[None]:
    """Bind ``path`` as the working directory for the enclosing block."""
    token = _CURRENT.set(path)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def repoint(path: Path) -> None:
    """Repoint the running turn's binding so the very next read sees ``path``.

    For the one caller that moves a session's root mid-turn (the
    ``rebind_workdir`` grant), after persisting the override it repoints.
    The enclosing ``bind``'s finally still resets the variable at turn end;
    the next turn resolves the persisted override instead.
    """
    _CURRENT.set(path)


def is_within(path: Path, root: Path) -> bool:
    """Whether ``path`` sits inside ``root``, comparing physical paths.

    A persisted override is resolved through symlinks (``validate_override``
    calls ``.resolve()``); ``mount_root()`` and ``workspace`` generally are
    not. Comparing an unresolved and a resolved path with ``is_relative_to``
    can disagree about a directory that is, on disk, the very same place.
    """
    return path.resolve().is_relative_to(root.resolve())


def _raven_home() -> Path:
    """Raven's **Home** (``RAVEN_HOME``, by default ``~/.raven``), resolved."""
    from raven.home import raven_home

    return raven_home().expanduser().resolve()


def _inside_state(path: Path, homes: tuple[Path, ...], raven_root: Path) -> str | None:
    """Why ``path`` (resolved) lies in Raven's own state, or ``None`` when it does not.

    Containing that state is a separate question: a checkpoint can leave a
    contained tree out, but a root inside one has nothing else to snapshot.
    """
    for home in homes:
        # Agent home itself is rejected for the same reason its subtrees are, and
        # more strongly: from there every protected subtree is one relative path
        # away, so an ordinary relative write lands on the agent's own memory.
        if path == home:
            return f"working directory must not be the agent home directory itself ({home})"
        for subtree in _PROTECTED_SUBTREES:
            candidate = home / subtree
            if path == candidate or candidate in path.parents:
                return f"working directory must not be inside the agent's {subtree} tree ({candidate})"
    if path == raven_root:
        return f"working directory must not be Raven's home directory itself ({raven_root})"
    if raven_root in path.parents:
        entry = path.relative_to(raven_root).parts[0]
        if entry in _STATE_ENTRIES:
            return f"working directory must not be inside Raven's own state ({raven_root / entry})"
    return None


def validate_override(value: str | Path, agent_home: Path) -> Path:
    """Check a user-supplied working directory, returning it resolved."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"working directory must be an absolute path, got {value!r}")
    resolved = path.resolve()
    home = Path(agent_home).expanduser().resolve()
    raven_root = _raven_home()
    # An ancestor is worse still. `~/.raven` is not merely agent home's parent,
    # it is the instance data directory -- config.json (provider keys), oauth/
    # (provider tokens), cron/, sentinel/, logs/. The per-turn checkpoint runs
    # `add -A` over the working directory, and the `.raven/` default exclude
    # cannot help when `.raven` *is* the work-tree root, so pointing a session
    # here would commit every credential into a shadow git repo.
    if resolved in home.parents:
        raise ValueError(f"working directory must not contain the agent home directory ({home})")
    if resolved in raven_root.parents:
        raise ValueError(f"working directory must not contain Raven's home directory ({raven_root})")
    reason = _inside_state(resolved, (home,), raven_root)
    if reason is not None:
        raise ValueError(reason)
    return resolved


def state_refusal(root: Path, agent_home: Path | None = None) -> str | None:
    """Why a snapshot must not be rooted at ``root``, or ``None`` when it may be.

    Judged on its own rather than through ``validate_override``: whatever
    entrance chose ``root``, a snapshot of it is history that outlives a
    rotated key. The default agent home is held to the same rule as the
    configured one, since a process homed elsewhere can still be handed it.
    """
    from raven.config.paths import default_workspace

    homes = tuple(dict.fromkeys(Path(h).expanduser().resolve() for h in (agent_home, default_workspace()) if h))
    return _inside_state(Path(root).expanduser().resolve(), homes, _raven_home())


def state_within(root: Path, agent_home: Path | None = None) -> tuple[Path, ...]:
    """Raven's own state lying strictly inside ``root``, as resolved paths.

    What a snapshot or a listing of ``root`` leaves out: Raven's home, agent
    home and the configuration file in use, matched by where they are rather
    than by being named ``.raven``. A path inside another of them is folded
    into it.
    """
    from raven.home import get_config_path

    root = Path(root).expanduser().resolve()
    candidates = [_raven_home(), Path(get_config_path()).expanduser().resolve()]
    if agent_home is not None:
        candidates.append(Path(agent_home).expanduser().resolve())
    inside = {path for path in candidates if root in path.parents}
    return tuple(sorted(path for path in inside if not any(other in path.parents for other in inside)))


class WorkdirResolver:
    """Maps a session key to the directory that session's turns work in."""

    def __init__(
        self,
        policy: WorkdirPolicy,
        *,
        agent_home: Path,
        launch_dir: Path | None = None,
        session_root: Path | None = None,
        sessions: Any = None,
        explicit_workdir: Path | None = None,
        channel_workspaces: dict[str, str] | None = None,
    ) -> None:
        self._policy = policy
        self._agent_home = Path(agent_home).expanduser()
        self._launch_dir = Path(launch_dir) if launch_dir else Path.cwd()
        self._session_root = Path(session_root) if session_root else default_channel_root(self._agent_home)
        self._sessions = sessions
        self._explicit_workdir = Path(explicit_workdir) if explicit_workdir else None
        # channel name -> configured directory, from `channels.<name>.workspace`.
        # A channel with no entry (or an empty one) falls back to
        # `<session_root>/<channel>`.
        #
        # Validated here, at construction, and not where it is used. This value
        # arrives from a config file or the web UI's channels form, which are
        # the two places nothing else checks it -- every other way of naming a
        # working directory (`-w`, the per-session override) already goes
        # through `validate_override`. Without this, `channels.x.workspace =
        # <agent home>` is accepted and the per-turn `add -A` then runs over
        # user_memory/ and sessions/.
        #
        # A bad entry is dropped with a warning rather than raised: the value
        # may have been typed days ago in a browser, so failing the turn that
        # happens to use it reports the problem far from where it was made,
        # and falling back to `<session_root>/<channel>` is always usable.
        self._channel_workspaces: dict[str, Path] = {}
        for name, configured in (channel_workspaces or {}).items():
            if not configured:
                continue
            try:
                self._channel_workspaces[name] = validate_override(configured, self._agent_home)
            except ValueError as exc:
                logger.warning(
                    "channels.{}.workspace is not a usable working directory ({}); falling back to {}",
                    name,
                    exc,
                    self._session_root / safe_path_segment(name),
                )

    def resolve(self, session_key: str, *, create: bool = True) -> Path:
        """Resolve the working directory for ``session_key``, creating it by default.

        ``create=False`` is for callers that only want to know the path -- a
        read must not leave a directory behind on disk.
        """
        path = self._explicit_workdir or self._persisted(session_key) or self._default(session_key)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _persisted(self, session_key: str) -> Path | None:
        if self._sessions is None or not session_key:
            return None
        stored = self._sessions.get_or_create(session_key).metadata.get("workdir")
        if not stored:
            return None
        return validate_override(stored, self._agent_home)

    def _default(self, session_key: str) -> Path:
        """The policy default: the launch directory, or the channel's directory.

        Under ``PER_CHANNEL`` every chat on a channel shares one directory --
        the channel is what the user configures, and its conversations are the
        same kind of work. A single session can still be pinned elsewhere
        through the persisted override, which is checked before this.
        """
        if self._policy is WorkdirPolicy.LAUNCH_DIR:
            return self._launch_dir
        name = session_key.partition(":")[0]
        configured = self._channel_workspaces.get(name)
        if configured is not None:
            return configured
        return self._session_root / (safe_path_segment(name) or "_")

    def mount_root(self) -> Path:
        """The directory a sandbox VM must mount to cover every session.

        The default per-channel directories share one root, so a single mount
        serves them all; a launch-directory process only ever has the one. A
        channel pointed somewhere else by config, or an explicit override,
        falls outside it -- ``check_workdir_mounted`` refuses those rather than
        letting a turn run against a directory the VM cannot see.

        ``explicit_workdir`` wins here exactly as it does in ``resolve()``:
        when set, every session resolves to it regardless of policy, so the
        mount must follow it too.
        """
        if self._explicit_workdir is not None:
            return self._explicit_workdir
        if self._policy is WorkdirPolicy.LAUNCH_DIR:
            return self._launch_dir
        return self._session_root
