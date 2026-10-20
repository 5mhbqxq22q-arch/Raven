"""Resolve the working directory a turn runs in.

Agent home (user memory, skills, transcripts) is global and separate; this
module only decides where a turn reads and writes files.
"""

from __future__ import annotations

import os
import unicodedata
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
# browser profile, the engines' homes, installed agents and plugins. Matched
# under the resolved home -- and under `~/.raven`, which a few writers use
# whatever RAVEN_HOME says -- so a home not named `.raven` and one reached
# through a symlink are covered alike. Agent home (`workspace`) has its own
# rules above and the per-channel root (`tmp`) is where work happens, so
# neither is here, and a directory of the user's own beside them stays usable.
# A test walks raven/ for every directory created under the home, so a new one
# cannot be left off unnoticed.
_STATE_ENTRIES = frozenset(
    {
        "acp-launches",
        "agents",
        "background",
        "bridge",
        "browser-profile",
        "cache",
        "credentials",
        "cron",
        "deliverables",
        "everos",
        "imports",
        "keys",
        "knn_embcache",
        "knowledge",
        "logs",
        "machines",
        "matrix-store",
        "media",
        "mochat",
        "oauth",
        "plugins",
        "render",
        "routing",
        "runtime",
        "sandbox",
        "sentinel",
        "skills",
        "subagent_sessions",
        "subagents",
        "telemetry",
        "traces",
        "weixin",
        "whatsapp-auth",
    }
)

# What Raven keeps in the data directory beside the config file
# (``get_runtime_subdir`` and ``get_data_dir()``). It is Raven's home unless
# `--config` moves it, and then it is often the user's project, so only these
# names are protected there, never the directory itself or anything else in it.
# The same test keeps this list exactly equal to what the code anchors there.
_DATA_DIR_STATE = frozenset(
    {
        "acp-launches",
        "cache",
        "credentials",
        "cron",
        "deliverables",
        "everos",
        "everos-server.lock",
        "everos-server.pid",
        "gateway.lock",
        "import_state.json",
        "knowledge",
        "logs",
        "matrix-store",
        "media",
        "mochat",
        "plugins",
        "render",
        "sandbox",
        "sentinel",
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


def _state_roots() -> tuple[Path, ...]:
    """Raven's home, and the default one a few writers use whatever ``RAVEN_HOME`` says."""
    from raven.contracts.path_policy import DEFAULT_HOME_DIRNAME

    return tuple(dict.fromkeys((_raven_home(), (Path.home() / DEFAULT_HOME_DIRNAME).resolve())))


def _data_dir() -> Path:
    """The data directory beside the config file in use, resolved, created by nothing here."""
    from raven.home import get_config_path

    return Path(get_config_path()).expanduser().resolve().parent


def _fold(spelling: str | Path) -> str:
    """A spelling as a case- and normalization-insensitive volume compares it."""
    return unicodedata.normalize("NFC", os.fspath(spelling)).casefold()


def _same(a: Path, b: Path) -> bool:
    """Whether two resolved paths name one directory, asked of the filesystem too.

    ``resolve()`` neither folds case nor normalizes Unicode, and macOS's
    default volume does both, so two spellings that differ only that way pass
    ``==`` while naming one directory. Only such a pair is worth a ``stat``:
    every other answers from the strings, which keeps the per-turn check off a
    slow network filesystem.
    """
    if a == b:
        return True
    if _fold(a) != _fold(b):
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _respelled(root: Path, parts: tuple[str, ...]) -> Path:
    """``root`` joined with ``parts``, each spelled the way its entry on disk is.

    A path built from a typed spelling names the right file to the filesystem
    and the wrong one to git and to a directory walk, which compare names as
    stored. A part with no entry yet keeps the spelling it was given.
    """
    path = root
    for part in parts:
        try:
            names = os.listdir(path)
        except OSError:
            names = []
        if part not in names:
            folded = _fold(part)
            part = next((name for name in names if _fold(name) == folded and _same(path / name, path / part)), part)
        path = path / part
    return path


def _anchor(path: Path, root: Path) -> Path | None:
    """``path`` or the ancestor of it that is ``root``, or ``None``."""
    return next((candidate for candidate in (path, *path.parents) if _same(candidate, root)), None)


def _contains(path: Path, target: Path) -> bool:
    """Whether ``path`` is a strict ancestor of ``target``."""
    return any(_same(path, parent) for parent in target.parents)


def _inside_state(path: Path, homes: tuple[Path, ...]) -> str | None:
    """Why ``path`` (resolved) lies in Raven's own state, or ``None`` when it does not.

    Containing that state is a separate question: a checkpoint can leave a
    contained tree out, but a root inside one has nothing else to snapshot.
    """
    for home in homes:
        # Agent home itself is rejected for the same reason its subtrees are, and
        # more strongly: from there every protected subtree is one relative path
        # away, so an ordinary relative write lands on the agent's own memory.
        if _same(path, home):
            return f"working directory must not be the agent home directory itself ({home})"
        for subtree in _PROTECTED_SUBTREES:
            candidate = home / subtree
            if _anchor(path, candidate) is not None:
                return f"working directory must not be inside the agent's {subtree} tree ({candidate})"
    for root in _state_roots():
        if _same(path, root):
            return f"working directory must not be Raven's home directory itself ({root})"
        anchor = _anchor(path.parent, root)
        if anchor is not None:
            entry = path.relative_to(anchor).parts[0]
            if _fold(entry) in _STATE_ENTRIES:
                return f"working directory must not be inside Raven's own state ({root / entry})"
    data = _data_dir()
    anchor = _anchor(path.parent, data)
    if anchor is not None:
        entry = path.relative_to(anchor).parts[0]
        if _fold(entry) in _DATA_DIR_STATE:
            return f"working directory must not be inside Raven's own state ({data / entry})"
    return None


def validate_override(value: str | Path, agent_home: Path) -> Path:
    """Check a user-supplied working directory, returning it resolved."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"working directory must be an absolute path, got {value!r}")
    resolved = path.resolve()
    home = Path(agent_home).expanduser().resolve()
    # An ancestor of agent home is worse than agent home itself (refused in
    # `_inside_state`). `~/.raven` is not merely agent home's parent,
    # it is the instance data directory -- config.json (provider keys), oauth/
    # (provider tokens), cron/, sentinel/, logs/. The per-turn checkpoint runs
    # `add -A` over the working directory, and the `.raven/` default exclude
    # cannot help when `.raven` *is* the work-tree root, so pointing a session
    # here would commit every credential into a shadow git repo.
    if _contains(resolved, home):
        raise ValueError(f"working directory must not contain the agent home directory ({home})")
    raven_root = _raven_home()
    if _contains(resolved, raven_root):
        raise ValueError(f"working directory must not contain Raven's home directory ({raven_root})")
    reason = _inside_state(resolved, (home,))
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
    return _inside_state(Path(root).expanduser().resolve(), homes)


def state_within(root: Path, agent_home: Path | None = None) -> tuple[Path, ...]:
    """Raven's own state lying strictly inside ``root``, as resolved paths.

    What a snapshot or a listing of ``root`` leaves out: Raven's home, agent
    home, the configuration file in use and what Raven keeps beside it (the
    data directory `--config` can move into a project), matched by where they
    are rather than by being named ``.raven``. A path inside another of them
    is folded into it.

    Containment is asked of the filesystem, as the refusal side asks it: a
    config path typed in another case or Unicode normalization names the same
    tree on macOS. Each path comes back spelled under ``root`` the way its
    entries are on disk, since the excludes, pathspecs and listing built from
    it compare names as stored.
    """
    from raven.home import get_config_path

    root = Path(root).expanduser().resolve()
    data = _data_dir()
    candidates = [
        *_state_roots(),
        Path(get_config_path()).expanduser().resolve(),
        *(data / name for name in sorted(_DATA_DIR_STATE)),
    ]
    if agent_home is not None:
        candidates.append(Path(agent_home).expanduser().resolve())
    inside = set()
    for path in candidates:
        anchor = _anchor(path.parent, root)
        if anchor is not None:
            inside.add(_respelled(root, path.relative_to(anchor).parts))
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
