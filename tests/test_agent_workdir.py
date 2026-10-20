"""Unit tests for the session working-directory resolver."""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from raven.agent.workdir import (
    WorkdirPolicy,
    WorkdirResolver,
    bind,
    current,
    validate_override,
)


class _StubSessions:
    """Stands in for SessionManager: only ``get_or_create`` is used."""

    def __init__(self, metadata_by_key: dict[str, dict]) -> None:
        self._metadata_by_key = metadata_by_key

    def get_or_create(self, key: str):
        return SimpleNamespace(metadata=self._metadata_by_key.get(key, {}))


def test_every_chat_on_a_channel_shares_one_directory(tmp_path: Path) -> None:
    """The channel is the unit a user configures, so it is the unit of
    isolation; two chats on it are the same kind of work."""
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
    )
    assert resolver.resolve("web:2fb833e6") == tmp_path / "chanwork" / "web"
    assert resolver.resolve("web:other") == tmp_path / "chanwork" / "web"
    assert resolver.resolve("qq:2fb833e6") == tmp_path / "chanwork" / "qq"


def test_a_configured_channel_workspace_wins_over_the_default_root(tmp_path: Path) -> None:
    pinned = tmp_path / "elsewhere"
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
        channel_workspaces={"qq": str(pinned), "web": ""},
    )
    assert resolver.resolve("qq:123", create=False) == pinned
    # An empty configured value is not a configured value.
    assert resolver.resolve("web:123", create=False) == tmp_path / "chanwork" / "web"


def test_per_session_creates_the_directory(tmp_path: Path) -> None:
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
    )
    assert resolver.resolve("web:abc").is_dir()


def test_colonless_key_gets_a_placeholder_chat_id(tmp_path: Path) -> None:
    """``heartbeat`` is the one session key in the codebase with no colon."""
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
    )
    assert resolver.resolve("heartbeat") == tmp_path / "chanwork" / "heartbeat"


def test_chat_id_with_separators_is_sanitised(tmp_path: Path) -> None:
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
    )
    resolved = resolver.resolve("whatsapp:../../etc/passwd")
    assert (tmp_path / "chanwork") in resolved.parents


@pytest.mark.parametrize(
    "session_key",
    ["web:..", "web:.", "..:chat", ".:chat", "..:.."],
)
def test_dot_only_key_segments_cannot_escape_the_session_root(tmp_path: Path, session_key: str) -> None:
    """``safe_filename`` passes ``.`` and ``..`` through untouched.

    That is harmless for ``sessions/<chat_id>.jsonl`` (the suffix makes an
    ordinary name) but not for a bare directory segment: ``ws/web/..`` resolves
    to ``ws``, so every session with such a key would share one directory and
    one shadow repo -- the exact isolation this layout exists to provide. The
    sibling case with separators is not enough to catch this: ``/`` becomes
    ``_``, which leaves one inert segment, while a bare ``..`` never had one.
    """
    root = tmp_path / "chanwork"
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=root,
    )
    resolved = resolver.resolve(session_key, create=False)
    assert root in resolved.parents
    assert resolved != root
    assert ".." not in resolved.parts


def test_launch_dir_ignores_the_session_key(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    resolver = WorkdirResolver(
        WorkdirPolicy.LAUNCH_DIR,
        agent_home=tmp_path / "home",
        launch_dir=project,
    )
    assert resolver.resolve("cli:one") == project
    assert resolver.resolve("cli:two") == project


def test_persisted_override_beats_the_policy_default(tmp_path: Path) -> None:
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
        sessions=_StubSessions({"web:abc": {"workdir": str(pinned)}}),
    )
    assert resolver.resolve("web:abc") == pinned


def test_explicit_workdir_beats_the_persisted_override(tmp_path: Path) -> None:
    """A flag typed for this run wins over anything on disk."""
    pinned = tmp_path / "pinned"
    flagged = tmp_path / "flagged"
    pinned.mkdir()
    flagged.mkdir()
    resolver = WorkdirResolver(
        WorkdirPolicy.LAUNCH_DIR,
        agent_home=tmp_path,
        launch_dir=tmp_path,
        explicit_workdir=flagged,
        sessions=_StubSessions({"cli:abc": {"workdir": str(pinned)}}),
    )
    assert resolver.resolve("cli:abc") == flagged


def test_override_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        validate_override("relative/path", tmp_path)


@pytest.mark.parametrize("subtree", ["user_memory", "skills", "sessions"])
def test_override_cannot_point_into_agent_home_internals(tmp_path: Path, subtree: str) -> None:
    with pytest.raises(ValueError, match=subtree):
        validate_override(tmp_path / subtree / "nested", tmp_path)


def test_override_cannot_be_agent_home_itself(tmp_path: Path) -> None:
    """Working in agent home puts every protected subtree one relative path away."""
    with pytest.raises(ValueError, match="agent home"):
        validate_override(tmp_path, tmp_path)


def test_override_cannot_be_an_ancestor_of_agent_home(tmp_path: Path) -> None:
    """An ancestor is refused for a stronger reason than agent home itself.

    The realistic ancestor is `~/.raven`, which is not merely agent home's
    parent but the instance data directory: `config.json` (provider keys),
    `oauth/`, `cron/`, `logs/`. The per-turn checkpoint runs `add -A` over the
    working directory, and the `.raven/` default exclude cannot help when
    `.raven` is itself the work-tree root, so working there commits every
    credential into a shadow repo. `user_memory/` and `skills/` are also back
    within relative reach, one level deeper than from agent home.
    """
    home = tmp_path / "nested" / "home"
    home.mkdir(parents=True)
    with pytest.raises(ValueError, match="agent home"):
        validate_override(tmp_path, home)
    with pytest.raises(ValueError, match="agent home"):
        validate_override(tmp_path / "nested", home)


def test_override_may_be_a_sibling_of_agent_home(tmp_path: Path) -> None:
    """Containment is the test, not proximity: a directory next to agent home
    holds none of it, so `add -A` there captures nothing protected."""
    home = tmp_path / "nested" / "home"
    home.mkdir(parents=True)
    sibling = tmp_path / "nested" / "project"
    sibling.mkdir()
    assert validate_override(sibling, home) == sibling


def test_binding_is_scoped(tmp_path: Path) -> None:
    assert current() is None
    with bind(tmp_path):
        assert current() == tmp_path
    assert current() is None


def test_repoint_moves_the_running_turns_binding_and_only_that(tmp_path: Path) -> None:
    """``repoint`` serves the ``rebind_workdir`` grant: the very next read in
    the same task sees the new root, and the enclosing ``bind``'s finally
    still resets at turn end -- nothing leaks past the turn."""
    from raven.agent.workdir import repoint

    first = tmp_path / "first"
    moved = tmp_path / "moved"
    assert current() is None
    with bind(first):
        repoint(moved)
        assert current() == moved
    assert current() is None


def test_mount_root_covers_every_per_session_directory(tmp_path: Path) -> None:
    """One sandbox mount has to contain every directory the process can produce."""
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=tmp_path,
        session_root=tmp_path / "chanwork",
    )
    root = resolver.mount_root()
    assert root == tmp_path / "chanwork"
    assert root in resolver.resolve("web:abc").parents


def test_mount_root_is_the_launch_dir_for_terminal_entrypoints(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    resolver = WorkdirResolver(
        WorkdirPolicy.LAUNCH_DIR,
        agent_home=tmp_path / "home",
        launch_dir=project,
    )
    assert resolver.mount_root() == project


@pytest.mark.parametrize(
    "configured",
    [
        "",  # agent home itself, filled in by the test
        "skills/nested",
        "relative/path",
        "..",
    ],
)
def test_a_bad_channel_workspace_is_dropped_not_obeyed(tmp_path: Path, configured: str, caplog) -> None:
    """`channels.<name>.workspace` reaches the resolver from a config file and
    from the web UI's channels form, and nothing else validates it. Left
    unchecked, `channels.x.workspace = <agent home>` is accepted and the
    per-turn `add -A` then runs over user_memory/ and sessions/ -- the shape
    every other entry point already rejects.
    """
    home = tmp_path / "home"
    home.mkdir()
    value = str(home) if not configured else str(home / configured) if configured != "relative/path" else configured

    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=home,
        session_root=tmp_path / "chanwork",
        channel_workspaces={"qq": value},
    )

    assert resolver.resolve("qq:1", create=False) == tmp_path / "chanwork" / "qq"


def test_a_valid_channel_workspace_is_still_honoured(tmp_path: Path) -> None:
    """The guard must reject shapes, not the feature."""
    home = tmp_path / "home"
    pinned = tmp_path / "elsewhere"
    home.mkdir()
    pinned.mkdir()
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=home,
        session_root=tmp_path / "chanwork",
        channel_workspaces={"qq": str(pinned)},
    )

    assert resolver.resolve("qq:1", create=False) == pinned


def test_default_channel_root_never_lands_in_the_system_tmp(tmp_path: Path) -> None:
    """A home directly under the filesystem root is the case a plain sibling
    gets wrong: `--home /workspace` (a common container mount) would put every
    channel in the shared, world-writable, reboot-cleared system /tmp.
    """
    from raven.agent.workdir import default_channel_root

    assert default_channel_root(Path("/workspace")) == Path("/workspace-tmp")
    assert default_channel_root(Path("/")) == Path("/raven-tmp")
    # The shape everyone actually runs is unchanged.
    assert default_channel_root(Path("/root/.raven/workspace")) == Path("/root/.raven/tmp")
    assert default_channel_root(tmp_path / "a" / "home") == tmp_path / "a" / "tmp"


@pytest.fixture
def instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An instance directory deliberately not named `.raven`, selected through `RAVEN_HOME`."""
    root = tmp_path / "raven-data"
    (root / "workspace").mkdir(parents=True)
    monkeypatch.setenv("RAVEN_HOME", str(root))
    return root


@pytest.mark.parametrize(
    "entry",
    ["oauth", "cron/jobs", "credentials/mcp", "logs", "plugins/demo", "agents/raven-code", "subagent_sessions/x/acp"],
)
def test_override_cannot_point_into_instance_state(instance: Path, entry: str) -> None:
    """Agent home's three subtrees are not the whole of Raven's state: the
    instance directory beside it holds provider tokens, MCP credentials, cron
    jobs and the engines' homes, and a checkpoint rooted in any of them would
    commit what it holds."""
    target = instance / entry
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="Raven's"):
        validate_override(target, instance / "workspace")


@pytest.mark.parametrize("entry", ["tmp", "tmp/web", "workspace/projects/demo", "my-projects/demo"])
def test_instance_work_areas_stay_usable(instance: Path, entry: str) -> None:
    """The per-channel scratch root and agent home's own work areas live in
    the instance directory too, and are what the gateway hands out; a
    directory of the user's own there is not Raven's state either."""
    target = instance / entry
    target.mkdir(parents=True)
    assert validate_override(target, instance / "workspace") == target.resolve()


def test_a_configured_agent_home_inside_the_instance_keeps_its_work_areas(instance: Path) -> None:
    home = instance / "homes" / "main"
    project = instance / "homes" / "tmp" / "web"
    project.mkdir(parents=True)
    home.mkdir(parents=True)
    assert validate_override(project, home) == project.resolve()
    with pytest.raises(ValueError, match="Raven's"):
        validate_override(instance / "oauth", home)


def test_override_cannot_contain_the_instance_when_agent_home_lives_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With agent home moved out, only the instance rule stands between a
    project that holds `RAVEN_HOME` and a checkpoint of its tokens."""
    project = tmp_path / "project"
    state = project / ".state"
    (state / "oauth").mkdir(parents=True)
    home = tmp_path / "agent"
    home.mkdir()
    monkeypatch.setenv("RAVEN_HOME", str(state))

    for refused in (project, state, state / "oauth"):
        with pytest.raises(ValueError, match="Raven's"):
            validate_override(refused, home)
    assert validate_override(tmp_path / "agent-sibling", home) == (tmp_path / "agent-sibling").resolve()


def test_instance_state_is_judged_through_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = tmp_path / "real-instance"
    (real / "workspace").mkdir(parents=True)
    (real / "oauth").mkdir()
    alias = tmp_path / "alias-instance"
    alias.symlink_to(real, target_is_directory=True)
    link = tmp_path / "project-link"
    link.symlink_to(real / "oauth", target_is_directory=True)
    monkeypatch.setenv("RAVEN_HOME", str(alias))

    with pytest.raises(ValueError, match="Raven's"):
        validate_override(real / "oauth", alias / "workspace")
    with pytest.raises(ValueError, match="Raven's"):
        validate_override(link, alias / "workspace")


def test_a_moved_config_puts_its_state_beside_it(tmp_path: Path) -> None:
    """`--config` moves the data directory with the config file: the MCP
    credentials and the cron store follow it, and must be refused there too.
    The directory itself is often the user's project, so it stays usable."""
    from raven.home import set_config_path

    data = tmp_path / "project"
    for rel in ("credentials/mcp", "cron", "render", "src"):
        (data / rel).mkdir(parents=True)
    set_config_path(data / "config.json")
    home = tmp_path / "agent"
    home.mkdir()

    for refused in ("credentials/mcp", "cron", "render"):
        with pytest.raises(ValueError, match="Raven's own state"):
            validate_override(data / refused, home)
    assert validate_override(data, home) == data.resolve()
    assert validate_override(data / "src", home) == (data / "src").resolve()


def test_the_default_home_stays_protected_when_raven_home_moves(instance: Path) -> None:
    """The browser profile and the routing caches are written under `~/.raven`
    whatever `RAVEN_HOME` says, so moving the home does not unprotect them."""
    profile = Path.home() / ".raven" / "browser-profile"
    profile.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Raven's"):
        validate_override(profile, instance / "workspace")


def test_instance_state_is_judged_by_the_filesystem_not_the_spelling(
    instance: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a case-insensitive volume (macOS by default) `resolve()` keeps the
    case it was given, so a path spelled in another case names the same tree
    and must be refused for it. Simulated here by a case-folding `samefile`."""
    import os

    from raven.agent import workdir

    monkeypatch.setattr(workdir.os.path, "samefile", lambda a, b: os.fspath(a).casefold() == os.fspath(b).casefold())
    spelled = Path(str(instance).upper())

    with pytest.raises(ValueError, match="Raven's"):
        validate_override(spelled / "OAUTH" / "nested", instance / "workspace")
    with pytest.raises(ValueError, match="agent's sessions"):
        validate_override(spelled / "WORKSPACE" / "SESSIONS", instance / "workspace")
    assert validate_override(spelled / "MY-PROJECTS", instance / "workspace") == spelled / "MY-PROJECTS"


def test_instance_state_is_judged_whatever_the_unicode_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A volume that ignores normalization (macOS) names one directory by its
    NFC and NFD spellings, so the refusal must fold both, not only case.
    Simulated with a `samefile` that folds alike; not a native run."""
    import os
    import unicodedata

    from raven.agent import workdir

    def fold(path: object) -> str:
        return unicodedata.normalize("NFC", os.fspath(path)).casefold()

    monkeypatch.setattr(workdir.os.path, "samefile", lambda a, b: fold(a) == fold(b))
    nfc = tmp_path / unicodedata.normalize("NFC", "caf\u00e9-home")
    (nfc / "workspace").mkdir(parents=True)
    (nfc / "oauth").mkdir()
    monkeypatch.setenv("RAVEN_HOME", str(nfc))
    nfd = tmp_path / unicodedata.normalize("NFD", "caf\u00e9-home")

    with pytest.raises(ValueError, match="Raven's"):
        validate_override(nfd / "oauth", nfc / "workspace")
    with pytest.raises(ValueError, match="agent home|Raven's"):
        validate_override(nfd, nfc / "workspace")


def test_a_channel_workspace_in_instance_state_falls_back(instance: Path) -> None:
    resolver = WorkdirResolver(
        WorkdirPolicy.PER_CHANNEL,
        agent_home=instance / "workspace",
        channel_workspaces={"qq": str(instance / "oauth"), "web": str(instance / "channels" / "web")},
    )

    assert resolver.resolve("qq:1", create=False) == instance / "tmp" / "qq"
    assert resolver.resolve("web:1", create=False) == (instance / "channels" / "web").resolve()


# Entries directly under Raven's home that are files, or that hold work rather
# than state; every other entry the code creates there must be a state entry.
_HOME_FILES = {
    "beta.json",
    "config.json",
    "connections.json",
    "env",
    "everos-server.lock",
    "everos-server.pid",
    "gateway.lock",
    "import_state.json",
    "redaction_history.json",
    "serve.json",
    "upgrade.json",
    "web.json",
    "web.log",
}
_HOME_WORK = {"workspace"}


def _module_strings(tree: ast.Module) -> dict[str, object]:
    """A module's literal constants, and the ones it imports from another module."""
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("raven.contracts"):
            module = importlib.import_module(node.module)
            for alias in node.names:
                values[alias.asname or alias.name] = getattr(module, alias.name, None)
    return values


def _scopes(tree: ast.Module) -> list[ast.AST]:
    """The module and each function in it, the scopes a local name is bound in."""
    return [tree, *(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))]


def _entries_created_under_the_home() -> dict[str, dict[str, set[str]]]:
    """Every top-level name the runtime puts under Raven's home, by where it is anchored.

    ``"home"`` is ``raven_home()`` or ``Path.home() / ".raven"``; ``"data"`` is
    the data directory beside the config file (``get_data_dir()`` and
    ``get_runtime_subdir()``), which ``--config`` moves away from the home.
    Reached directly, through ``joinpath``, or through a local name bound to
    one of those roots first; in ``raven/`` and in the ``plugins-dist``
    packages alike.
    """
    found: dict[str, dict[str, set[str]]] = {}
    sources = [*Path("raven").rglob("*.py"), *Path("plugins-dist").rglob("*.py")]
    for path in sorted(p for p in sources if "tests" not in p.parts):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _module_strings(tree)
        params = {
            arg.arg
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for arg in node.args.args
        }

        def name_of(expr: ast.expr) -> str:
            if isinstance(expr, ast.Starred):
                held = constants.get(expr.value.id) if isinstance(expr.value, ast.Name) else None
                value = held[0] if isinstance(held, tuple) and held else None
                return value if isinstance(value, str) else f"<unresolved {ast.unparse(expr)}>"
            try:
                value = ast.literal_eval(expr)
            except ValueError:
                value = None
                if isinstance(expr, ast.Name):
                    value = constants.get(expr.id)
                elif isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
                    held = constants.get(expr.value.id)
                    index = ast.literal_eval(expr.slice)
                    value = held[index] if isinstance(held, tuple) else None
            return value if isinstance(value, str) else f"<unresolved {ast.unparse(expr)}>"

        def called(expr: ast.expr) -> str | None:
            if not isinstance(expr, ast.Call) or expr.args:
                return None
            func = expr.func
            return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)

        def anchor(expr: ast.expr, bound: dict[str, str]) -> str | None:
            if called(expr) == "raven_home":
                return "home"
            if called(expr) == "get_data_dir":
                return "data"
            if (
                isinstance(expr, ast.BinOp)
                and isinstance(expr.op, ast.Div)
                and called(expr.left) == "home"
                and name_of(expr.right) == ".raven"
            ):
                return "home"
            if isinstance(expr, ast.Name):
                return bound.get(expr.id)
            return None

        def record(name: str, where: str, node: ast.AST) -> None:
            found.setdefault(name, {}).setdefault(where, set()).add(f"{path}:{node.lineno}")

        for scope in _scopes(tree):
            bound: dict[str, str] = {}
            for node in ast.walk(scope):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    where = anchor(node.value, {})
                    if where is not None:
                        bound[node.targets[0].id] = where
            for node in ast.walk(scope):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    where = anchor(node.left, bound)
                    if where is not None and not (isinstance(node.right, ast.Name) and node.right.id in params):
                        record(name_of(node.right), where, node)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
                    if node.func.attr == "joinpath":
                        where = anchor(node.func.value, bound)
                        if where is not None:
                            record(name_of(node.args[0]), where, node)
                if isinstance(node, ast.Call) and node.args:
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    argument = node.args[0]
                    if name == "get_runtime_subdir" and not (isinstance(argument, ast.Name) and argument.id in params):
                        record(name_of(argument), "data", node)
    return found


def test_every_directory_raven_creates_in_its_home_is_known_state() -> None:
    """The state lists are the boundary, so a directory the runtime starts
    writing under its home must be named in them, or chosen as a file or work
    area here; and an entry nothing creates any more must leave the list. The
    data-directory list is exactly what is anchored there, since `--config`
    puts that directory in a user's project and over-matching would hide the
    project's own files from the checkpoint."""
    from raven.agent.workdir import _DATA_DIR_STATE, _STATE_ENTRIES

    found = _entries_created_under_the_home()

    unclassified = {
        name: sites for name, sites in found.items() if name not in _STATE_ENTRIES | _HOME_FILES | _HOME_WORK
    }
    assert unclassified == {}
    assert _STATE_ENTRIES - found.keys() == set()
    assert _STATE_ENTRIES.isdisjoint(_HOME_FILES | _HOME_WORK)
    assert {name for name, sites in found.items() if "data" in sites} == _DATA_DIR_STATE


def test_the_scan_sees_each_way_a_path_reaches_the_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The completeness test is only as good as its scan: it must see a name
    joined with `joinpath`, one joined after the root was bound to a local
    name, and one in a plugin package."""
    (tmp_path / "raven").mkdir()
    (tmp_path / "plugins-dist" / "demo").mkdir(parents=True)
    (tmp_path / "raven" / "a.py").write_text(
        "RUNS = ('alpha', 'runs')\n"
        "def f():\n    return raven_home().joinpath(*RUNS)\n"
        "def g():\n    root = raven_home()\n    return root / 'beta'\n",
        encoding="utf-8",
    )
    (tmp_path / "plugins-dist" / "demo" / "b.py").write_text(
        "def h():\n    return get_runtime_subdir('gamma')\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    found = _entries_created_under_the_home()

    assert {name: sorted(sites) for name, sites in found.items()} == {
        "alpha": ["home"],
        "beta": ["home"],
        "gamma": ["data"],
    }
