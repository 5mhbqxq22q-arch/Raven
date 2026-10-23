"""Read a pre-merge ``action.checks`` block as the ``judge`` that replaced it.

A Harness saved before the judgement seat became one surface carries a
declarative rule table, and sometimes a judge beside it. Both still have to
load: an artifact is durable, and a reader that refused one would take a
Persona out of service for a field the writer never chose.

The table is translated rather than kept, because keeping it would be the
second surface again. Every operator it had maps onto something ``judge``
already reads -- ``prior`` is the call history the rules matched on -- so the
translation is total rather than best-effort, and the generated source passes
the same gate a model-written judge does.
"""

from __future__ import annotations

import ast
from typing import Any

from loguru import logger

from raven.agent.subagent.charter_code import ALLOWED_CALLS, MAX_SOURCE_CHARS, compile_function

MIGRATED_NOTE = "migrated from action.checks"


def _literal(value: str) -> str:
    """One string as a Python literal the gate admits."""
    return repr(str(value))


def _when_clause(when: dict[str, Any]) -> str:
    """The ``when`` narrowing, as conditions on ``params``."""
    return "".join(
        f" and str(params.get({_literal(key)}, '')) == {_literal(value)}" for key, value in sorted(when.items())
    )


def _grouped(rules: list[Any]) -> dict[tuple[str, str], dict[str, list[tuple[str, str]]]]:
    """Rules by (tool, when), then by operator, as (value, message) pairs.

    Grouped before anything is written because a table is mostly the same few
    shapes repeated: thirty-two forbidden fragments for one tool are one list
    and one comprehension, not thirty-two blocks. The source has to stay under
    the ceiling a charter carries, and a translation that cannot be dispatched
    has preserved nothing.
    """
    out: dict[tuple[str, str], dict[str, list[tuple[str, str]]]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        tool = str(rule.get("tool") or "")
        if not tool:
            continue
        when = rule.get("when") if isinstance(rule.get("when"), dict) else {}
        key = (tool, _when_clause(when))
        by_operator = out.setdefault(key, {})
        own = str(rule.get("message") or "")
        prefix = str(rule.get("pathPrefix") or rule.get("path_prefix") or "")
        if prefix:
            by_operator.setdefault("prefix", []).append((prefix, own))
        forbid = str(rule.get("forbid") or "")
        if forbid:
            by_operator.setdefault("forbid", []).append((forbid, own))
        earlier = str(rule.get("requiresPrior") or rule.get("requires_prior") or "")
        if earlier:
            message = str(rule.get("message") or f"{tool}: run {earlier} first")
            match_param = str(rule.get("matchParam") or rule.get("match_param") or "")
            by_operator.setdefault("prior", []).append((f"{earlier}\x00{match_param}", message))
    return out


def _values(values: list[tuple[str, str]]) -> str:
    return "[" + ", ".join(_literal(value) for value, _ in values) + "]"


def _pairs(values: list[tuple[str, str]]) -> str:
    return "[" + ", ".join(f"({_literal(value)}, {_literal(message)})" for value, message in values) + "]"


def _operator_lines(values: list[tuple[str, str]], test: str, default: str) -> list[str]:
    """One operator's rules, with default messages built at run time.

    A default message quotes the value it is about, so writing both would put
    every forbidden path in the source twice -- once as the needle and once
    inside its own sentence, escaped again. The sentence is cheap to build
    where it is needed, and the source then grows with the rules rather than
    with the rules squared.
    """
    plain = [pair for pair in values if not pair[1]]
    custom = [pair for pair in values if pair[1]]
    lines: list[str] = []
    if plain:
        lines.append(f"        refusals = refusals + [{default} for v in {_values(plain)} if {test}]")
    if custom:
        lines.append(f"        refusals = refusals + [m for v, m in {_pairs(custom)} if {test}]")
    return lines


def judge_from_rules(rules: list[Any]) -> str:
    """The rule table as one ``judge``, or "" when it refuses nothing."""
    blocks: list[str] = []
    for (tool, when), by_operator in _grouped(rules).items():
        body: list[str] = []
        if prefixes := by_operator.get("prefix"):
            body.append("        path = str(params.get('path', '') or params.get('file_path', ''))")
            body += _operator_lines(
                prefixes,
                "path and not path.startswith(v)",
                f"{_literal(tool + ': path must start with ')} + v",
            )
        if forbidden := by_operator.get("forbid"):
            body.append("        seen = ' '.join([str(v) for v in params.values()])")
            body += _operator_lines(
                forbidden,
                "v in seen",
                f"{_literal(tool + ': this call may not contain ')} + {_literal(chr(39))} + v + {_literal(chr(39))}",
            )
        for encoded, message in by_operator.get("prior", []):
            earlier, _, match_param = encoded.partition("\x00")
            same = f" and p.get({_literal(match_param)}) == params.get({_literal(match_param)})" if match_param else ""
            body += [
                f"        if not [p for n, p in prior if n == {_literal(earlier)}{same}]:",
                f"            refusals = refusals + [{_literal(message)}]",
            ]
        if body:
            blocks += [f"    if name == {_literal(tool)}{when}:", *body]
    if not blocks:
        return ""
    return "\n".join(["def judge(name, params, prior):", "    refusals = []", *blocks, "    return refusals"])


_RULE_ARGS = ("name", "params", "prior")


class _RenameNames(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str], prefix: str) -> None:
        self.mapping = mapping
        self.prefix = prefix

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        elif node.id not in ALLOWED_CALLS:
            node.id = self.prefix + node.id
        return node


class _ReturnsWithRules(ast.NodeTransformer):
    """Every ``return`` of the hand-written judge, with the rules' refusals ahead of it.

    A nested definition keeps its own returns: they answer whoever calls it,
    not the dispatch.
    """

    def __init__(self, refusals: str, answer: str) -> None:
        self.refusals = refusals
        self.answer = answer

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return node

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        held = ast.Assign(targets=[ast.Name(self.answer, ast.Store())], value=node.value or ast.Constant(None))
        return [held, _union_return(self.refusals, self.answer)]


def _union_return(refusals: str, answer: str) -> ast.stmt:
    # The same reading ``charter.judge`` gives a single judge's answer: a string
    # is one refusal, a list or tuple is several, anything else is silence.
    return ast.parse(
        f"return {refusals} + ([{answer}] if isinstance({answer}, str) "
        f"else list({answer}) if isinstance({answer}, (list, tuple)) else [])"
    ).body[0]


def _free_prefix(tree: ast.AST) -> str:
    taken = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    taken |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    prefix = "migrated_"
    while any(name.startswith(prefix) for name in taken):
        prefix += "x_"
    return prefix


def merged_judge(code: str, rules_source: str) -> str:
    """One ``judge`` that refuses what either the hand-written judge or the table refused.

    The pre-merge runtime ran the table and then the code, and a call either
    refused did not run. Neither half may simply call the other: the gate
    admits calls only to a named handful of builtins. So the table's checks
    are written into the hand-written judge's own body, ahead of it, under
    names the hand-written code does not use, and each of its returns hands
    back the table's refusals first.

    One difference stays. The old runtime still applied the table when the
    code raised; here a raise anywhere silences the whole judge, because the
    gate admits no ``try``. The table runs first, so only a judge that raises
    on a call the table also refuses loses that refusal.
    """
    tree = ast.parse(code)
    hand = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "judge")
    prefix = _free_prefix(tree)
    refusals, answer = prefix + "refusals", prefix + "answer"
    rules_def = ast.parse(rules_source).body[0]
    renamer = _RenameNames(dict(zip(_RULE_ARGS, (arg.arg for arg in hand.args.args))), prefix)
    prologue = [renamer.visit(stmt) for stmt in rules_def.body[:-1]]  # type: ignore[attr-defined]
    returns = _ReturnsWithRules(refusals, answer)
    body: list[ast.stmt] = []
    for stmt in hand.body:
        rewritten = returns.visit(stmt)
        body.extend(rewritten if isinstance(rewritten, list) else [rewritten])
    docstring = body[:1] if ast.get_docstring(hand) is not None else []
    # Falling off the end was silence from the hand-written half, which leaves the table's answer.
    fall_through = ast.parse(f"return {refusals}").body[0]
    hand.body = [*docstring, *prologue, *body[len(docstring) :], fall_through]
    return ast.unparse(ast.fix_missing_locations(tree))


def _judge_for(code: str, rules: list[Any]) -> str:
    """The judge a ``checks`` block meant: its code, its rules, or both at once."""
    generated = judge_from_rules(rules)
    count = len([rule for rule in rules if isinstance(rule, dict)])
    if code and generated:
        try:
            compile_function(code, "judge")
        except Exception as exc:  # noqa: BLE001 - a refused judge was silence before the merge too
            # The old runtime ran the table and treated a judge it could not
            # compile as having no opinion. The table alone is that behaviour.
            logger.warning(
                "playbook: action.checks.code was refused by the gate ({}); keeping its {} rule(s) "
                "without it, as the pre-merge runtime did",
                exc,
                count,
            )
            code = ""
        else:
            generated = merged_judge(code, generated)
    elif code:
        return code
    # Refused here rather than carried to a dispatch that would drop it. A
    # translation over the ceiling cannot be sent, and a seat that loaded
    # without the rules it was written with enforces nothing while reading as
    # if it does -- the failure the stored artifact is least able to notice.
    # The error names what to do.
    if len(generated) > MAX_SOURCE_CHARS:
        written = " and its hand-written judge" if code else ""
        raise ValueError(
            f"action.checks carries {count} rules{written} that translate "
            f"to {len(generated)} characters of judge, over the {MAX_SOURCE_CHARS} a charter may carry. "
            "State them as one judge under action.functions, which can say the same thing in fewer "
            "characters than one block per rule."
        )
    return generated


def migrated_action(action: dict[str, Any]) -> dict[str, Any]:
    """``action`` with any ``checks`` block read as ``functions.judge``.

    A block with both ``code`` and ``rules`` becomes one judge that refuses
    what either refused, as the pre-merge runtime did; see :func:`merged_judge`.
    """
    checks = action.get("checks")
    if not isinstance(checks, dict):
        return action
    migrated = {key: value for key, value in action.items() if key != "checks"}
    functions = dict(migrated.get("functions") or {})
    code = str(checks.get("code") or "").strip()
    rules = checks.get("rules") if isinstance(checks.get("rules"), list) else []
    if "judge" not in functions and (judge := _judge_for(code, rules)):
        functions["judge"] = judge
    if functions:
        migrated["functions"] = functions
    return migrated


__all__ = ["MIGRATED_NOTE", "judge_from_rules", "merged_judge", "migrated_action"]
