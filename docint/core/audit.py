"""Data-protection audit trail: how docint records what it does with personal data.

``docint.utils.audit_event`` defines what an audit line is and
``docint.utils.audit_runtime`` how every app writes and reads the shared
parts (stream, purpose header, request ids). This module wires both into
docint: one :class:`Auditor` per process, the request gate every audited
route goes through, the per-operation record, and the command-line session.
What each route and command records is listed in ``docs/audit.md``.

Fail closed: when an event cannot be written the operation fails with it.
:meth:`Operation.record` raises, a request answers the generic 500 without
its data, a job is marked failed, and a command exits non-zero.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Request
from loguru import logger

from docint import __version__
from docint.core.auth.principal import Principal, resolve_principal
from docint.utils.audit_event import Actor, AiCall, AuditEmitter, AuditWriteError, Purpose, Target
from docint.utils.audit_meter import TransferMeter, metering
from docint.utils.audit_runtime import (
    PURPOSE_HEADER,
    PURPOSE_REQUIRED_STATUS,
    REQUEST_ID_HEADER,
    PurposeError,
    clean_request_id,
    id_chunks,
    open_audit_stream,
    parse_purpose_header,
)
from docint.utils.env_cfg import AuditConfig, PrincipalConfig, load_audit_cli_env, load_audit_env, load_principal_env

#: The compose project name, which the audit store's verifier binds chains to.
SERVICE = "docint"
#: Seconds between heartbeat events on a long-lived chain.
HEARTBEAT_INTERVAL_S = 60.0
#: The system itself, for startup work, the retention sweep and jobs no request started.
SYSTEM_ACTOR = Actor(user=None, via="system")

_PURPOSE_REQUIRED_DETAIL = "A stated purpose is required for this request."
_FORBIDDEN_DETAIL = "Forbidden."


class AuditTrailClosed(RuntimeError):
    """Raised when no audit chain is open to write to: none installed, or it was closed."""


#: What a failed audit write raises. Work that swallows its own errors (an
#: optional ingest stage, say) re-raises these, so it fails rather than
#: carrying on unrecorded.
AUDIT_FAILURES: tuple[type[Exception], ...] = (AuditWriteError, AuditTrailClosed)


class Auditor:
    """One process's audit chain, and the purpose rule it enforces.

    Thread-safe: the emitter serializes writes.
    """

    def __init__(self, emitter: AuditEmitter, *, require_purpose: Collection[str] = frozenset()) -> None:
        """Wrap a started emitter.

        Args:
            emitter (AuditEmitter): The process's emitter, already started.
            require_purpose (Collection[str]): Actions that need a stated purpose.
        """
        self._emitter = emitter
        self.require_purpose = frozenset(require_purpose)

    def needs_purpose(self, actions: Iterable[str]) -> bool:
        """Whether any of an operation's actions needs a stated purpose.

        Args:
            actions (Iterable[str]): Every action the operation may record.

        Returns:
            bool: ``True`` when ``AUDIT_REQUIRE_PURPOSE`` lists one of them.
        """
        return any(action in self.require_purpose for action in actions)

    def emit(
        self,
        action: str,
        operation: str,
        actor: Actor,
        *,
        outcome: str = "ok",
        request_id: str | None = None,
        parent_event_id: str | None = None,
        purpose: Purpose | None = None,
        target: Target | None = None,
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
        ai: AiCall | None = None,
    ) -> dict[str, object]:
        """Write one event, splitting a long id list instead of letting it be cut.

        The first chunk of ``target.ids`` rides the event itself; each further
        chunk gets a continuation event (same action, ``<operation>.ids``)
        pointing at it. ``target.count`` always counts every id.

        Args:
            action (str): The audit action.
            operation (str): The dotted operation name.
            actor (Actor): Who caused it.
            outcome (str): ``ok``, ``denied`` or ``error``.
            request_id (str | None): The gateway's correlation id.
            parent_event_id (str | None): The event this one belongs to.
            purpose (Purpose | None): The stated purpose.
            target (Target | None): What was touched, by reference.
            criteria (Mapping[str, object] | None): The caller's parameters.
            recipient (str | None): Where the data went.
            ai (AiCall | None): Model provenance.

        Returns:
            dict[str, object]: The primary event as written.

        Raises:
            AuditWriteError: When the line could not be written.
            AuditTrailClosed: When the chain is not open.
        """
        try:
            return self._emit(
                action,
                operation,
                actor,
                outcome=outcome,
                request_id=request_id,
                parent_event_id=parent_event_id,
                purpose=purpose,
                target=target,
                criteria=criteria,
                recipient=recipient,
                ai=ai,
            )
        except AuditWriteError:
            raise
        except RuntimeError as exc:
            raise AuditTrailClosed("audit chain is not open") from exc

    def _emit(
        self,
        action: str,
        operation: str,
        actor: Actor,
        *,
        outcome: str,
        request_id: str | None,
        parent_event_id: str | None,
        purpose: Purpose | None,
        target: Target | None,
        criteria: Mapping[str, object] | None,
        recipient: str | None,
        ai: AiCall | None,
    ) -> dict[str, object]:
        """Write an event and its id continuations; see :meth:`emit`.

        Args:
            action (str): The audit action.
            operation (str): The dotted operation name.
            actor (Actor): Who caused it.
            outcome (str): The outcome.
            request_id (str | None): The gateway's correlation id.
            parent_event_id (str | None): The event this one belongs to.
            purpose (Purpose | None): The stated purpose.
            target (Target | None): What was touched.
            criteria (Mapping[str, object] | None): The caller's parameters.
            recipient (str | None): Where the data went.
            ai (AiCall | None): Model provenance.

        Returns:
            dict[str, object]: The primary event as written.
        """
        chunks: list[list[str]] = id_chunks([str(item) for item in target.ids]) if target is not None else []
        primary = target
        if target is not None:
            count = target.count if target.count is not None else len(target.ids)
            first: Sequence[str] = chunks[0] if chunks else ()
            primary = Target(kind=target.kind, ids=first, count=count, container=target.container)
        event = self._emitter.emit(
            action,
            operation,
            actor,
            outcome=outcome,
            request_id=request_id,
            parent_event_id=parent_event_id,
            purpose=purpose,
            target=primary,
            criteria=criteria,
            recipient=recipient,
            ai=ai,
        )
        if target is not None:
            for chunk in chunks[1:]:
                self._emitter.emit(
                    action,
                    f"{operation}.ids",
                    actor,
                    outcome=outcome,
                    request_id=request_id,
                    parent_event_id=str(event["event_id"]),
                    purpose=purpose,
                    target=Target(kind=target.kind, ids=chunk, count=len(chunk), container=target.container),
                    recipient=recipient,
                )
        return event

    def close(self) -> None:
        """Close the chain with its shutdown event; later calls do nothing."""
        self._emitter.close()


_lock = threading.Lock()
_installed: list[Auditor] = []


def install(auditor: Auditor) -> None:
    """Make ``auditor`` the process's audit chain until :func:`uninstall`.

    Installs stack, so a chain installed inside another one (an app started
    in a process that already has one) hands back to it when it ends.

    Args:
        auditor (Auditor): The chain to write to.
    """
    with _lock:
        _installed.append(auditor)


def uninstall(auditor: Auditor) -> None:
    """Stop writing to ``auditor``; the chain installed before it takes over.

    Args:
        auditor (Auditor): A chain passed to :func:`install`.
    """
    with _lock:
        if auditor in _installed:
            _installed.remove(auditor)


def current() -> Auditor:
    """Return the process's audit chain.

    Returns:
        Auditor: The chain installed last.

    Raises:
        AuditTrailClosed: When none is installed, so audited work fails
            rather than going unrecorded.
    """
    with _lock:
        if not _installed:
            raise AuditTrailClosed("audit trail not started")
        return _installed[-1]


def identity_mode(principal_cfg: PrincipalConfig) -> dict[str, object]:
    """State how far identities are trusted, for a chain's ``config.audit`` event.

    The same keys in every app of the federation.

    Args:
        principal_cfg (PrincipalConfig): The principal settings in force.

    Returns:
        dict[str, object]: ``edge_secret_enforced`` and
            ``default_identity_enabled`` (the dev fallback in effect: set, and
            no edge secret).
    """
    return {
        "edge_secret_enforced": principal_cfg.edge_secret is not None,
        "default_identity_enabled": principal_cfg.default_identity is not None and principal_cfg.edge_secret is None,
    }


def open_auditor(
    config: AuditConfig,
    *,
    mode: Mapping[str, object],
    heartbeat_interval_s: float | None = None,
) -> Auditor:
    """Start a new chain on the configured stream and record its operating mode.

    Call it after any ``fork`` (the FastAPI lifespan, a command's start),
    never at import time.

    Args:
        config (AuditConfig): The audit settings.
        mode (Mapping[str, object]): What the ``config.audit`` event states
            beside the purpose rule. Never a secret or a path.
        heartbeat_interval_s (float | None): Override the heartbeat interval;
            ``0`` disables it.

    Returns:
        Auditor: The started chain.

    Raises:
        ValueError: If ``AUDIT_OUTPUT`` names neither mode.
        OSError: If the container stream cannot be opened.
    """
    interval = HEARTBEAT_INTERVAL_S if heartbeat_interval_s is None else heartbeat_interval_s
    emitter = AuditEmitter(SERVICE, __version__, stream=open_audit_stream(config.output), heartbeat_interval_s=interval)
    emitter.start()
    auditor = Auditor(emitter, require_purpose=config.require_purpose)
    auditor.emit(
        "admin",
        "config.audit",
        SYSTEM_ACTOR,
        criteria={**mode, "require_purpose": sorted(config.require_purpose)},
    )
    return auditor


@dataclass(frozen=True)
class AuditContext:
    """Who an operation runs for and why: carried from a request into its jobs.

    Attributes:
        actor (Actor): Who caused the work.
        purpose (Purpose | None): The stated purpose, ``None`` when none was stated.
        request_id (str | None): The gateway's correlation id.
    """

    actor: Actor
    purpose: Purpose | None = None
    request_id: str | None = None


#: The context of work no request started.
SYSTEM_CONTEXT = AuditContext(actor=SYSTEM_ACTOR)


def stated_purpose(case_ref: str | None, reason: str | None) -> Purpose | None:
    """Build the event's purpose, or ``None`` when neither field is stated.

    Args:
        case_ref (str | None): The case or file reference.
        reason (str | None): The short justification.

    Returns:
        Purpose | None: The purpose.
    """
    return Purpose(case_ref=case_ref, reason=reason) if case_ref or reason else None


@dataclass
class Operation:
    """One audited operation and the events it has written.

    The primary event (:meth:`record`) is written once, when the operation
    completes; related events (:meth:`child`, :meth:`transfers`) point at it.

    Attributes:
        auditor (Auditor): The chain to write to.
        context (AuditContext): Actor, purpose and request id.
        action (str): The primary event's action.
        operation (str): The primary event's operation name.
        container (str | None): The logical collection, put on every target.
        criteria (dict[str, object]): Parameters every event of it carries.
        event_id (str | None): The primary event's id, once written.
        settled (bool): Whether the primary event was attempted.
    """

    auditor: Auditor
    context: AuditContext
    action: str
    operation: str
    container: str | None = None
    criteria: dict[str, object] = field(default_factory=dict)
    event_id: str | None = None
    settled: bool = False

    def _target(self, target: Target | None) -> Target | None:
        """Give a target the operation's collection when it names none.

        Args:
            target (Target | None): The target as the caller built it.

        Returns:
            Target | None: The target to write.
        """
        if target is None or target.container is not None or self.container is None:
            return target
        return Target(kind=target.kind, ids=target.ids, count=target.count, container=self.container)

    def _emit(
        self,
        action: str,
        operation: str,
        *,
        outcome: str,
        parent_event_id: str | None,
        target: Target | None,
        criteria: Mapping[str, object] | None,
        recipient: str | None,
        ai: AiCall | None,
        actor: Actor | None = None,
    ) -> str:
        """Write one event of this operation.

        Args:
            action (str): The action.
            operation (str): The operation name.
            outcome (str): The outcome.
            parent_event_id (str | None): The event it belongs to.
            target (Target | None): What was touched.
            criteria (Mapping[str, object] | None): Parameters beside the operation's own.
            recipient (str | None): Where the data went.
            ai (AiCall | None): Model provenance.
            actor (Actor | None): Who, when not the context's actor.

        Returns:
            str: The event's id.
        """
        merged = {**self.criteria, **(criteria or {})}
        event = self.auditor.emit(
            action,
            operation,
            actor or self.context.actor,
            outcome=outcome,
            request_id=self.context.request_id,
            parent_event_id=parent_event_id,
            purpose=self.context.purpose,
            target=self._target(target),
            criteria=merged or None,
            recipient=recipient,
            ai=ai,
        )
        return str(event["event_id"])

    def record(
        self,
        *,
        target: Target | None = None,
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
        ai: AiCall | None = None,
        outcome: str = "ok",
        actor: Actor | None = None,
    ) -> str:
        """Write the operation's primary event, once.

        Marked settled before the write, so a failed write is never followed
        by an ``error`` event the broken stream could not take either.

        Args:
            target (Target | None): What was touched, by reference.
            criteria (Mapping[str, object] | None): Parameters beside the operation's own.
            recipient (str | None): Where the data went.
            ai (AiCall | None): Model provenance.
            outcome (str): ``ok``, ``denied`` or ``error``.
            actor (Actor | None): Who, when not the context's actor.

        Returns:
            str: The primary event's id.

        Raises:
            RuntimeError: When the primary event was already written.
        """
        if self.settled:
            raise RuntimeError(f"audit operation {self.operation} already recorded")
        self.settled = True
        self.event_id = self._emit(
            self.action,
            self.operation,
            outcome=outcome,
            parent_event_id=None,
            target=target,
            criteria=criteria,
            recipient=recipient,
            ai=ai,
            actor=actor,
        )
        return self.event_id

    def child(
        self,
        action: str,
        operation: str,
        *,
        target: Target | None = None,
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
        ai: AiCall | None = None,
        outcome: str = "ok",
    ) -> str:
        """Write an event of this operation that points at its primary event.

        Args:
            action (str): The action.
            operation (str): The operation name.
            target (Target | None): What was touched.
            criteria (Mapping[str, object] | None): Parameters beside the operation's own.
            recipient (str | None): Where the data went.
            ai (AiCall | None): Model provenance.
            outcome (str): The outcome.

        Returns:
            str: The event's id.

        Raises:
            RuntimeError: When the primary event has not been written yet.
        """
        if self.event_id is None:
            raise RuntimeError(f"audit operation {self.operation} has no primary event yet")
        return self._emit(
            action,
            operation,
            outcome=outcome,
            parent_event_id=self.event_id,
            target=target,
            criteria=criteria,
            recipient=recipient,
            ai=ai,
        )

    def fail(self, *, target: Target | None = None, criteria: Mapping[str, object] | None = None) -> None:
        """Record the operation as failed, unless an outcome was already written.

        Args:
            target (Target | None): What it touched before failing, if known.
            criteria (Mapping[str, object] | None): Parameters beside the operation's own.
        """
        if not self.settled:
            self.record(target=target, criteria=criteria, outcome="error")

    def transfers(self, meter: TransferMeter | None, *, prefix: str | None = None) -> None:
        """Write one ``ai_call`` event per stage the operation sent data to inference from.

        Counts are calls made, whether or not the model answered: the data
        left either way.

        Args:
            meter (TransferMeter | None): What the operation's clients counted.
            prefix (str | None): Operation-name prefix; the operation's own
                name when omitted (``ingest.run`` → ``ingest.run.embed``).
        """
        if meter is None:
            return
        for key, tally in meter.snapshot():
            self.child(
                "ai_call",
                f"{prefix or self.operation}.{key.stage}",
                target=Target(kind=key.kind, count=tally.items),
                criteria={"requests": tally.requests},
                recipient=key.recipient,
                ai=AiCall(model_alias=key.model),
            )


@contextmanager
def metered(
    op: Operation,
    *,
    prefix: str | None = None,
    on_error: Callable[[BaseException], Mapping[str, object] | None] | None = None,
) -> Iterator[TransferMeter]:
    """Count an operation's inference calls; if it fails, record that with what it sent.

    On success the caller records the primary event and then writes
    ``op.transfers(meter)``. On an exception the operation is recorded as
    failed and its transfers so far are written beneath it, since data that
    left was sent whether or not the operation finished. A failed audit write
    is not followed by more writes.

    Args:
        op (Operation): The operation doing the work.
        prefix (str | None): Operation-name prefix for its transfer events.
        on_error (Callable[[BaseException], Mapping[str, object] | None] | None):
            Criteria for the failure event, from the exception (e.g. that the
            run was cancelled). Never the exception's text.

    Yields:
        TransferMeter: The meter bound for the block.
    """
    with metering() as meter:
        try:
            yield meter
        except AUDIT_FAILURES:
            raise
        except BaseException as exc:
            if not op.settled:
                op.fail(criteria=on_error(exc) if on_error is not None else None)
            if op.event_id is not None:
                op.transfers(meter, prefix=prefix)
            raise


def actor_for(principal: Principal) -> Actor:
    """Map a resolved principal to the audit actor.

    Args:
        principal (Principal): The request's principal.

    Returns:
        Actor: The caller, with ``on_behalf_of`` when an admin works in
            another owner's namespace.
    """
    owner = principal.effective_owner
    return Actor(principal.name, principal.via, on_behalf_of=owner if owner != principal.name else None)


def _request_purpose(request: Request) -> Purpose | None:
    """Read the request's stated purpose.

    Args:
        request (Request): The incoming request.

    Returns:
        Purpose | None: The purpose, ``None`` when none is stated.

    Raises:
        HTTPException: 400 for a malformed, oversized or ambiguous header. No
            event: there is no valid operation yet.
    """
    try:
        case_ref, reason = parse_purpose_header(request.headers.get(PURPOSE_HEADER))
    except PurposeError as exc:
        raise HTTPException(status_code=400, detail="Invalid request.") from exc
    return stated_purpose(case_ref, reason)


def request_context(request: Request, principal: Principal) -> AuditContext:
    """Build a request's audit context.

    Args:
        request (Request): The incoming request.
        principal (Principal): Its resolved principal.

    Returns:
        AuditContext: Actor, purpose and request id.

    Raises:
        HTTPException: 400 for a malformed purpose header.
    """
    return AuditContext(
        actor=actor_for(principal),
        purpose=_request_purpose(request),
        request_id=clean_request_id(request.headers.get(REQUEST_ID_HEADER)),
    )


def _foreign_owner_attempt(principal: Principal) -> bool:
    """Whether a non-admin asked to act in another owner's namespace.

    Args:
        principal (Principal): The request's principal.

    Returns:
        bool: ``True`` for an ``owner`` parameter naming someone else.
    """
    requested = principal.requested_owner
    return requested is not None and requested != principal.name and not principal.is_admin


def _route_collection(request: Request) -> str | None:
    """Return the logical collection a request names in its path or query.

    Args:
        request (Request): The incoming request.

    Returns:
        str | None: The collection, ``None`` when it is in the body or absent.
    """
    raw = request.path_params.get("name") or request.query_params.get("collection")
    name = str(raw or "").strip()
    return name or None


def begin(
    request: Request,
    principal: Principal,
    action: str,
    operation: str,
    *,
    also: Sequence[str] = (),
    container: str | None = None,
    criteria: Mapping[str, object] | None = None,
) -> Operation:
    """Open an audited operation for a request, refusing what the trail must refuse.

    A non-admin's ``owner`` parameter naming someone else is refused (403)
    and a missing purpose where one is required (428); both are recorded as
    ``denied``, without a target.

    Args:
        request (Request): The incoming request.
        principal (Principal): Its resolved principal.
        action (str): The operation's primary action.
        operation (str): The operation name.
        also (Sequence[str]): Further actions the operation may record (e.g.
            ``ai_call`` for a chat turn), checked against the purpose rule too.
        container (str | None): The logical collection, when known.
        criteria (Mapping[str, object] | None): Parameters known up front.

    Returns:
        Operation: The open operation; its failure is recorded by the error
            handlers if the request ends in a server error first.

    Raises:
        HTTPException: 400 for a malformed purpose header, 403 for a foreign
            ``owner``, 428 when a required purpose is missing.
        RuntimeError: When no audit chain is installed.
    """
    auditor = current()
    context = request_context(request, principal)
    op = Operation(
        auditor,
        context,
        action,
        operation,
        container=container if container is not None else _route_collection(request),
        criteria=dict(criteria or {}),
    )
    # A refusal has no target, so the collection it was about goes in its criteria.
    refused = {"collection": op.container} if op.container else None
    if _foreign_owner_attempt(principal):
        op.record(
            outcome="denied",
            criteria=refused,
            actor=Actor(principal.name, principal.via, on_behalf_of=principal.requested_owner),
        )
        raise HTTPException(status_code=403, detail=_FORBIDDEN_DETAIL)
    if auditor.needs_purpose((action, *also)) and context.purpose is None:
        op.record(outcome="denied", criteria=refused)
        raise HTTPException(status_code=PURPOSE_REQUIRED_STATUS, detail=_PURPOSE_REQUIRED_DETAIL)
    operations: list[Operation] = getattr(request.state, "audit_operations", [])
    request.state.audit_operations = [*operations, op]
    return op


def audited(action: str, operation: str, *, also: Sequence[str] = ()) -> Callable[..., Operation]:
    """Build the route dependency that opens an audited operation.

    Args:
        action (str): The operation's primary action.
        operation (str): The operation name.
        also (Sequence[str]): Further actions it may record.

    Returns:
        Callable[..., Operation]: A FastAPI dependency.
    """

    def dependency(
        request: Request,
        principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
    ) -> Operation:
        """Open the operation for this request.

        Args:
            request (Request): The incoming request.
            principal (Principal): Its resolved principal.

        Returns:
            Operation: The open operation.
        """
        return begin(request, principal, action, operation, also=also)

    return dependency


def settle_quietly(
    op: Operation,
    *,
    target: Target | None = None,
    criteria: Mapping[str, object] | None = None,
) -> None:
    """Record an interrupted operation as failed, unless it has an outcome already.

    For work that is failing anyway (a server error, a broken stream, a client
    gone mid-answer): a write that fails here is logged, and the failure the
    caller is already handling goes on.

    Args:
        op (Operation): The interrupted operation.
        target (Target | None): What it touched before failing, if known.
        criteria (Mapping[str, object] | None): What is known about how far it got.
    """
    if op.settled:
        return
    try:
        op.fail(target=target, criteria=criteria)
    except Exception:
        logger.exception("Could not record the failure of audited operation '{}'", op.operation)


def settle_failed_operations(request: Request) -> None:
    """Record every open operation of a request that ends in a server error.

    Called by the error handlers.

    Args:
        request (Request): The failing request.
    """
    operations: list[Operation] = getattr(request.state, "audit_operations", [])
    for op in operations:
        settle_quietly(op)


@dataclass(frozen=True)
class CliAudit:
    """A command's own audit chain and context.

    Attributes:
        auditor (Auditor): The command's chain.
        context (AuditContext): The operator and the stated purpose.
    """

    auditor: Auditor
    context: AuditContext

    def operation(
        self,
        action: str,
        operation: str,
        *,
        container: str | None = None,
        criteria: Mapping[str, object] | None = None,
    ) -> Operation:
        """Open one operation of the command.

        Args:
            action (str): The primary action.
            operation (str): The operation name.
            container (str | None): The logical collection.
            criteria (Mapping[str, object] | None): Parameters known up front.

        Returns:
            Operation: The open operation.
        """
        return Operation(
            self.auditor, self.context, action, operation, container=container, criteria=dict(criteria or {})
        )


class PurposeRefused(SystemExit):
    """Raised to end a command that needs a stated purpose and has none (exit code 2)."""

    def __init__(self) -> None:
        """Exit with code 2."""
        super().__init__(2)


@contextmanager
def cli_audit(command: str, action: str, operation: str, *, also: Sequence[str] = ()) -> Iterator[CliAudit]:
    """Run a command inside its own audit chain.

    Opens the chain on ``AUDIT_OUTPUT`` (inside the backend container, the
    container's own log), refuses a run whose actions need a purpose it does
    not state, and closes the chain however the command ends.

    Args:
        command (str): The command's name, stated on the chain's config event.
        action (str): The command's primary action.
        operation (str): Its operation name, for a refusal's event.
        also (Sequence[str]): Further actions it may record.

    Yields:
        CliAudit: The chain and context to record with.

    Raises:
        PurposeRefused: Exit code 2 for a missing required purpose, or a
            malformed ``AUDIT_CASE_REF``/``AUDIT_REASON``.
    """
    try:
        cli = load_audit_cli_env()
    except PurposeError as exc:
        logger.error("Refused: {}.", exc)
        raise PurposeRefused() from exc
    auditor = open_auditor(load_audit_env(), mode={**identity_mode(load_principal_env()), "command": command})
    try:
        context = AuditContext(actor=Actor(cli.user, "cli"), purpose=stated_purpose(cli.case_ref, cli.reason))
        if auditor.needs_purpose((action, *also)) and context.purpose is None:
            Operation(auditor, context, action, operation).record(outcome="denied")
            logger.error("Refused: '{}' needs a stated purpose. Set AUDIT_CASE_REF or AUDIT_REASON.", command)
            raise PurposeRefused()
        yield CliAudit(auditor, context)
    finally:
        auditor.close()


def job_context(raw: Any) -> AuditContext:
    """Return the audit context a job carries, or the system's for one no request started.

    Args:
        raw (Any): The job's ``audit`` field.

    Returns:
        AuditContext: The context to record the job's events with.
    """
    return raw if isinstance(raw, AuditContext) else SYSTEM_CONTEXT
