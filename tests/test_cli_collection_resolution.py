"""Tests for resolving the collection name an operator types on the CLI."""

from __future__ import annotations

import types
from typing import Any, cast

import pytest

from docint.cli._collection import CollectionNotFoundError, resolve_collection_name
from docint.core.search.index import BackfillSummary


def _rag(*, existing: set[str], mappings: list[tuple[str | None, str]]) -> Any:
    """Build a RAG stand-in with a Qdrant client and an ownership manager.

    Args:
        existing (set[str]): Physical collection names Qdrant knows about.
        mappings (list[tuple[str | None, str]]): ``(owner, logical)`` rows.

    Returns:
        Any: A stand-in exposing the two surfaces the resolver uses.
    """
    owners = types.SimpleNamespace(
        list_all=lambda: list(mappings),
        resolve=lambda owner, logical: f"u{abs(hash(owner)) % 10**12:012x}__{logical}",
    )
    return cast(
        Any,
        types.SimpleNamespace(
            qdrant_client=types.SimpleNamespace(collection_exists=lambda collection_name: collection_name in existing),
            ensure_collection_owner_manager=lambda: owners,
        ),
    )


def test_a_physical_name_is_used_as_typed() -> None:
    """An operator who already knows the internal name must not be second-guessed."""
    rag = _rag(existing={"u0123456789ab__docs"}, mappings=[])

    assert resolve_collection_name(rag, "u0123456789ab__docs") == "u0123456789ab__docs"


def test_a_logical_name_resolves_to_its_physical_collection() -> None:
    """The UI shows logical names, so that is what an operator will type.

    Collections are owner-namespaced (``u<hash>__<logical>``), and passing the
    logical name straight to Qdrant just 404s.
    """
    rag = _rag(existing=set(), mappings=[("alice", "docs")])

    resolved = resolve_collection_name(rag, "docs")

    assert resolved.endswith("__docs")
    assert resolved != "docs"


def test_an_unknown_name_raises_rather_than_resolving_to_nothing() -> None:
    """A typo must stop the run, not produce a no-op that reports success."""
    rag = _rag(existing=set(), mappings=[("alice", "docs")])

    with pytest.raises(CollectionNotFoundError) as excinfo:
        resolve_collection_name(rag, "dcos")

    assert "dcos" in str(excinfo.value)


def test_a_name_owned_by_several_owners_is_ambiguous() -> None:
    """Two users may own the same logical name; guessing could target the wrong one."""
    rag = _rag(existing=set(), mappings=[("alice", "docs"), ("bob", "docs")])

    with pytest.raises(CollectionNotFoundError) as excinfo:
        resolve_collection_name(rag, "docs")

    message = str(excinfo.value)
    assert "alice" in message
    assert "bob" in message
    # The physical names must be in the message: an operator cannot derive
    # them (they hash the owner), so naming only the owners is a dead end.
    assert message.count("__docs") == 2


def test_search_index_cli_fails_loudly_on_an_unknown_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed migration must never end with "Search index ready".

    The backfill's scroll is fail-soft, so a 404 collection produced
    ``0 scanned, 0 written`` and the CLI then announced success. An operator
    working through a list of collections would tick that one off.
    """
    from docint.cli import search_index as cli

    monkeypatch.setattr(cli, "RAG", lambda **kwargs: cast(Any, types.SimpleNamespace(unload_models=lambda: None)))
    monkeypatch.setattr(
        cli,
        "resolve_collection_name",
        lambda rag, typed: (_ for _ in ()).throw(CollectionNotFoundError(f"No collection named {typed!r}.")),
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.build_search_index("nope")

    assert excinfo.value.code != 0


def test_search_index_cli_fails_when_the_index_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the payload index, search is case-sensitive on non-ASCII text.

    Reporting success would leave the operator believing a collection is
    searchable when German title-case tokens silently will not match.
    """
    from docint.cli import search_index as cli

    monkeypatch.setattr(
        cli,
        "RAG",
        lambda **kwargs: cast(Any, types.SimpleNamespace(unload_models=lambda: None, qdrant_client=object())),
    )
    monkeypatch.setattr(cli, "resolve_collection_name", lambda rag, typed: "u0__docs")
    monkeypatch.setattr(cli, "ensure_search_index", lambda client, collection: False)

    with pytest.raises(SystemExit) as excinfo:
        cli.build_search_index("docs")

    assert excinfo.value.code != 0


def test_bulk_indexes_every_collection_and_skips_none() -> None:
    """The bulk run covers whatever the collection list reports.

    It works on *physical* names, which sidesteps the logical-name ambiguity
    entirely: two users owning the same logical name are simply two entries.
    """
    from docint.cli import search_index as cli

    done: list[str] = []
    rag = cast(Any, types.SimpleNamespace(list_collections=lambda: ["u1__a", "u2__a", "u1__b"]))

    failures = cli.index_all_collections(rag, index_one=done.append)

    assert done == ["u1__a", "u2__a", "u1__b"]
    assert failures == []


def test_bulk_continues_past_a_failing_collection() -> None:
    """One bad collection must not strand the other eighteen.

    Halting would leave the rest unmigrated with no signal about which; the
    non-zero exit still reports that something failed.
    """
    from docint.cli import search_index as cli

    done: list[str] = []

    def _index_one(name: str) -> None:
        if name == "u2__bad":
            raise RuntimeError("qdrant exploded")
        done.append(name)

    rag = cast(Any, types.SimpleNamespace(list_collections=lambda: ["u1__a", "u2__bad", "u1__b"]))

    failures = cli.index_all_collections(rag, index_one=_index_one)

    assert done == ["u1__a", "u1__b"]
    assert failures == ["u2__bad"]


def test_bulk_cli_exits_non_zero_when_any_collection_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial migration must not look like a clean one."""
    from docint.cli import search_index as cli

    monkeypatch.setattr(cli, "RAG", lambda **kwargs: cast(Any, types.SimpleNamespace(unload_models=lambda: None)))
    monkeypatch.setattr(cli, "index_all_collections", lambda rag, **kwargs: ["u2__bad"])

    with pytest.raises(SystemExit) as excinfo:
        cli.build_all_search_indexes()

    assert excinfo.value.code != 0


def test_bulk_cli_exits_zero_when_every_collection_succeeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean run must not report failure."""
    from docint.cli import search_index as cli

    monkeypatch.setattr(cli, "RAG", lambda **kwargs: cast(Any, types.SimpleNamespace(unload_models=lambda: None)))
    monkeypatch.setattr(cli, "index_all_collections", lambda rag, **kwargs: [])

    cli.build_all_search_indexes()  # must not raise


def _fake_rag_class(client: Any) -> Any:
    """Return a stand-in for the ``RAG`` class the CLI constructs.

    A plain lambda will not do: the CLI also reads
    ``RAG._extract_payload_text`` off the class to inject into the backfill.

    Args:
        client (Any): Qdrant client stand-in exposed on the instance.

    Returns:
        Any: A class the CLI can instantiate.
    """

    class _FakeRag:
        _extract_payload_text = staticmethod(lambda payload: "chunk text")
        _extract_indexable_text = staticmethod(lambda payload: "chunk text")

        def __init__(self, **kwargs: Any) -> None:
            self.qdrant_client = client

        def unload_models(self) -> None:
            """No models to unload."""

    return _FakeRag


def test_search_index_also_covers_the_image_companion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image captions and tags live in the ``_images`` companion.

    Indexing only the main collection leaves every figure and video keyframe
    unfindable by keyword while the run reports success.
    """
    from docint.cli import search_index as cli

    indexed: list[str] = []
    backfilled: list[str] = []
    client = types.SimpleNamespace(collection_exists=lambda collection_name: True)
    monkeypatch.setattr(cli, "RAG", _fake_rag_class(client))
    monkeypatch.setattr(cli, "resolve_collection_name", lambda rag, typed: "u0__docs")
    monkeypatch.setattr(cli, "ensure_search_index", lambda c, name: indexed.append(name) or True)
    monkeypatch.setattr(
        cli,
        "backfill_search_text",
        lambda c, name, **kwargs: backfilled.append(name) or BackfillSummary(scanned=1, written=1, skipped=0, empty=0),
    )

    cli.build_search_index("docs")

    assert indexed == ["u0__docs", "u0__docs_images"]
    assert backfilled == ["u0__docs", "u0__docs_images"]


def test_search_index_skips_a_missing_image_companion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collection with no images has no companion; that is not a failure."""
    from docint.cli import search_index as cli

    indexed: list[str] = []
    client = types.SimpleNamespace(collection_exists=lambda collection_name: not collection_name.endswith("_images"))
    monkeypatch.setattr(cli, "RAG", _fake_rag_class(client))
    monkeypatch.setattr(cli, "resolve_collection_name", lambda rag, typed: "u0__docs")
    monkeypatch.setattr(cli, "ensure_search_index", lambda c, name: indexed.append(name) or True)
    monkeypatch.setattr(
        cli,
        "backfill_search_text",
        lambda c, name, **kwargs: BackfillSummary(scanned=1, written=1, skipped=0, empty=0),
    )

    cli.build_search_index("docs")

    assert indexed == ["u0__docs"]


def test_search_index_cli_ensures_field_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backport must also create the field-search lane's payload indexes.

    Otherwise a pre-existing collection answers field-scoped searches only
    after a lazily-ensured first request, rather than being ready immediately
    after an operator runs the backport.
    """
    from docint.cli import search_index as cli

    client = types.SimpleNamespace(collection_exists=lambda collection_name: False)
    monkeypatch.setattr(cli, "RAG", _fake_rag_class(client))
    monkeypatch.setattr(cli, "resolve_collection_name", lambda rag, typed: "u0__docs")
    monkeypatch.setattr(cli, "ensure_search_index", lambda c, name: True)
    monkeypatch.setattr(
        cli,
        "backfill_search_text",
        lambda c, name, **kwargs: BackfillSummary(scanned=1, written=1, skipped=0, empty=0),
    )
    ensured: list[str] = []
    monkeypatch.setattr(cli, "ensure_field_indexes", lambda client, collection: ensured.append(collection) or True)

    cli.build_search_index("docs")

    assert ensured == ["u0__docs"]


def test_search_index_cli_warns_but_does_not_fail_when_field_indexes_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing field index degrades one feature, not the whole migration.

    Unlike the ``search_text`` index, a failed field index must not abort the
    run — field search is a secondary use of this command; keyword search
    still works without those indexes.
    """
    from docint.cli import search_index as cli

    client = types.SimpleNamespace(collection_exists=lambda collection_name: False)
    monkeypatch.setattr(cli, "RAG", _fake_rag_class(client))
    monkeypatch.setattr(cli, "resolve_collection_name", lambda rag, typed: "u0__docs")
    monkeypatch.setattr(cli, "ensure_search_index", lambda c, name: True)
    monkeypatch.setattr(
        cli,
        "backfill_search_text",
        lambda c, name, **kwargs: BackfillSummary(scanned=1, written=1, skipped=0, empty=0),
    )
    monkeypatch.setattr(cli, "ensure_field_indexes", lambda client, collection: False)

    cli.build_search_index("docs")  # must not raise


def test_search_index_cli_ensures_field_indexes_on_the_image_companion_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Posting-author searches reach the companion, so it needs the indexes as well."""
    from docint.cli import search_index as cli

    client = types.SimpleNamespace(collection_exists=lambda collection_name: True)
    monkeypatch.setattr(cli, "RAG", _fake_rag_class(client))
    monkeypatch.setattr(cli, "resolve_collection_name", lambda rag, typed: "u0__docs")
    monkeypatch.setattr(cli, "ensure_search_index", lambda c, name: True)
    monkeypatch.setattr(
        cli,
        "backfill_search_text",
        lambda c, name, **kwargs: BackfillSummary(scanned=1, written=1, skipped=0, empty=0),
    )
    ensured: list[str] = []
    monkeypatch.setattr(cli, "ensure_field_indexes", lambda client, collection: ensured.append(collection) or True)

    cli.build_search_index("docs")

    assert ensured == ["u0__docs", "u0__docs_images"]
