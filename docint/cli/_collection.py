"""Resolve the collection name an operator types into a physical one.

Collections are *logical* to users: the SPA shows ``mydocs`` while Qdrant holds
``u<owner-hash>__mydocs`` (see the multi-tenant convention in ``CLAUDE.md``).
CLI tools take a name from a human, so they have to accept the one that human
has actually seen — passing a logical name straight to Qdrant just 404s.
"""

from __future__ import annotations

from typing import Any


class CollectionNotFoundError(RuntimeError):
    """Raised when a typed collection name cannot be resolved to exactly one collection."""


def resolve_collection_name(rag: Any, typed: str) -> str:
    """Return the physical Qdrant collection for a name an operator typed.

    Accepts either form. A name that already exists in Qdrant is used as typed,
    so an operator who knows the internal name is not second-guessed;
    otherwise it is treated as a logical name and resolved through the
    ownership store.

    Args:
        rag (Any): A ``RAG`` exposing ``qdrant_client`` and
            ``ensure_collection_owner_manager()``.
        typed (str): The name as entered.

    Returns:
        str: The physical collection name.

    Raises:
        CollectionNotFoundError: When the name matches no collection, or is a
            logical name owned by several principals — guessing between them
            could target the wrong user's data.
    """
    name = typed.strip()
    if not name:
        raise CollectionNotFoundError("No collection name given.")

    try:
        if rag.qdrant_client.collection_exists(collection_name=name):
            return name
    except Exception as exc:  # pragma: no cover - transport-dependent
        raise CollectionNotFoundError(f"Could not reach Qdrant to look up {name!r}: {exc}") from exc

    owners = rag.ensure_collection_owner_manager()
    matches = [(owner, logical) for owner, logical in owners.list_all() if logical == name]

    if not matches:
        raise CollectionNotFoundError(
            f"No collection named {name!r}. Pass the name shown in the app, or the physical Qdrant name."
        )

    if len(matches) > 1:
        # Name the physical collections, not just the owners. An operator
        # cannot derive them — they hash the owner — so listing owners alone
        # leaves them with an error and no way to act on it. Each of these
        # needs its own run; migrating one silently leaves the other user's
        # collection unsearchable.
        candidates = []
        for owner, logical in sorted(matches, key=lambda row: str(row[0])):
            physical = owners.resolve(owner, logical)
            candidates.append(f"  {physical}   (owner: {owner})")
        listing = "\n".join(candidates)
        raise CollectionNotFoundError(
            f"{name!r} is owned by several users, so it is ambiguous. "
            f"Re-run once per collection with the physical name:\n{listing}"
        )

    owner, logical = matches[0]
    physical = owners.resolve(owner, logical)
    if not physical:
        raise CollectionNotFoundError(f"Could not resolve {name!r} for owner {owner!r}.")
    return str(physical)
