"""The audit record of an ingest run, shared by the ``ingest`` command and the API's ingest jobs.

A run is one ``collect`` naming the files it read by content hash. The
social linker's joins (postings with their comments and media) are a
``combine`` beneath it, and every stage that sent content to inference is
an ``ai_call`` with its counts. Both callers record a run here, so a run
reads the same in the log whichever way it was started.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from docint.core.audit import Operation
from docint.core.rag import IngestStats
from docint.utils.audit_event import Target
from docint.utils.audit_meter import TransferMeter

#: ``IngestStats`` fields that are references, not counts: the audit event's, not the run-summary line's.
STATS_REFERENCES = ("file_hashes", "social_link")


def stats_counts(stats: IngestStats | None) -> dict[str, Any] | None:
    """Return a run's counters, without the references the audit event carries.

    Args:
        stats (IngestStats | None): The run's statistics.

    Returns:
        dict[str, Any] | None: The counters.
    """
    if stats is None:
        return None
    return {key: value for key, value in asdict(stats).items() if key not in STATS_REFERENCES}


def record_run(op: Operation, stats: IngestStats | None, meter: TransferMeter, *, empty: bool) -> None:
    """Record a finished ingest run: the documents collected, the links made, then what went to inference.

    Args:
        op (Operation): The run's open ``collect`` operation.
        stats (IngestStats | None): What the run did; ``None`` when it read nothing.
        meter (TransferMeter): The run's model calls.
        empty (bool): Whether the run ingested nothing.
    """
    hashes = list(stats.file_hashes) if stats is not None else []
    op.record(
        target=Target(kind="document", ids=hashes, count=len(hashes)),
        criteria={**(stats_counts(stats) or {}), "empty": empty},
    )
    social = stats.social_link if stats is not None else None
    if social is not None and (social.postings or social.media):
        op.child(
            "combine",
            "ingest.social_link",
            target=Target(kind="posting", ids=[uuid for uuid in social.postings if uuid], count=len(social.postings)),
            criteria={"dossiers": social.dossiers, "comments": social.comments, "media_linked": social.media},
        )
    op.transfers(meter, prefix="ingest")
