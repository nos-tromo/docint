"""Thin HTTP client for the remote Nextext media-processing service.

Mirrors the posture of ``ner_client``/``clip_client``: docint forwards a
media file to Nextext, which decodes it (PyAV/ffmpeg), runs VAD → diarize →
Whisper, and exposes a ``docint.jsonl`` transcript plus a ``keyframes.zip``
artifact. docint stays CPU-only and media-dependency-free.

All failures are fail-soft: ``process_media`` returns a ``NextextResult`` with
``status='error'`` and empty payloads rather than raising, so one bad clip
never aborts a batch.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from loguru import logger

from docint.utils.env_cfg import NextextConfig, load_nextext_env

__all__ = ["NextextClient", "NextextResult"]

_TERMINAL_OK = "completed"
_TERMINAL_FAIL = {"failed", "error", "cancelled"}


@dataclass(frozen=True)
class NextextResult:
    """Outcome of a single media file processed by Nextext."""

    status: str
    transcript_jsonl: bytes | None = None
    keyframes: list[bytes] = field(default_factory=list)
    error: str | None = None


class NextextClient:
    """Submit a media file to Nextext, await the job, and fetch its artifacts."""

    def __init__(self, cfg: NextextConfig | None = None, *, client: httpx.Client | None = None) -> None:
        """Create the client.

        Args:
            cfg (NextextConfig | None): Configuration; defaults to ``load_nextext_env()``.
            client (httpx.Client | None): Injected HTTP client (tests pass a
                ``MockTransport``-backed client). When ``None``, one is built
                from ``cfg`` with the Bearer header when an API key is set.
        """
        self._cfg = cfg if cfg is not None else load_nextext_env()
        if client is not None:
            self._client = client
        else:
            headers = {"Authorization": f"Bearer {self._cfg.api_key}"} if self._cfg.api_key else {}
            self._client = httpx.Client(base_url=self._cfg.api_base, timeout=self._cfg.timeout, headers=headers)

    def _options_payload(self) -> str:
        """Return the JSON ``options`` form field forwarded to Nextext."""
        # Note: keyframe_dedup_cosine is NOT forwarded; near-duplicate pruning
        # is applied client-side in docint's image service.
        return json.dumps(
            {
                "keyframes_per_minute": self._cfg.keyframes_per_minute,
                "keyframes_max": self._cfg.keyframes_max,
            }
        )

    def _await_job(self, job_id: str) -> str:
        """Poll a job until it reaches a terminal status or the budget expires.

        Args:
            job_id (str): The job identifier returned by submission.

        Returns:
            str: Terminal status (``'completed'`` on success, server terminal
                statuses from ``_TERMINAL_FAIL`` on job failure, ``'timeout'``
                on poll expiry, ``'poll_error'`` on HTTP/transport failure).
        """
        deadline = time.monotonic() + self._cfg.poll_max_seconds
        while True:
            try:
                resp = self._client.get(f"/jobs/{job_id}")
                resp.raise_for_status()
            except httpx.HTTPError:
                logger.warning("Nextext poll error for job {}", job_id)
                return "poll_error"
            status = str(resp.json().get("status") or "").lower()
            if status == _TERMINAL_OK or status in _TERMINAL_FAIL:
                return status
            if time.monotonic() >= deadline:
                return "timeout"
            time.sleep(self._cfg.poll_interval)

    def _fetch_artifact(self, job_id: str, name: str) -> bytes | None:
        """Fetch one job artifact, returning ``None`` when absent (404)."""
        resp = self._client.get(f"/jobs/{job_id}/artifacts/{name}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _unzip_jpegs(blob: bytes | None) -> list[bytes]:
        """Return the JPEG members of a keyframes zip in name order."""
        if not blob:
            return []
        frames: list[bytes] = []
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    frames.append(zf.read(name))
        return frames

    def process_media(self, file_path: Path) -> NextextResult:
        """Run one media file through Nextext and return its artifacts.

        Args:
            file_path (Path): Path to the audio/video file to process.

        Returns:
            NextextResult: Transcript JSONL bytes (or ``None`` when no speech)
                and keyframe JPEG bytes. Possible statuses: ``'completed'`` on
                success, server terminal statuses (``'failed'``, ``'error'``,
                ``'cancelled'``), ``'timeout'`` if polling expires, ``'poll_error'``
                on HTTP failures during polling, ``'error'`` on submission/fetch
                exceptions, ``'disabled'`` if Nextext is not enabled. Fail-soft:
                non-``completed`` statuses yield empty payloads.
        """
        if not self._cfg.enabled:
            return NextextResult(status="disabled")
        try:
            with file_path.open("rb") as handle:
                resp = self._client.post(
                    "/jobs",
                    files={"file": (file_path.name, handle, "application/octet-stream")},
                    data={"options": self._options_payload()},
                )
            resp.raise_for_status()
            job_id = str(resp.json()["job_id"])
            status = self._await_job(job_id)
            if status != _TERMINAL_OK:
                logger.warning("Nextext job {} for {} ended status={}", job_id, file_path.name, status)
                return NextextResult(status=status)
            transcript = self._fetch_artifact(job_id, "docint.jsonl")
            keyframes = self._unzip_jpegs(self._fetch_artifact(job_id, "keyframes.zip"))
            return NextextResult(status="completed", transcript_jsonl=transcript, keyframes=keyframes)
        except Exception as exc:
            logger.warning("Nextext processing failed for {}: {}", file_path.name, exc)
            return NextextResult(status="error", error=str(exc))
