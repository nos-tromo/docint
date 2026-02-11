from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING, cast

import pandas as pd
from llama_index.core import Response
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode
from loguru import logger
from sqlalchemy.orm import Session

from docint.agents.context import TurnContext as AgentTurnContext
from docint.core.state.base import _make_session_maker
from docint.core.state.citation import Citation
from docint.core.state.conversation import Conversation
from docint.core.state.turn import Turn

if TYPE_CHECKING:
    from docint.core.rag import RAG


@dataclass(slots=True)
class SessionManager:
    """
    Owns chat session state, persistence, and exports.
    """

    rag: RAG
    chat_engine: RetrieverQueryEngine | CondenseQuestionChatEngine | None = field(
        default=None, init=False
    )
    chat_memory: ChatMemoryBuffer | None = field(default=None, init=False)
    agent_contexts: dict[str, AgentTurnContext] = field(
        default_factory=dict, init=False
    )
    _SessionMaker: Any | None = field(default=None, init=False, repr=False)
    session_id: str | None = field(default=None, init=False)
    session_store: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to set up the session store.
        """
        self.session_store = self.rag.session_store

    def init_session_store(self, db_url: str | None = None) -> None:
        """
        Initialize the session store.

        Args:
           db_url (str | None): Optional database URL to override default.
        """
        if db_url:
            self.session_store = db_url
        self._SessionMaker = _make_session_maker(self.session_store)

    def reset_runtime(self) -> None:
        """
        Reset the runtime state.
        """
        self.session_id = None
        self.chat_engine = None
        self.chat_memory = None

    def start_session(self, requested_id: str | None = None) -> str:
        """
        Start a new chat session.

        Args:
            requested_id (str | None, optional): The ID of the session to start. Defaults to None.

        Returns:
            str: The ID of the started session.

        Raises:
            RuntimeError: If the session cannot be started.
        """
        if not requested_id:
            requested_id = str(uuid.uuid4())
        self.session_id = requested_id

        # Initialize agent context for this session
        self.agent_contexts.setdefault(
            requested_id, AgentTurnContext(session_id=requested_id)
        )

        with self._session_scope() as s:
            self._load_or_create_convo(s, requested_id)

        rolling = self._get_rolling_summary(requested_id)
        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=2000, chat_history=[]
        )
        if rolling and self.chat_memory is not None:
            self.chat_memory.put(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Conversation summary so far:\n{rolling}",
                )
            )

        engine = self.rag.query_engine
        if engine is None:
            logger.error("RuntimeError: Query engine has not been initialized.")
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )

        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=engine,
            memory=self.chat_memory,
            llm=self.rag.gen_model,
        )
        return requested_id

    def get_agent_context(self, session_id: str) -> AgentTurnContext:
        """
        Return the agent context for a session, creating it if missing.

        Args:
            session_id (str): The ID of the session.

        Returns:
            AgentTurnContext: The agent context for the session.
        """
        return self.agent_contexts.setdefault(
            session_id, AgentTurnContext(session_id=session_id)
        )

    def _get_session_context(self, session_id: str) -> str:
        """
        Build a context string from rolling summary plus recent unsummarized turns.

        Args:
            session_id (str): The ID of the session.

        Returns:
            str: The context string.
        """
        with self._session_scope() as s:
            conv = s.get(Conversation, session_id)
            if not conv:
                return ""

            summary = cast(str | None, conv.rolling_summary) or ""
            turns = conv.turns

            # Reconstruct recent history not covered by summary
            # Matches logic in _maybe_update_summary (every 5 turns)
            # If turns=6, 6%5=1, last 1 turn is not in summary.
            # If turns=4, 4%5=4, last 4 turns are not in summary.
            remainder = len(turns) % 5

            if remainder == 0:
                return summary

            recent_turns = turns[-remainder:]
            parts = []
            if summary:
                parts.append(summary)

            for t in recent_turns:
                parts.append(f"User: {t.user_text}\nAssistant: {t.model_response}")

            return "\n\n".join(parts)

    def chat(self, user_msg: str) -> dict[str, Any]:
        """
        Handle a chat message from the user.

        Args:
            user_msg (str): The message from the user.

        Returns:
            dict[str, Any]: The response data.

        Raises:
            ValueError: If the user message is empty.
            RuntimeError: If the query engine has not been initialized.
        """
        if not user_msg.strip():
            logger.error("ValueError: Chat prompt cannot be empty.")
            raise ValueError("Chat prompt cannot be empty.")

        engine = self.rag.query_engine
        if engine is None:
            logger.error("RuntimeError: Query engine has not been initialized.")
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )

        session_id = self.session_id
        if self.chat_engine is None or session_id is None:
            session_id = self.start_session(session_id)

        # Include both summary and recent unsummarized turns in context
        session_context = self._get_session_context(session_id)
        retrieval_query = (
            f"{session_context}\n\nUser question: {user_msg}"
            if session_context
            else user_msg
        )

        resp = engine.query(retrieval_query)
        response = self.rag._normalize_response_data(user_msg, resp)
        self._persist_turn(session_id, user_msg, resp, response)
        self._maybe_update_summary(session_id)
        return response

    def stream_chat(self, user_msg: str) -> Iterator[str | dict]:
        """
        Handle a streaming chat message from the user.

        Args:
            user_msg (str): The message from the user.

        Yields:
            str | dict: Chunks of text, followed by a dict with metadata.

        Raises:
            ValueError: If the user message is empty.
            RuntimeError: If the index has not been initialized.
        """
        if not user_msg.strip():
            logger.error("ValueError: Chat prompt cannot be empty.")
            raise ValueError("Chat prompt cannot be empty.")

        # Ensure index exists
        if self.rag.index is None:
            self.rag.create_index()

        if self.rag.index is None:
            raise RuntimeError("Index not initialized")

        # Create a temporary streaming engine
        k = min(max(self.rag.retrieve_similarity_top_k, self.rag.rerank_top_n * 8), 64)
        streaming_engine = RetrieverQueryEngine.from_args(
            retriever=self.rag.index.as_retriever(similarity_top_k=k),
            llm=self.rag.gen_model,
            node_postprocessors=[self.rag.reranker],
            streaming=True,
        )

        session_id = self.session_id
        if self.chat_engine is None or session_id is None:
            session_id = self.start_session(session_id)

        summary = self._get_rolling_summary(session_id)
        retrieval_query = (
            f"{summary}\n\nUser question: {user_msg}" if summary else user_msg
        )

        response = streaming_engine.query(retrieval_query)

        full_text = ""
        # response.response_gen is the generator
        for token in response.response_gen:
            full_text += token
            yield token

        # Create a Response object to reuse normalization logic
        final_response = Response(
            response=full_text, source_nodes=response.source_nodes
        )

        normalized = self.rag._normalize_response_data(user_msg, final_response)
        self._persist_turn(session_id, user_msg, final_response, normalized)
        self._maybe_update_summary(session_id)

        # Yield metadata
        yield {
            "sources": normalized.get("sources", []),
            "session_id": session_id,
            "reasoning": normalized.get("reasoning"),
        }

    def export_session(
        self, session_id: str | None = None, out_dir: str | Path = "session"
    ) -> Path:
        """
        Export the chat session to a directory.

        Args:
            session_id (str | None, optional): The ID of the session to export. Defaults to None.
            out_dir (str | Path, optional): The output directory. Defaults to "session".

        Returns:
            Path: The path to the exported session directory.

        Raises:
            ValueError: If the session ID is None or the session does not exist.
            RuntimeError: If the session store is not initialized.
        """
        with self._session_scope() as s:
            if not session_id and self.session_id is not None:
                session_id = self.session_id
            if session_id is None:
                raise ValueError("Session ID cannot be None.")
            conv = s.get(Conversation, session_id)
            if conv is None:
                logger.error(
                    "ValueError: No conversation found for session_id={}",
                    session_id,
                )
                raise ValueError(f"No conversation found for session_id={session_id}")

            out_dir = Path(out_dir) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            rolling_summary = cast(str | None, conv.rolling_summary) or ""
            session_meta = {
                "schema_version": "1.0.0",
                "session_id": conv.id,
                "created_at": conv.created_at.replace(tzinfo=timezone.utc).isoformat(),
                "turn_count": len(conv.turns),
                "rolling_summary": rolling_summary,
                "models": {
                    "embed_model_id": self.rag.embed_model_id,
                    "sparse_model_id": self.rag.sparse_model_id,
                    "rerank_model_id": self.rag.gen_model_id,
                    "gen_model_id": self.rag.gen_model_id,
                },
                "retrieval": {
                    "similarity_top_k": self.rag.retrieve_similarity_top_k,
                    "top_n": self.rag.rerank_top_n,
                },
                "vector_store": {
                    "type": "qdrant",
                    "url": self.rag.qdrant_host,
                    "collection": self.rag.qdrant_collection,
                    "host_dir": str(self.rag.qdrant_col_dir or ""),
                },
            }
            (out_dir / "session.json").write_text(
                json.dumps(session_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            with (out_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
                for t in conv.turns:
                    obj = {
                        "turn_idx": t.idx,
                        "created_at": t.created_at.replace(
                            tzinfo=timezone.utc
                        ).isoformat(),
                        "user_text": t.user_text,
                        "rewritten_query": t.rewritten_query,
                        "assistant_text": t.model_response,
                        "reasoning": t.reasoning,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            self._export_citations(out_dir, conv)
            self._export_transcript(out_dir, conv, rolling_summary)
            self._write_manifest(out_dir)
            return out_dir

    def init_session_store_if_needed(self) -> None:
        """
        Initialize the session store if it has not been initialized yet.
        """
        if self._SessionMaker is None:
            self.init_session_store()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """
        Provide a transactional scope around a series of operations.

        Yields:
            Iterator[Session]: A new database session.

        Raises:
            RuntimeError: If the SessionMaker is not initialized.
        """
        self.init_session_store_if_needed()
        if self._SessionMaker is None:
            raise RuntimeError("SessionMaker is not initialized.")
        session = self._SessionMaker()
        try:
            yield session
        finally:
            session.close()

    def _load_or_create_convo(self, session: Session, session_id: str) -> Conversation:
        """
        Load an existing conversation or create a new one.

        Args:
            session (Session): The database session.
            session_id (str): The ID of the session.

        Returns:
            Conversation: The loaded or created conversation.
        """
        conv = session.get(Conversation, session_id)
        if conv is None:
            conv = Conversation(id=session_id)
            if self.rag.qdrant_collection:
                conv.collection_name = cast(Any, self.rag.qdrant_collection)
            session.add(conv)
            session.commit()
        return conv

    def _get_rolling_summary(self, session_id: str) -> str:
        """
        Get the rolling summary for a conversation.

        Args:
            session_id (str): The ID of the session.

        Returns:
            str: The rolling summary for the conversation.
        """
        with self._session_scope() as s:
            conv = s.get(Conversation, session_id)
            if conv is None:
                return ""
            summary_text = cast(str | None, conv.rolling_summary)
            return summary_text or ""

    def _persist_turn(
        self, session_id: str, user_msg: str, resp: Any, data: dict
    ) -> None:
        """
        Persist a user message and the assistant's response in the database.

        Args:
            session_id (str): The ID of the session.
            user_msg (str): The user message.
            resp (Any): The assistant's response.
            data (dict): Additional data to persist.
        """
        with self._session_scope() as s:
            conv = self._load_or_create_convo(s, session_id)

            meta = getattr(resp, "metadata", {}) or {}
            rewritten = meta.get("query_str") or meta.get("compressed_query_str")

            reasoning = data.get("reasoning")
            next_idx = len(conv.turns)
            t = Turn(
                conversation_id=conv.id,
                idx=next_idx,
                user_text=user_msg,
                rewritten_query=rewritten,
                model_response=data.get("response") or "",
                reasoning=reasoning,
            )
            s.add(t)
            s.flush()

            for src_node in getattr(resp, "source_nodes", []) or []:
                node = getattr(src_node, "node", None)
                meta_node = getattr(node, "metadata", {}) or {}
                filename = (
                    meta_node.get("file_name")
                    or meta_node.get("filename")
                    or meta_node.get("file_path")
                    or meta_node.get("source")
                    or meta_node.get("document_id")
                    or ""
                )
                filetype = (
                    meta_node.get("mimetype")
                    or meta_node.get("filetype")
                    or meta_node.get("content_type")
                    or ""
                )
                file_hash = meta_node.get("file_hash")
                source_kind = meta_node.get("source", "")
                page = meta_node.get("page_label") or meta_node.get("page") or None
                table_meta = meta_node.get("table") or {}
                row_index = table_meta.get("row_index")
                node_id = None
                if node is not None:
                    node_id = getattr(node, "node_id", None) or getattr(
                        node, "id_", None
                    )

                score = (
                    float(getattr(src_node, "score", 0.0))
                    if hasattr(src_node, "score")
                    else None
                )

                s.add(
                    Citation(
                        turn_id=t.id,
                        node_id=str(node_id) if node_id is not None else None,
                        score=score,
                        filename=filename,
                        file_hash=file_hash,
                        filetype=filetype,
                        source=source_kind,
                        page=int(page) if page is not None else None,
                        row=int(row_index) if row_index is not None else None,
                    )
                )

            s.commit()

    def _maybe_update_summary(self, session_id: str, every_n_turns: int = 5) -> None:
        """
        Update the rolling summary for a conversation if the conditions are met.

        Args:
            session_id (str): The ID of the session.
            every_n_turns (int, optional): The frequency of updates. Defaults to 5.
        """
        with self._session_scope() as s:
            conv = s.get(Conversation, session_id)
            if (
                not conv
                or len(conv.turns) == 0
                or (len(conv.turns) % every_n_turns) != 0
            ):
                return

            slice_text = []
            for turn in conv.turns[-every_n_turns:]:
                slice_text.append(
                    f"User: {turn.user_text}\nAssistant: {turn.model_response}"
                )
            prompt = self.rag.summarize_prompt + "\n\n".join(slice_text)

            summary_resp = self.rag.gen_model.complete(prompt)
            existing_summary = cast(str | None, conv.rolling_summary) or ""
            new_summary = (existing_summary + "\n" + summary_resp.text).strip()
            conv.rolling_summary = new_summary
            s.commit()

    def _export_citations(self, out_dir: Path, conv: Conversation) -> None:
        """
        Export citations from a conversation to a Parquet file.

        Args:
            out_dir (Path): The output directory.
            conv (Conversation): The conversation object.
        """
        try:
            rows: list[dict[str, Any]] = []
            for t in conv.turns:
                for c in t.citations:
                    rows.append(
                        {
                            "turn_idx": t.idx,
                            "node_id": c.node_id,
                            "score": c.score,
                            "filename": c.filename,
                            "filetype": c.filetype,
                            "source": c.source,
                            "page": c.page,
                            "row": c.row,
                        }
                    )
            if rows:
                df = pd.DataFrame(rows)
                df.to_parquet(out_dir / "citations.parquet", index=False)
            else:
                empty_columns = [
                    "turn_idx",
                    "node_id",
                    "score",
                    "filename",
                    "filetype",
                    "source",
                    "page",
                    "row",
                ]
                empty_source: dict[str, list[Any]] = {col: [] for col in empty_columns}
                pd.DataFrame(empty_source).to_parquet(
                    out_dir / "citations.parquet", index=False
                )
        except Exception as e:
            logger.warning(
                "Skipping citations.parquet export (pandas/pyarrow not available?): {}",
                e,
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions ordered by creation date (descending).

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        with self._session_scope() as s:
            convs = s.query(Conversation).order_by(Conversation.created_at.desc()).all()
            results = []
            for c in convs:
                title = "New Chat"
                if c.turns:
                    first_turn = c.turns[0]
                    title = (
                        first_turn.user_text[:50] + "..."
                        if len(first_turn.user_text) > 50
                        else first_turn.user_text
                    )

                results.append(
                    {
                        "id": c.id,
                        "created_at": c.created_at.isoformat(),
                        "title": title,
                        "collection": c.collection_name,
                    }
                )
            return results

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get the full message history for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        with self._session_scope() as s:
            conv = s.query(Conversation).filter_by(id=session_id).first()
            if not conv:
                return []

            messages = []
            for t in conv.turns:
                # User message
                messages.append({"role": "user", "content": t.user_text})

                # Assistant message
                sources = []
                for c in t.citations:
                    text_val = ""
                    if c.node_id:
                        try:
                            text_val = self._get_node_text_by_id(c.node_id) or ""
                        except Exception:
                            pass

                    src = {
                        "text": text_val,
                        "preview_text": text_val[:280].strip() if text_val else "",
                        "filename": c.filename,
                        "filetype": c.filetype,
                        "source": c.source,
                        "score": c.score,
                        "page": c.page,
                        "row": c.row,
                        "file_hash": c.file_hash,
                    }
                    sources.append(src)

                msg_entry = {
                    "role": "assistant",
                    "content": t.model_response,
                    "sources": sources,
                }
                if t.reasoning:
                    msg_entry["reasoning"] = t.reasoning
                messages.append(msg_entry)

            return messages

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id (str): The ID of the session to delete.

        Returns:
            bool: True if deleted, False otherwise.
        """
        with self._session_scope() as s:
            conv = s.query(Conversation).filter_by(id=session_id).first()
            if conv:
                s.delete(conv)
                s.commit()
                return True
            return False

    def _export_transcript(
        self, out_dir: Path, conv: Conversation, rolling_summary: str
    ) -> None:
        """
        Export the conversation transcript to a Markdown file.

        Args:
            out_dir (Path): The output directory.
            conv (Conversation): The conversation object.
            rolling_summary (str): The rolling summary of the conversation.
        """
        conv_id = str(conv.id)
        lines = ["# Transcript", f"Session: `{conv_id}`", ""]
        if rolling_summary:
            lines += ["## Rolling Summary", "", rolling_summary, ""]
        for t in conv.turns:
            lines += [
                f"## Turn {t.idx}",
                f"**User**: {t.user_text}",
                f"**Rewritten**: {t.rewritten_query or ''}",
                f"**Assistant**: {t.model_response}",
            ]
            if t.reasoning:
                lines += [
                    f"<details><summary>Reasoning</summary>\n\n{t.reasoning}\n\n</details>"
                ]
            if t.citations:
                lines += ["**Citations (with source excerpts):**"]
                for c in t.citations:
                    loc = (
                        f"page {c.page}"
                        if c.page is not None
                        else (f"row {c.row}" if c.row is not None else "")
                    )
                    header = f"- {c.filename} {loc} (score={c.score})"
                    excerpt = None
                    if c.node_id:
                        excerpt = self._get_node_text_by_id(c.node_id)
                    if excerpt is None:
                        excerpt = "[source text unavailable]"
                    else:
                        excerpt = excerpt.strip()
                        max_chars = 800
                        if len(excerpt) > max_chars:
                            excerpt = excerpt[:max_chars].rstrip() + " …"
                    lines += [
                        header,
                        ">\n> " + "\n> ".join(excerpt.splitlines()) + "\n>",
                    ]
            lines += [""]
        (out_dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_manifest(self, out_dir: Path) -> None:
        """
        Write a manifest file for the exported conversation data.

        Args:
            out_dir (Path): The output directory.
        """

        def sha256_file(p: Path) -> str:
            """
            Compute the SHA256 hash of a file.

            Args:
                p (Path): The path to the file.

            Returns:
                str: The SHA256 hash of the file.
            """
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        manifest = {}
        for name in ["session.json", "messages.jsonl", "transcript.md"]:
            fp = out_dir / name
            if fp.exists():
                manifest[name] = {
                    "sha256": sha256_file(fp),
                    "bytes": fp.stat().st_size,
                }
        parquet_fp = out_dir / "citations.parquet"
        if parquet_fp.exists():
            manifest["citations.parquet"] = {
                "sha256": sha256_file(parquet_fp),
                "bytes": parquet_fp.stat().st_size,
            }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _get_node_text_by_id(self, node_id: str) -> str | None:
        """
        Retrieve the text content of a node by its ID.

        Args:
            node_id (str): The ID of the node.

        Returns:
            str | None: The text content of the node, or None if not found.
        """
        try:
            index = self.rag.index
            if index is None:
                return None
            docstore = getattr(index, "storage_context", None)
            if docstore is not None:
                docstore = getattr(docstore, "docstore", None)
            else:
                docstore = getattr(index, "docstore", None)
            if docstore is None:
                return None
            for getter in ("get_node", "get", "get_document"):
                fn = getattr(docstore, getter, None)
                if callable(fn):
                    try:
                        node = fn(node_id)
                    except Exception:
                        continue
                    if node is None:
                        continue
                    text = getattr(node, "text", None)
                    if isinstance(text, str) and text:
                        return text
                    if (
                        isinstance(node, BaseNode)
                        and hasattr(node, "get_content")
                        and callable(node.get_content)
                    ):
                        content = node.get_content()
                        if isinstance(content, str) and content:
                            return content
        except Exception:
            pass

        try:
            recs = self.rag.qdrant_client.retrieve(
                collection_name=self.rag.qdrant_collection, ids=[node_id]
            )
            if recs:
                payload = getattr(recs[0], "payload", None)
                if isinstance(payload, dict):
                    txt = (
                        payload.get("text")
                        or payload.get("chunk")
                        or payload.get("content")
                    )
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
        except Exception:
            return None
        return None
