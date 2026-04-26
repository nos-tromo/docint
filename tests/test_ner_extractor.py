"""Tests for NER extractor helpers."""

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, cast

import pytest

from docint.utils import ner_extractor as ner_extractor_module


class FakeTokenizer:
    """Minimal tokenizer that counts whitespace-delimited tokens."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> list[str]:
        """Return one pseudo-token per whitespace-delimited word.

        Args:
            text: Input text to tokenize.
            add_special_tokens: Ignored in this fake tokenizer.
            truncation: Ignored in this fake tokenizer.

        Returns:
            One pseudo-token string per word.
        """
        del add_special_tokens, truncation
        return text.split()


class FakeWordsSplitter:
    """Words splitter counting whitespace-delimited tokens, mirroring FakeTokenizer.

    Using the same split boundaries as FakeTokenizer ensures existing test
    assertions on chunk boundaries remain correct when the new words_splitter
    code path is exercised.
    """

    _PATTERN = re.compile(r"\S+")

    def __call__(self, text: str):
        """Yield (token, start, end) triples for each whitespace-delimited span.

        Args:
            text: Input text to split.

        Yields:
            tuple[str, int, int]: Token string and its character offsets.
        """
        for match in self._PATTERN.finditer(text):
            yield match.group(), match.start(), match.end()


def _fake_gliner_runtime(
    *,
    max_len: int = 768,
) -> tuple[Any, Any]:
    """Build fake GLiNER config and data processor objects.

    Args:
        max_len: Configured GLiNER context window.

    Returns:
        Tuple of fake config and fake data processor objects.
    """
    config = SimpleNamespace(max_len=max_len)
    data_processor = SimpleNamespace(
        transformer_tokenizer=FakeTokenizer(),
        words_splitter=FakeWordsSplitter(),
    )
    return config, data_processor


@pytest.fixture(autouse=True)
def clear_gliner_runtime_cache() -> Generator[None, None, None]:
    """Ensure GLiNER runtime cache state does not leak across tests."""
    ner_extractor_module._GLINER_RUNTIME_CACHE.clear()
    yield
    ner_extractor_module._GLINER_RUNTIME_CACHE.clear()


def test_build_gliner_ner_extractor_uses_expanded_default_labels(
    monkeypatch,
) -> None:
    """Default GLiNER labels should use the expanded schema-specific set.

    Args:
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    seen: dict[str, object] = {}

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            """Initialize fake model runtime metadata."""
            self.config, self.data_processor = _fake_gliner_runtime()

        def to(self, _device: str) -> "FakeModel":
            """Return the model unchanged when moved to a device.

            Args:
                _device: The device to move the model to (ignored in this fake).

            Returns:
                The model instance itself, unchanged.
            """
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Record the requested labels and return a canned prediction.

            Args:
                text: The input text for entity prediction (ignored in this fake).
                labels: The list of entity labels to predict, which should include the expanded set.
                threshold: The confidence threshold for predictions (ignored in this fake).

            Returns:
                A hardcoded list of predicted entities.
            """
            seen["text"] = text
            seen["labels"] = list(labels)
            seen["threshold"] = threshold
            return [{"text": "Alice", "label": "person", "score": 0.9}]

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor()
    entities, relations = extractor("Alice met Bob in Berlin")

    assert entities == [{"text": "Alice", "type": "person", "score": 0.9}]
    assert relations == []
    assert seen["text"] == "Alice met Bob in Berlin"
    assert seen["threshold"] == 0.3
    labels = set(cast(list[str], seen["labels"]))
    assert {"bank_account", "fac", "group", "loc", "org", "phone", "weapon"} <= labels
    assert "organization" not in labels
    assert "location" not in labels


def test_build_gliner_ner_extractor_rewrites_backbone_to_local_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Offline GLiNER loads should use local snapshot paths only.

    Args:
        tmp_path: pytest fixture providing a temporary directory for test files.
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    gliner_snapshot = tmp_path / "gliner"
    gliner_snapshot.mkdir()
    (gliner_snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (gliner_snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (gliner_snapshot / "pytorch_model.bin").write_bytes(b"weights")
    (gliner_snapshot / "gliner_config.json").write_text(
        json.dumps({"model_name": "microsoft/deberta-v3-large"}),
        encoding="utf-8",
    )

    deberta_snapshot = tmp_path / "deberta"
    deberta_snapshot.mkdir()
    (deberta_snapshot / "config.json").write_text("{}", encoding="utf-8")

    seen: dict[str, object] = {}

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            """Initialize fake model runtime metadata."""
            self.config, self.data_processor = _fake_gliner_runtime()

        def to(self, _device: str) -> "FakeModel":
            """Return the model unchanged when moved to a device.

            Args:
                _device: The device to move the model to (ignored in this fake).

            Returns:
                The model instance itself, unchanged.
            """
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Return a canned prediction.

            Args:
                text: The input text for entity prediction (ignored in this fake).
                labels: The list of entity labels to predict (ignored in this fake).
                threshold: The confidence threshold for predictions (ignored in this fake).

            Returns:
                A hardcoded list of predicted entities.
            """
            del text, labels, threshold
            return []

    def _fake_from_pretrained(model_id: str, *, local_files_only: bool) -> FakeModel:
        """Simulate GLiNER's from_pretrained method, recording the requested model ID and local_files_only flag.

        Args:
            model_id (str): The identifier of the model to load, expected to be a Hugging Face repo ID.
            local_files_only (bool): Whether to restrict loading to local files only.

        Returns:
            FakeModel: A stand-in model instance for testing purposes.
        """
        config = json.loads(
            (Path(model_id) / "gliner_config.json").read_text(encoding="utf-8")
        )
        seen["model_id"] = model_id
        seen["local_files_only"] = local_files_only
        seen["model_name"] = config["model_name"]
        return FakeModel()

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="gliner-community/gliner_large-v2.5"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache=tmp_path),
    )

    def _fake_resolve_hf_cache_path(cache_dir: Path, repo_id: str) -> Path | None:
        """Simulate resolution of Hugging Face cache paths, returning the appropriate local snapshot
        path for known repo IDs.

        Args:
            cache_dir (Path): The base directory for Hugging Face cache (ignored in this fake).
            repo_id (str): The identifier of the model repository.

        Returns:
            Path | None: The local snapshot path if known, otherwise None.
        """
        del cache_dir
        if repo_id == "gliner-community/gliner_large-v2.5":
            return gliner_snapshot
        if repo_id == "microsoft/deberta-v3-large":
            return deberta_snapshot
        return None

    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        _fake_resolve_hf_cache_path,
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=_fake_from_pretrained),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor()
    entities, relations = extractor("Alice met Bob in Berlin")

    assert entities == []
    assert relations == []
    assert seen["local_files_only"] is True
    assert Path(cast(str, seen["model_id"])) != gliner_snapshot
    assert seen["model_name"] == str(deberta_snapshot)


def test_build_gliner_ner_extractor_respects_requested_cpu_device(
    monkeypatch,
) -> None:
    """GLiNER should stay on CPU when the caller requests ``cpu``.

    Args:
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    move_calls: list[str] = []

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            """Initialize fake model runtime metadata."""
            self.config, self.data_processor = _fake_gliner_runtime()

        def to(self, device: str) -> "FakeModel":
            """Record requested device moves.

            Args:
                device: Requested execution device.

            Returns:
                FakeModel: The model instance itself.
            """
            move_calls.append(device)
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Return no entities for the provided text.

            Args:
                text: Input text.
                labels: Requested labels.
                threshold: Confidence threshold.

            Returns:
                list[dict[str, object]]: Empty predictions.
            """
            del text, labels, threshold
            return []

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "device_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_built",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor(device="cpu")
    entities, relations = extractor("Alice met Bob in Berlin")

    assert entities == []
    assert relations == []
    assert move_calls == []


def test_build_gliner_ner_extractor_moves_to_requested_cuda_device(
    monkeypatch,
) -> None:
    """GLiNER should honor an explicit CUDA device selection.

    Args:
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    move_calls: list[str] = []

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            """Initialize fake model runtime metadata."""
            self.config, self.data_processor = _fake_gliner_runtime()

        def to(self, device: str) -> "FakeModel":
            """Record requested device moves.

            Args:
                device: Requested execution device.

            Returns:
                FakeModel: The model instance itself.
            """
            move_calls.append(device)
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Return no entities for the provided text.

            Args:
                text: Input text.
                labels: Requested labels.
                threshold: Confidence threshold.

            Returns:
                list[dict[str, object]]: Empty predictions.
            """
            del text, labels, threshold
            return []

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "device_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_built",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor(device="cuda:1")
    entities, relations = extractor("Alice met Bob in Berlin")

    assert entities == []
    assert relations == []
    assert move_calls == ["cuda:1"]


def test_build_gliner_ner_extractor_chunks_long_inputs_on_sentence_boundaries(
    monkeypatch,
) -> None:
    """Long inputs should be split into sentence-sized GLiNER requests.

    Args:
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    seen_texts: list[str] = []

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            """Initialize fake model runtime metadata."""
            self.config, self.data_processor = _fake_gliner_runtime(max_len=7)

        def to(self, _device: str) -> "FakeModel":
            """Return the model unchanged when moved to a device.

            Args:
                _device: The device to move the model to (ignored in this fake).

            Returns:
                The model instance itself, unchanged.
            """
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Record chunk inputs and return one entity per chunk.

            Args:
                text: The input text for entity prediction, expected to be a sentence-sized chunk.
                labels: The list of entity labels to predict (ignored in this fake).
                threshold: The confidence threshold for predictions (ignored in this fake).

            Returns:
                A hardcoded list of predicted entities, one per chunk.
            """
            del labels, threshold
            seen_texts.append(text)
            return [{"text": text.split()[0], "label": "person", "score": 0.9}]

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor()
    entities, relations = extractor(
        "Alice met Bob in Berlin. Carol visited Paris yesterday. Delta leads Acme now."
    )

    assert seen_texts == [
        "Alice met Bob in Berlin.",
        "Carol visited Paris yesterday.",
        "Delta leads Acme now.",
    ]
    assert [entity["text"] for entity in entities] == ["Alice", "Carol", "Delta"]
    assert relations == []


def test_build_gliner_ner_extractor_falls_back_to_word_chunks_for_long_sentence(
    monkeypatch,
) -> None:
    """An oversized sentence should fall back to word-boundary chunking.

    Args:
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    seen_texts: list[str] = []

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            """Initialize fake model runtime metadata."""
            self.config, self.data_processor = _fake_gliner_runtime(max_len=5)

        def to(self, _device: str) -> "FakeModel":
            """Return the model unchanged when moved to a device.

            Args:
                _device: The device to move the model to (ignored in this fake).

            Returns:
                The model instance itself, unchanged.
            """
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Record chunk inputs and return no entities.

            Args:
                text: The input text for entity prediction, expected to be a chunk of words.
                labels: The list of entity labels to predict (ignored in this fake).
                threshold: The confidence threshold for predictions (ignored in this fake).

            Returns:
                An empty list of predicted entities, simulating a case where chunking is needed.
            """
            del labels, threshold
            seen_texts.append(text)
            return []

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor()
    entities, relations = extractor("alpha beta gamma delta epsilon zeta eta")

    assert entities == []
    assert relations == []
    assert seen_texts == [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta",
    ]


def test_chunk_text_for_gliner_uses_gliner_word_count_not_bpe_count() -> None:
    """Chunking must count in words_splitter units, not BPE sub-word units.

    This is a regression test for the truncation warning:
      'Sentence of length N has been truncated to 768'
    which occurred because the BPE token count for CJK or punctuation-dense text
    is far smaller than GLiNER's internal word count, allowing oversized chunks
    through the budget guard.
    """

    class AlwaysOneBPETokenizer:
        """Tokenizer that always returns 1 BPE token regardless of input length."""

        def encode(
            self,
            text: str,
            *,
            add_special_tokens: bool = False,
            truncation: bool = False,
        ) -> list[str]:
            """Return a single pseudo-token for any input.

            Args:
                text: Input text (ignored — always returns one token).
                add_special_tokens: Ignored.
                truncation: Ignored.

            Returns:
                A single-element list, simulating extreme BPE under-counting.
            """
            del add_special_tokens, truncation, text
            return ["TOKEN"]

    class CharacterWordsSplitter:
        """Words splitter yielding one token per non-space character."""

        def __call__(self, text: str):
            """Yield (char, start, end) for each non-space character.

            Args:
                text: Input text to split.

            Yields:
                tuple[str, int, int]: Single character and its offsets.
            """
            for i, ch in enumerate(text):
                if ch != " ":
                    yield ch, i, i + 1

    # Budget: 3 GLiNER words per chunk.
    # Input: 9 non-space characters = 9 GLiNER words.
    # BPE-only (without fix): 1 token for the whole string ≤ 3 → single chunk → truncation.
    # With fix (words_splitter): 9 > 3 → split into 3 chunks of 3 characters each.
    chunks = ner_extractor_module._chunk_text_for_gliner(
        text="abcdefghi",
        max_tokens=3,
        tokenizer=AlwaysOneBPETokenizer(),
        words_splitter=CharacterWordsSplitter(),
    )

    splitter = CharacterWordsSplitter()
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}: {chunks}"
    for chunk in chunks:
        word_count = sum(1 for _ in splitter(chunk))
        assert word_count <= 3, (
            f"Chunk '{chunk}' has {word_count} words, exceeds budget of 3"
        )

    # Demonstrate the pre-fix failure: without words_splitter the BPE tokenizer
    # always returns 1 token regardless of text length, so the budget guard never
    # fires and the whole 9-character string is packed into one oversized chunk.
    pre_fix_chunks = ner_extractor_module._chunk_text_for_gliner(
        text="abcdefghi",
        max_tokens=3,
        tokenizer=AlwaysOneBPETokenizer(),
        words_splitter=None,
    )
    assert len(pre_fix_chunks) == 1, (
        f"Expected 1 oversized chunk without words_splitter, got {len(pre_fix_chunks)}"
    )


def test_build_gliner_ner_extractor_serializes_concurrent_model_access(
    monkeypatch,
) -> None:
    """Concurrent extractor calls should not trip GLiNER borrow errors.

    Args:
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    class FakeModel:
        """GLiNER stand-in that fails when called concurrently."""

        def __init__(self) -> None:
            self.config, self.data_processor = _fake_gliner_runtime()
            self._borrow_lock = threading.Lock()

        def to(self, _device: str) -> "FakeModel":
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            del labels, threshold
            if not self._borrow_lock.acquire(blocking=False):
                raise RuntimeError("Already borrowed")
            try:
                time.sleep(0.05)
                return [{"text": text, "label": "person", "score": 0.9}]
            finally:
                self._borrow_lock.release()

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    extractor = ner_extractor_module.build_gliner_ner_extractor()
    results: list[tuple[list[dict], list[dict]]] = []
    errors: list[BaseException] = []

    def _run(text: str) -> None:
        try:
            results.append(extractor(text))
        except BaseException as exc:  # pragma: no cover - defensive branch
            errors.append(exc)

    threads = [
        threading.Thread(target=_run, args=("Alice met Bob",)),
        threading.Thread(target=_run, args=("Carol visited Paris",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    extracted_texts = sorted(result[0][0]["text"] for result in results)
    assert extracted_texts == ["Alice met Bob", "Carol visited Paris"]


def test_build_gliner_ner_extractor_reuses_loaded_runtime_across_calls(
    monkeypatch,
) -> None:
    """Repeated extractor construction should reuse the same GLiNER runtime."""

    load_calls: list[tuple[str, bool]] = []

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def __init__(self) -> None:
            self.config, self.data_processor = _fake_gliner_runtime()

        def to(self, _device: str) -> "FakeModel":
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            del text, labels, threshold
            return []

    def _fake_from_pretrained(model_id: str, *, local_files_only: bool) -> FakeModel:
        load_calls.append((model_id, local_files_only))
        return FakeModel()

    fake_gliner_class = SimpleNamespace(from_pretrained=_fake_from_pretrained)

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="urchade/gliner_small-v2.1"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache="/tmp"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: fake_gliner_class,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        ner_extractor_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    first_extractor = ner_extractor_module.build_gliner_ner_extractor()
    second_extractor = ner_extractor_module.build_gliner_ner_extractor()

    assert first_extractor("Alice met Bob") == ([], [])
    assert second_extractor("Carol visited Paris") == ([], [])
    assert load_calls == [("urchade/gliner_small-v2.1", False)]


def test_build_gliner_ner_extractor_fails_fast_when_offline_model_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Offline mode should not fall through to Hugging Face repo resolution.

    Args:
        tmp_path: pytest fixture providing a temporary directory for test files.
        monkeypatch: pytest fixture for safely patching functions and environment variables.
    """

    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        ner_extractor_module,
        "load_model_env",
        lambda: SimpleNamespace(ner_model="gliner-community/gliner_large-v2.5"),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "load_path_env",
        lambda: SimpleNamespace(hf_hub_cache=tmp_path),
    )
    monkeypatch.setattr(
        ner_extractor_module,
        "resolve_hf_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        ner_extractor_module,
        "_get_gliner_class",
        lambda: SimpleNamespace(
            from_pretrained=lambda model_id, *, local_files_only: calls.append(
                (model_id, local_files_only)
            )
        ),
    )

    try:
        ner_extractor_module.build_gliner_ner_extractor()
    except FileNotFoundError as exc:
        assert "gliner-community/gliner_large-v2.5" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected build_gliner_ner_extractor() to fail")

    assert calls == []
