"""Tests for NER extractor helpers."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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
    data_processor = SimpleNamespace(transformer_tokenizer=FakeTokenizer())
    return config, data_processor


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
