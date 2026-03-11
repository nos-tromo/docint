"""Tests for NER extractor helpers."""

from types import SimpleNamespace
from typing import cast

from docint.utils import ner_extractor as ner_extractor_module


def test_build_gliner_ner_extractor_uses_expanded_default_labels(
    monkeypatch,
) -> None:
    """Default GLiNER labels should use the expanded schema-specific set."""

    seen: dict[str, object] = {}

    class FakeModel:
        """Minimal GLiNER stand-in used for unit testing."""

        def to(self, _device: str) -> "FakeModel":
            """Return the model unchanged when moved to a device."""
            return self

        def predict_entities(
            self,
            text: str,
            labels: list[str],
            *,
            threshold: float,
        ) -> list[dict[str, object]]:
            """Record the requested labels and return a canned prediction."""
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
        lambda _cache, _model_id: None,
    )
    monkeypatch.setattr(
        ner_extractor_module.GLiNER,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeModel(),
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
