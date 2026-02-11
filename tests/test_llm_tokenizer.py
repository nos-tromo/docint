"""Tests for LLM tokenizer loading and prompt function building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from docint.utils.llama_cpp_cfg import (
    build_prompt_functions,
    completion_to_prompt_qwen3,
    messages_to_prompt_qwen3,
)
from docint.utils.model_cfg import load_tokenizer


# ---------------------------------------------------------------------------
# Qwen3 fallback helpers
# ---------------------------------------------------------------------------


class _FakeRole:
    """
    Simulate a ChatMessage role with a .value attribute.
    """

    def __init__(self, value: str) -> None:
        self.value = value


class _FakeMessage:
    """
    Simulate a ChatMessage object from llama-index.
    """

    def __init__(self, role: str, content: str) -> None:
        self.role = _FakeRole(role)
        self.content = content


def test_messages_to_prompt_qwen3_formats_correctly() -> None:
    """
    Test that messages_to_prompt_qwen3 formats messages in ChatML with thinking disabled.
    """
    messages = [
        _FakeMessage("system", "You are helpful."),
        _FakeMessage("user", "Hello"),
    ]
    result = messages_to_prompt_qwen3(messages)
    assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
    assert "<|im_start|>user\nHello<|im_end|>" in result
    assert result.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_completion_to_prompt_qwen3_wraps_completion() -> None:
    """
    Test that completion_to_prompt_qwen3 wraps a string in ChatML format.
    """
    result = completion_to_prompt_qwen3("What is AI?")
    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in result
    assert "<|im_start|>user\nWhat is AI?<|im_end|>" in result
    assert result.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


# ---------------------------------------------------------------------------
# build_prompt_functions
# ---------------------------------------------------------------------------


def test_build_prompt_functions_falls_back_without_tokenizer() -> None:
    """
    Test that build_prompt_functions returns Qwen3 fallbacks when no tokenizer is found.
    """
    messages_fn, completion_fn = build_prompt_functions(
        tokenizer_id=None, model_id=None, cache_dir=None
    )
    assert messages_fn is messages_to_prompt_qwen3
    assert completion_fn is completion_to_prompt_qwen3


@patch("docint.utils.llama_cpp_cfg._load_tokenizer")
def test_build_prompt_functions_uses_tokenizer(mock_load: MagicMock) -> None:
    """
    Test that build_prompt_functions returns tokenizer-based closures when a tokenizer is found.

    Args:
        mock_load (MagicMock): The mock object for the tokenizer loading function.
    """
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "<formatted>"
    mock_load.return_value = fake_tokenizer

    messages_fn, completion_fn = build_prompt_functions(
        tokenizer_id="org/model", model_id=None, cache_dir=None
    )

    # Should NOT be the Qwen3 fallbacks
    assert messages_fn is not messages_to_prompt_qwen3
    assert completion_fn is not completion_to_prompt_qwen3

    # Calling completion_fn should delegate to apply_chat_template
    result = completion_fn("test prompt")
    assert result == "<formatted>"
    fake_tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = fake_tokenizer.apply_chat_template.call_args
    assert call_kwargs.kwargs["tokenize"] is False
    assert call_kwargs.kwargs["add_generation_prompt"] is True
    assert call_kwargs.kwargs["enable_thinking"] is False


@patch("docint.utils.llama_cpp_cfg._load_tokenizer")
def test_build_prompt_functions_messages_fn_converts_roles(
    mock_load: MagicMock,
) -> None:
    """
    Test that the tokenizer-based messages_to_prompt correctly extracts roles.

    Args:
        mock_load (MagicMock): The mock object for the tokenizer loading function.
    """
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "<formatted>"
    mock_load.return_value = fake_tokenizer

    messages_fn, _ = build_prompt_functions(
        tokenizer_id="org/model", model_id=None, cache_dir=None
    )

    messages = [
        _FakeMessage("system", "Be helpful."),
        _FakeMessage("user", "Hi"),
    ]
    messages_fn(messages)

    call_args = fake_tokenizer.apply_chat_template.call_args
    chat_messages = call_args.args[0]
    assert chat_messages == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]


# ---------------------------------------------------------------------------
# load_tokenizer (model_cfg)
# ---------------------------------------------------------------------------


@patch("docint.utils.model_cfg.resolve_hf_cache_path")
@patch("docint.utils.model_cfg.AutoTokenizer")
def test_load_tokenizer_skips_when_cached(
    mock_auto: MagicMock, mock_resolve: MagicMock, tmp_path: Path
) -> None:
    """
    Test that load_tokenizer does not download when tokenizer is already cached.

    Args:
        mock_auto (MagicMock): The mock object for the AutoTokenizer class.
        mock_resolve (MagicMock): The mock object for the cache path resolver function.
        tmp_path (Path): The temporary path fixture.
    """
    mock_resolve.return_value = tmp_path / "cached_tokenizer"

    load_tokenizer("org/model", tmp_path)
    mock_auto.from_pretrained.assert_not_called()


@patch("docint.utils.model_cfg.resolve_hf_cache_path", return_value=None)
@patch("docint.utils.model_cfg.AutoTokenizer")
def test_load_tokenizer_downloads_when_not_cached(
    mock_auto: MagicMock, mock_resolve: MagicMock, tmp_path: Path
) -> None:
    """
    Test that load_tokenizer downloads from HF when not cached locally.

    Args:
        mock_auto (MagicMock): The mock object for the AutoTokenizer class.
        mock_resolve (MagicMock): The mock object for the cache path resolver function.
        tmp_path (Path): The temporary path fixture.
    """
    load_tokenizer("org/model", tmp_path)

    mock_auto.from_pretrained.assert_called_once_with(
        pretrained_model_name_or_path="org/model",
        cache_dir=tmp_path,
        trust_remote_code=True,
    )
