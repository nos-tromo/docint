"""OpenAI-compatible model helpers and budget-aware embedding wrappers."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam

from docint.utils.env_cfg import (
    OpenAIConfig,
    load_model_env,
    load_openai_env,
    load_path_env,
)
from docint.utils.llm_sanitize import strip_reasoning


class EmbeddingInputTooLongError(RuntimeError):
    """Raised when an embedding input still exceeds the context window."""


class LocalOpenAI(LlamaIndexOpenAI):
    """Subclass of LlamaIndex's OpenAI that handles local models with unknown names gracefully."""

    def __init__(
        self, context_window: int = 4096, num_output: int = 256, **kwargs: Any
    ) -> None:
        """Initialize the LocalOpenAI instance.

        Args:
            context_window (int): The context window size for the model. Defaults to 4096.
            num_output (int): Tokens reserved for the model response in prompt-helper
                calculations.  Defaults to 256 (llama_index default).
        """
        super().__init__(**kwargs)
        self._context_window = context_window
        self._num_output = num_output

    @property
    def metadata(self) -> LLMMetadata:
        """Return model metadata, always honouring the configured context window.

        llama_index's built-in model registry only knows about first-party
        OpenAI model names. For locally-served models (Ollama, vLLM,
        llama-server) the registry either raises ``ValueError`` or returns
        a stale context-window size.  We therefore always override
        ``context_window`` and ``num_output`` with the values the caller
        provided (which ultimately come from the ``OPENAI_CTX_WINDOW`` /
        ``OPENAI_NUM_OUTPUT`` env vars or their defaults).

        Returns:
            LLMMetadata: The metadata for the model.
        """
        try:
            base = super().metadata
            return LLMMetadata(
                context_window=self._context_window,
                num_output=self._num_output,
                is_chat_model=base.is_chat_model,
                is_function_calling_model=base.is_function_calling_model,
                model_name=base.model_name,
            )
        except ValueError:
            # Fallback for unknown models (e.g. local paths for llama-server)
            return LLMMetadata(
                context_window=self._context_window,
                num_output=self._num_output,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )


class BudgetedOpenAIEmbedding(OpenAIEmbedding):
    """Embedding client that fails loudly when an oversize input reaches it.

    The pre-embed re-splitter (:mod:`docint.utils.embed_chunking`) is the
    only supported defense against oversize inputs: any request that
    still overflows the provider's context window indicates the splitter
    missed the input, and silently truncating here would corrupt the
    vector store with prefix-only embeddings. Instead, this wrapper
    detects the overflow and raises
    :class:`EmbeddingInputTooLongError` so ingestion aborts and the
    operator can diagnose the gap (usually a misconfigured
    ``EMBED_CTX_TOKENS`` relative to the provider's true limit).
    """

    _context_window: int = PrivateAttr(default=8192)

    def __init__(
        self,
        *args: Any,
        context_window: int = 8192,
        **kwargs: Any,
    ) -> None:
        """Initialize the embedding client with a configured context budget.

        Args:
            *args (Any): Positional arguments forwarded to ``OpenAIEmbedding``.
            context_window (int): Configured context window for the embedding model.
            **kwargs (Any): Keyword arguments forwarded to ``OpenAIEmbedding``.
        """
        super().__init__(*args, **kwargs)
        self._context_window = max(1, int(context_window))

    @staticmethod
    def _extract_context_limit_details(message: str) -> tuple[int | None, int | None]:
        """Parse context-limit details from a provider error message.

        Args:
            message (str): Provider error message.

        Returns:
            tuple[int | None, int | None]: Parsed model limit and input token count.
        """
        patterns = (
            (
                r"maximum context length is\s*(\d+)\s*tokens.*?contains at least\s*(\d+)\s*input tokens",
                (1, 2),
            ),
            (
                r"passed\s*(\d+)\s*input tokens.*?context length is only\s*(\d+)\s*tokens",
                (2, 1),
            ),
            (
                r"passed\s*(\d+)\s*input tokens.*?maximum input length of\s*(\d+)\s*tokens",
                (2, 1),
            ),
        )
        for pattern, groups in patterns:
            match = re.search(
                pattern,
                message,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                model_limit_group, input_tokens_group = groups
                return int(match.group(model_limit_group)), int(
                    match.group(input_tokens_group)
                )
        return None, None

    @classmethod
    def _is_context_limit_error(cls, exc: Exception) -> bool:
        """Return whether an exception indicates an oversized embedding request.

        Recognizes the common phrasings across OpenAI-compatible providers:
        OpenAI itself (``"maximum context length" ... "input tokens"``),
        vLLM (``"context length is only" ... "input tokens"`` or
        ``"maximum input length"``), and ollama / generic providers
        (``"the input length exceeds the context length"``, without any
        token counts). Guarded so unrelated failures (connection refused,
        rate limit, auth) are NOT reclassified as context-limit overflows.

        Args:
            exc (Exception): Raised exception.

        Returns:
            bool: ``True`` when the exception is a context-limit failure.
        """
        message = str(exc).lower()
        return (
            (
                (
                    "maximum context length" in message
                    or "context length is only" in message
                )
                and "input tokens" in message
            )
            or ("maximum input length" in message and "input tokens" in message)
            or "context_length_exceeded" in message
            or ("input length exceeds" in message and "context length" in message)
        )

    def _raise_budget_overflow(
        self,
        exc: Exception,
        *,
        texts: list[str] | None = None,
    ) -> EmbeddingInputTooLongError:
        """Build a diagnostic ``EmbeddingInputTooLongError`` from *exc*.

        Ollama reports context overflow without token counts (so
        ``provider_input_tokens`` ends up ``None``), which makes the raw
        exception difficult to correlate with the offending batch. When
        the caller supplies the embed-batch texts, this helper records
        the batch size, the longest payload, and the total character
        count so operators can pinpoint the slipped input without
        re-deriving the batch.

        Args:
            exc (Exception): Provider exception that tripped the
                context-limit detector.
            texts (list[str] | None): Embed-batch texts the caller was
                trying to embed. Optional — when omitted, the error
                message falls back to the provider-level detail only.

        Returns:
            EmbeddingInputTooLongError: Ready to be raised by the caller.
        """
        model_limit, input_tokens = self._extract_context_limit_details(str(exc))
        batch_stats = ""
        if texts:
            lens = [len(t) for t in texts]
            batch_stats = (
                f", batch_size={len(texts)}"
                f", max_text_chars={max(lens)}"
                f", total_chars={sum(lens)}"
            )
        return EmbeddingInputTooLongError(
            "Embedding input exceeded context budget: "
            f"configured={self._context_window}, "
            f"provider_limit={model_limit}, "
            f"provider_input_tokens={input_tokens}{batch_stats}. "
            "Lower EMBED_CTX_TOKENS to match the provider's serving "
            "ceiling, or raise ollama's num_ctx via a Modelfile "
            "(PARAMETER num_ctx N) — see docs/deployment.md."
        )

    def get_text_embeddings_strict(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, raising on context overflow.

        Args:
            texts (list[str]): Text batch to embed.

        Returns:
            list[list[float]]: Embedding vectors aligned to input order.

        Raises:
            EmbeddingInputTooLongError: When any text triggers the
                provider's context-limit error. No retry, no truncation.
        """
        try:
            return super()._get_text_embeddings(texts)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            raise self._raise_budget_overflow(exc, texts=texts) from exc

    async def aget_text_embeddings_strict(self, texts: list[str]) -> list[list[float]]:
        """Async variant of :meth:`get_text_embeddings_strict`.

        Args:
            texts (list[str]): Text batch to embed.

        Returns:
            list[list[float]]: Embedding vectors aligned to input order.

        Raises:
            EmbeddingInputTooLongError: When any text triggers the
                provider's context-limit error.
        """
        try:
            return await super()._aget_text_embeddings(texts)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            raise self._raise_budget_overflow(exc, texts=texts) from exc

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, raising loudly on context overflow.

        Args:
            texts (list[str]): Text batch.

        Returns:
            list[list[float]]: Embedding vectors in input order.

        Raises:
            EmbeddingInputTooLongError: When the provider reports an
                oversize input. No retry, no silent truncation.
        """
        try:
            return super()._get_text_embeddings(texts)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            raise self._raise_budget_overflow(exc, texts=texts) from exc

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async batch embedding that raises loudly on context overflow.

        Args:
            texts (list[str]): Text batch.

        Returns:
            list[list[float]]: Embedding vectors in input order.

        Raises:
            EmbeddingInputTooLongError: When the provider reports an
                oversize input. No retry, no silent truncation.
        """
        try:
            return await super()._aget_text_embeddings(texts)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            raise self._raise_budget_overflow(exc, texts=texts) from exc


def get_openai_reasoning_effort(
    openai_config: OpenAIConfig,
    *,
    enabled: bool | None = None,
) -> str | None:
    """Return the OpenAI reasoning effort to request for chat completions.

    Reasoning is enabled whenever the request scope asks for it, regardless of
    which OpenAI-compatible provider is serving the model.

    Args:
        openai_config (OpenAIConfig): Parsed OpenAI environment configuration.
        enabled (bool | None): Optional per-request override for whether reasoning should be
            requested. When omitted, the config default is used.

    Returns:
        str | None: The configured reasoning effort string when enabled, otherwise ``None``.
    """
    if enabled is None:
        enabled = openai_config.thinking_enabled

    if not enabled:
        return None
    return openai_config.thinking_effort


@dataclass
class OpenAIPipeline:
    """Pipeline for text generation and image processing using OpenAI-compatible APIs."""

    client: OpenAI = field(init=False)
    vision_client: OpenAI = field(init=False)
    text_model_id: str = field(init=False)
    vision_model_id: str = field(init=False)
    prompt_dir: Path | None = field(init=False)
    reasoning_effort: str | None = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to load configurations."""
        _model_config = load_model_env()
        self.text_model_id = _model_config.text_model.removesuffix(".gguf")
        self.vision_model_id = _model_config.vision_model.removesuffix(".gguf")

        _openai_config = load_openai_env()
        api_key = _openai_config.api_key
        api_base = _openai_config.api_base
        max_retries = _openai_config.max_retries
        self.seed = _openai_config.seed
        timeout = _openai_config.timeout
        self.temperature = _openai_config.temperature
        self.top_p = _openai_config.top_p
        self.reasoning_effort = get_openai_reasoning_effort(_openai_config)

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.vision_client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=_openai_config.max_retries,
        )

        self.prompt_dir = load_path_env().prompts

    def load_prompt(self, kw: str = "system") -> str:
        """Load a prompt from the prompts directory based on the given keyword.

        Args:
            kw (str, optional): The keyword to identify the prompt file. Defaults to "system".

        Returns:
            str: The content of the prompt file.

        Raises:
            RuntimeError: If the prompt directory is not set.
            FileNotFoundError: If the prompt file for the given keyword does not exist.
        """
        if self.prompt_dir is None:
            logger.error("RuntimeError: Prompt directory is not set.")
            raise RuntimeError("Prompt directory is not set.")

        prompt_path = self.prompt_dir / f"{kw}.txt"
        if not prompt_path.is_file():
            logger.error(
                "FileNotFoundError: Prompt file for keyword '{}' not found.", kw
            )
            raise FileNotFoundError(f"Prompt file for keyword '{kw}' not found.")
        with open(prompt_path, "r", encoding="utf-8") as f:
            logger.info("Loaded prompt from '{}'", prompt_path)
            return f.read()

    def call_chat(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call OpenAI Chat completion.

        Args:
            prompt (str): The user prompt.
            system_prompt (str | None): Optional system prompt.

        Returns:
            str: The response text.

        Raises:
            RuntimeError: If the chat inference fails.
        """
        try:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            request_kwargs: dict[str, Any] = {}
            if self.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            response = self.client.chat.completions.create(
                model=self.text_model_id,
                messages=messages,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                **request_kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Error during chat inference: {}", e)
            raise RuntimeError(f"Chat inference failed: {e}")

    def call_vision(
        self, prompt: str, img_base64: str, mime_type: str = "image/jpeg"
    ) -> str:
        """Call OpenAI (or compatible) Vision model.

        Args:
            prompt (str): The prompt text.
            img_base64 (str): Base64 encoded image string.
            mime_type (str): MIME type of the encoded image (e.g. ``image/png``).
                Defaults to ``image/jpeg``.

        Returns:
            str: The model's response.

        Raises:
            RuntimeError: If the vision inference fails.
        """
        try:
            content_parts: list[ChatCompletionContentPartParam] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{img_base64}"},
                },
            ]
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": content_parts}
            ]

            request_kwargs: dict[str, Any] = {}
            if self.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            response = self.vision_client.chat.completions.create(
                model=self.vision_model_id,
                messages=messages,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                **request_kwargs,
            )
            raw = response.choices[0].message.content or ""
            clean, captured = strip_reasoning(raw)
            if captured:
                logger.debug(
                    "Stripped {} chars of reasoning from vision response",
                    len(captured),
                )
            return clean
        except Exception as e:
            logger.error("Error during vision inference: {}", e)
            raise RuntimeError(f"Vision inference failed: {e}")
