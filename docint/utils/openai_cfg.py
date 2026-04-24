"""OpenAI-compatible model helpers and truncation-aware embedding wrappers."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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


class TruncatingOpenAIEmbedding(OpenAIEmbedding):
    """Retry oversized embedding requests with progressively truncated input text."""

    _warning_callback: Callable[[str], None] | None = PrivateAttr(default=None)
    _context_window: int = PrivateAttr(default=8192)

    def __init__(
        self,
        *args: Any,
        context_window: int = 8192,
        warning_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the embedding client with context-window-aware truncation.

        Args:
            *args (Any): Positional arguments forwarded to ``OpenAIEmbedding``.
            context_window (int): Configured context window for the embedding model.
            warning_callback (Callable[[str], None] | None): Optional callback for truncation warnings.
            **kwargs (Any): Keyword arguments forwarded to ``OpenAIEmbedding``.
        """
        super().__init__(*args, **kwargs)
        self._context_window = max(1, int(context_window))
        self._warning_callback = warning_callback

    def set_warning_callback(self, callback: Callable[[str], None] | None) -> None:
        """Register a callback that receives truncation warnings.

        Args:
            callback (Callable[[str], None] | None): Callable invoked for each truncation warning, or ``None``.
        """
        self._warning_callback = callback

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
        rate limit, auth) do NOT trigger the truncation-retry path.

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

    def _emit_truncation_warning(
        self,
        *,
        original_chars: int,
        truncated_chars: int,
        model_limit: int | None,
        input_tokens: int | None,
    ) -> None:
        """Log and forward a truncation warning.

        Args:
            original_chars (int): Original payload size in characters.
            truncated_chars (int): Truncated payload size in characters.
            model_limit (int | None): Parsed model limit in tokens, when available.
            input_tokens (int | None): Parsed input token count, when available.
        """
        detail = ""
        if model_limit is not None and input_tokens is not None:
            detail = f" Provider reported {input_tokens} input tokens against a {model_limit}-token limit."
        message = (
            "Warning: truncated oversized embedding input "
            f"from {original_chars} to {truncated_chars} characters to fit the model context window."
            f"{detail}"
        )
        logger.warning(message)
        if self._warning_callback is None:
            return
        try:
            self._warning_callback(message)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.debug("Failed to deliver embedding truncation warning: {}", exc)

    def _emit_skip_warning(
        self,
        *,
        original_chars: int,
        truncated_chars: int,
        retries: int,
        model_limit: int | None,
        input_tokens: int | None,
    ) -> str:
        """Log and forward a warning for an input that must be skipped.

        Args:
            original_chars (int): Original payload size in characters.
            truncated_chars (int): Final candidate size in characters.
            retries (int): Number of truncation retries attempted.
            model_limit (int | None): Parsed model limit in tokens, when available.
            input_tokens (int | None): Parsed input token count, when available.

        Returns:
            str: The emitted warning message.
        """
        detail = ""
        if model_limit is not None and input_tokens is not None:
            detail = (
                " Provider reported "
                f"{input_tokens} input tokens against a "
                f"{model_limit}-token limit."
            )
        message = (
            "Warning: skipping embedding input after "
            f"{retries} truncation attempt(s). Final candidate length was "
            f"{truncated_chars} characters after starting at "
            f"{original_chars} characters, but the payload still exceeded "
            f"the model context window.{detail}"
        )
        logger.warning(message)
        if self._warning_callback is None:
            return message
        try:
            self._warning_callback(message)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.debug("Failed to deliver embedding skip warning: {}", exc)
        return message

    def _truncate_text(self, text: str, exc: Exception) -> str:
        """Return a smaller text candidate after a context-limit failure.

        Args:
            text (str): Current embedding payload.
            exc (Exception): Exception raised for the current payload.

        Returns:
            str: A shorter candidate string.
        """
        model_limit, input_tokens = self._extract_context_limit_details(str(exc))
        effective_limit = min(model_limit or self._context_window, self._context_window)
        reserve = min(256, max(32, effective_limit // 16))
        target_tokens = max(1, effective_limit - reserve)

        if input_tokens is not None and input_tokens > 0:
            target_chars = int(len(text) * (target_tokens / input_tokens))
        else:
            target_chars = int(len(text) * 0.8)

        target_chars = min(len(text) - 1, max(1, target_chars))
        candidate = text[:target_chars].rstrip()
        boundary = max(
            candidate.rfind("\n\n"), candidate.rfind("\n"), candidate.rfind(" ")
        )
        if boundary > max(32, target_chars // 2):
            candidate = candidate[:boundary].rstrip()

        if candidate and candidate != text:
            return candidate

        fallback_chars = max(1, len(text) - max(64, len(text) // 5))
        return text[:fallback_chars].rstrip() or text[:1]

    def _embed_text_with_truncation(
        self, text: str, n_retries: int = 24
    ) -> list[float]:
        """Embed a single text, truncating and retrying on context-limit errors.

        Args:
            text (str): Input text to embed.
            n_retries (int): Maximum number of truncation attempts before giving up.

        Returns:
            list[float]: The embedding vector.
        """
        original_chars = len(text)
        candidate = text
        last_exc: Exception | None = None

        for attempt in range(1, n_retries + 1):
            try:
                embedding = super()._get_text_embedding(candidate)
                if candidate != text:
                    model_limit, input_tokens = self._extract_context_limit_details(
                        str(last_exc) if last_exc is not None else ""
                    )
                    self._emit_truncation_warning(
                        original_chars=original_chars,
                        truncated_chars=len(candidate),
                        model_limit=model_limit,
                        input_tokens=input_tokens,
                    )
                return embedding
            except Exception as exc:
                if not self._is_context_limit_error(exc):
                    raise
                last_exc = exc
                next_candidate = self._truncate_text(candidate, exc)
                if not next_candidate or next_candidate == candidate:
                    model_limit, input_tokens = self._extract_context_limit_details(
                        str(exc)
                    )
                    message = self._emit_skip_warning(
                        original_chars=original_chars,
                        truncated_chars=len(candidate),
                        retries=attempt,
                        model_limit=model_limit,
                        input_tokens=input_tokens,
                    )
                    raise EmbeddingInputTooLongError(message) from exc
                candidate = next_candidate

        try:
            embedding = super()._get_text_embedding(candidate)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            model_limit, input_tokens = self._extract_context_limit_details(str(exc))
            message = self._emit_skip_warning(
                original_chars=original_chars,
                truncated_chars=len(candidate),
                retries=n_retries,
                model_limit=model_limit,
                input_tokens=input_tokens,
            )
            raise EmbeddingInputTooLongError(message) from exc

        if candidate != text:
            model_limit, input_tokens = self._extract_context_limit_details(
                str(last_exc) if last_exc is not None else ""
            )
            self._emit_truncation_warning(
                original_chars=original_chars,
                truncated_chars=len(candidate),
                model_limit=model_limit,
                input_tokens=input_tokens,
            )
        return embedding

    async def _aembed_text_with_truncation(
        self, text: str, n_retries: int = 24
    ) -> list[float]:
        """Async variant of ``_embed_text_with_truncation``.

        Args:
            text (str): Input text to embed.
            n_retries (int): Maximum number of truncation attempts before giving up.

        Returns:
            list[float]: The embedding vector.
        """
        original_chars = len(text)
        candidate = text
        last_exc: Exception | None = None

        for attempt in range(1, n_retries + 1):
            try:
                embedding = await super()._aget_text_embedding(candidate)
                if candidate != text:
                    model_limit, input_tokens = self._extract_context_limit_details(
                        str(last_exc) if last_exc is not None else ""
                    )
                    self._emit_truncation_warning(
                        original_chars=original_chars,
                        truncated_chars=len(candidate),
                        model_limit=model_limit,
                        input_tokens=input_tokens,
                    )
                return embedding
            except Exception as exc:
                if not self._is_context_limit_error(exc):
                    raise
                last_exc = exc
                next_candidate = self._truncate_text(candidate, exc)
                if not next_candidate or next_candidate == candidate:
                    model_limit, input_tokens = self._extract_context_limit_details(
                        str(exc)
                    )
                    message = self._emit_skip_warning(
                        original_chars=original_chars,
                        truncated_chars=len(candidate),
                        retries=attempt,
                        model_limit=model_limit,
                        input_tokens=input_tokens,
                    )
                    raise EmbeddingInputTooLongError(message) from exc
                candidate = next_candidate

        try:
            embedding = await super()._aget_text_embedding(candidate)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            model_limit, input_tokens = self._extract_context_limit_details(str(exc))
            message = self._emit_skip_warning(
                original_chars=original_chars,
                truncated_chars=len(candidate),
                retries=n_retries,
                model_limit=model_limit,
                input_tokens=input_tokens,
            )
            raise EmbeddingInputTooLongError(message) from exc

        if candidate != text:
            model_limit, input_tokens = self._extract_context_limit_details(
                str(last_exc) if last_exc is not None else ""
            )
            self._emit_truncation_warning(
                original_chars=original_chars,
                truncated_chars=len(candidate),
                model_limit=model_limit,
                input_tokens=input_tokens,
            )
        return embedding

    def get_text_embeddings_with_skips(
        self, texts: list[str]
    ) -> list[list[float] | None]:
        """Embed a batch of texts while allowing irreducible inputs to skip.

        Args:
            texts (list[str]): Text batch to embed.

        Returns:
            list[list[float] | None]: Embeddings aligned to the input order.
                ``None`` marks a skipped text that still exceeded the context
                window after truncation retries.
        """
        try:
            return cast(list[list[float] | None], super()._get_text_embeddings(texts))
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise

        embeddings: list[list[float] | None] = []
        for text in texts:
            try:
                embeddings.append(self._embed_text_with_truncation(text))
            except EmbeddingInputTooLongError:
                embeddings.append(None)
        return embeddings

    async def aget_text_embeddings_with_skips(
        self, texts: list[str]
    ) -> list[list[float] | None]:
        """Async batch embedding that allows irreducible inputs to skip.

        Args:
            texts (list[str]): Text batch to embed.

        Returns:
            list[list[float] | None]: Embeddings aligned to the input order.
                ``None`` marks a skipped text that still exceeded the context
                window after truncation retries.
        """
        try:
            return cast(
                list[list[float] | None],
                await super()._aget_text_embeddings(texts),
            )
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise

        embeddings: list[list[float] | None] = []
        for text in texts:
            try:
                embeddings.append(await self._aembed_text_with_truncation(text))
            except EmbeddingInputTooLongError:
                embeddings.append(None)
        return embeddings

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, falling back to per-item truncation on overflow.

        Args:
            texts (list[str]): Text batch.

        Returns:
            list[list[float]]: Embedding vectors in input order.
        """
        try:
            return super()._get_text_embeddings(texts)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            return [self._embed_text_with_truncation(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async batch embedding with truncation fallback.

        Args:
            texts (list[str]): Text batch.

        Returns:
            list[list[float]]: Embedding vectors in input order.
        """
        try:
            return await super()._aget_text_embeddings(texts)
        except Exception as exc:
            if not self._is_context_limit_error(exc):
                raise
            return [await self._aembed_text_with_truncation(text) for text in texts]


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
            response = self.vision_client.chat.completions.create(
                model=self.vision_model_id,
                messages=messages,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Error during vision inference: {}", e)
            raise RuntimeError(f"Vision inference failed: {e}")
