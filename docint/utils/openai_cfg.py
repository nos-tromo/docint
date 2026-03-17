from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.base.llms.types import LLMMetadata
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


class LocalOpenAI(LlamaIndexOpenAI):
    """Subclass of LlamaIndex's OpenAI that handles local models with unknown names gracefully."""

    def __init__(self, context_window: int = 4096, **kwargs: Any) -> None:
        """Initialize the LocalOpenAI instance.

        Args:
            context_window (int, optional): The context window size for the model. Defaults to 4096.
        """
        super().__init__(**kwargs)
        self._context_window = context_window

    @property
    def metadata(self) -> LLMMetadata:
        """Override the metadata property to provide fallback values for unknown models.

        Returns:
            LLMMetadata: The metadata for the model, with fallback values if the model name is not recognized.
        """
        try:
            return super().metadata
        except ValueError:
            # Fallback for unknown models (e.g. local paths for llama-server)
            return LLMMetadata(
                context_window=self._context_window,
                num_output=self.max_tokens or 2048,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )


def get_openai_reasoning_effort(openai_config: OpenAIConfig) -> str | None:
    """Return the OpenAI reasoning effort to request for chat completions.

    Reasoning is only enabled for the native OpenAI provider to avoid sending
    unsupported parameters to other OpenAI-compatible backends such as Ollama
    or llama.cpp.

    Args:
        openai_config: Parsed OpenAI environment configuration.

    Returns:
        The configured reasoning effort string when enabled for the OpenAI
        provider, otherwise ``None``.
    """
    if not openai_config.thinking_enabled:
        return None
    if openai_config.model_provider != "openai":
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
        self.text_model_id = _model_config.text_model_file.removesuffix(".gguf")
        self.vision_model_id = _model_config.vision_model_file.removesuffix(".gguf")

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
