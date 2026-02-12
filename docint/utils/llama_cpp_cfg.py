"""
Llama.cpp pipeline for text generation and vision model inference.

This module provides a pipeline for running inference with Llama.cpp models,
supporting both CPU and GPU (CUDA/Metal) acceleration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger
from PIL import Image

from docint.utils.env_cfg import load_model_env, load_path_env, resolve_hf_cache_path


def messages_to_prompt_qwen3(messages: Sequence) -> str:
    """
    Convert chat messages to the Qwen3 ChatML prompt format with thinking
    disabled.

    Produces the ``<|im_start|>role\\ncontent<|im_end|>`` structure that Qwen3
    expects and appends an empty ``<think>`` block so the model skips its
    internal chain-of-thought and responds directly.

    Args:
        messages: Sequence of ChatMessage objects.

    Returns:
        The formatted prompt string.
    """
    prompt = ""
    for message in messages:
        role = (
            message.role.value if hasattr(message.role, "value") else str(message.role)
        )
        content = message.content or ""
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    # Start the assistant turn with thinking explicitly disabled
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prompt


def completion_to_prompt_qwen3(completion: str) -> str:
    """
    Wrap a plain completion string in the Qwen3 ChatML format with thinking
    disabled.

    Args:
        completion: The raw completion/instruction text.

    Returns:
        The formatted prompt string.
    """
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{completion}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _load_tokenizer(
    tokenizer_id: str | None = None,
    model_id: str | None = None,
    cache_dir: Path | None = None,
):
    """
    Attempt to load a HuggingFace tokenizer, trying multiple sources.

    Resolution order:
        1. ``tokenizer_id`` (explicit tokenizer repo, e.g. ``Qwen/Qwen3-1.7B``)
        2. ``model_id`` (the GGUF repo itself — works when it ships tokenizer files)

    For each candidate the function first checks for a local HF cache snapshot
    and falls back to loading by name (which may trigger a download when
    ``DOCINT_OFFLINE`` is not set).

    Args:
        tokenizer_id: Explicit HuggingFace repo that contains tokenizer files.
        model_id: The model repo to try as a fallback (often the GGUF repo).
        cache_dir: HF hub cache directory for local resolution.

    Returns:
        A ``PreTrainedTokenizerBase`` instance, or ``None`` if loading failed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.debug("transformers not installed; tokenizer loading unavailable.")
        return None

    candidates: list[str] = []
    if tokenizer_id:
        candidates.append(tokenizer_id)
    if model_id and model_id not in candidates:
        candidates.append(model_id)

    for candidate in candidates:
        # Try local HF cache first
        if cache_dir:
            resolved = resolve_hf_cache_path(cache_dir, candidate)
            if resolved:
                try:
                    tok = AutoTokenizer.from_pretrained(
                        str(resolved), trust_remote_code=True
                    )
                    logger.info(
                        "Loaded tokenizer from local cache: {} ({})",
                        candidate,
                        resolved,
                    )
                    return tok
                except Exception as exc:
                    logger.debug(
                        "Failed to load tokenizer from cache path {}: {}",
                        resolved,
                        exc,
                    )

        # Try loading by name (may download if online)
        try:
            tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            logger.info("Loaded tokenizer from repo: {}", candidate)
            return tok
        except Exception as exc:
            logger.debug("Failed to load tokenizer {}: {}", candidate, exc)

    return None


def build_prompt_functions(
    tokenizer_id: str | None = None,
    model_id: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[Callable[[Sequence], str], Callable[[str], str]]:
    """
    Build ``messages_to_prompt`` and ``completion_to_prompt`` callables for
    the llama-index ``LlamaCPP`` wrapper.

    When a HuggingFace tokenizer can be loaded (via ``tokenizer_id`` or
    ``model_id``), the returned functions use
    ``tokenizer.apply_chat_template()`` which is model-agnostic — it reads
    the Jinja2 chat template shipped with the tokenizer config.

    If no tokenizer is available the functions fall back to the hardcoded
    Qwen3 ChatML helpers (``messages_to_prompt_qwen3`` /
    ``completion_to_prompt_qwen3``).

    Args:
        tokenizer_id: Explicit HuggingFace repo for the tokenizer (e.g.
            ``Qwen/Qwen3-4B-Instruct-2507``).  Set via the ``LLM_TOKENIZER``
            env var.
        model_id: The LLM repo id.  Used as a fallback tokenizer source.
        cache_dir: HF hub cache directory for local snapshot resolution.

    Returns:
        A ``(messages_to_prompt, completion_to_prompt)`` tuple.
    """
    tokenizer = _load_tokenizer(tokenizer_id, model_id, cache_dir)

    if tokenizer is None:
        logger.warning(
            "No tokenizer found (tokenizer_id={}, model_id={}). "
            "Falling back to hardcoded Qwen3 prompt format.",
            tokenizer_id,
            model_id,
        )
        return messages_to_prompt_qwen3, completion_to_prompt_qwen3

    logger.info("Using tokenizer-based prompt formatting.")

    def _messages_to_prompt(messages: Sequence) -> str:
        chat_messages = []
        for message in messages:
            role = (
                message.role.value
                if hasattr(message.role, "value")
                else str(message.role)
            )
            content = message.content or ""
            chat_messages.append({"role": role, "content": content})

        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _completion_to_prompt(completion: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": completion},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return _messages_to_prompt, _completion_to_prompt


@dataclass(slots=True)
class LlamaCppPipeline:
    """
    Pipeline for text generation and image processing with Llama.cpp.

    This pipeline manages model loading, caching, and inference for both
    text and vision models using the llama-cpp-python library.
    """

    prompt_dir: Path | None = field(default=None, init=False)
    model_id: str | None = field(default=None, init=False)
    model_file: str | None = field(default=None, init=False)
    model_cache_dir: Path | None = field(default=None, init=False)
    n_gpu_layers: int = field(default=-1, init=False)  # -1 = offload all to GPU
    n_ctx: int = field(default=32768, init=False)
    _sys_prompt: str | None = field(default=None, init=False, repr=False)
    _model: Llama | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to load configurations.
        """
        self.model_id = load_model_env().vlm
        self.model_file = load_model_env().vlm_file
        self.prompt_dir = load_path_env().prompts
        self.model_cache_dir = load_path_env().llama_cpp_cache

        # Create cache directory if it doesn't exist
        if self.model_cache_dir:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sys_prompt(self) -> str:
        """
        Get the system prompt for model inference.

        Returns:
            str: The system prompt.
        """
        if self._sys_prompt is None:
            self._sys_prompt = self.load_prompt()
            logger.info("System prompt loaded.")
        return self._sys_prompt

    def load_prompt(self, kw: str = "system") -> str:
        """
        Load a prompt from the prompts directory based on the given keyword.

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

    def _load_model(self, model_path: str) -> Llama:
        """
        Load a GGUF model with Llama.cpp.

        Args:
            model_path (str): Path to the GGUF model file.

        Returns:
            Llama: The loaded model instance.
        """
        logger.info("Loading Llama.cpp model from: {}", model_path)

        model = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

        logger.info("Successfully loaded model with {} GPU layers", self.n_gpu_layers)
        return model

    @property
    def model(self) -> Llama:
        """
        Lazily load and cache the model.

        Returns:
            Llama: The loaded model instance.
        """
        if self._model is None:
            model_path = self._resolve_model_path(self.model_id)
            self._model = self._load_model(model_path)
        return self._model

    def _resolve_model_path(self, model_id: str | None) -> str:
        """
        Resolve the model identifier to a local file path.

        Args:
            model_id (str | None): Model identifier (HF repo ID or local path).

        Returns:
            str: Path to the GGUF model file.

        Raises:
            ValueError: If model_id is None or invalid.
        """
        if model_id is None:
            raise ValueError("model_id cannot be None")

        # If model_file is a direct local path, return it
        if self.model_file:
            file_path = Path(self.model_file)
            if file_path.exists() and file_path.is_file():
                logger.info("Using local model at: {}", file_path)
                return str(file_path)

        # Check cache directory for the file
        if self.model_cache_dir and self.model_file:
            # Direct path in cache dir
            direct_path = self.model_cache_dir / self.model_file
            if direct_path.exists():
                logger.info("Using cached model at: {}", direct_path)
                return str(direct_path)

            # HF cache structure: models--{org}--{repo}/snapshots/{hash}/{file}
            resolved = resolve_hf_cache_path(
                self.model_cache_dir, model_id, self.model_file
            )
            if resolved:
                logger.info("Using HF cached model at: {}", resolved)
                return str(resolved)

        logger.error("Model not found: {} (file: {})", model_id, self.model_file)
        raise FileNotFoundError(
            f"Model not found: {self.model_file} (repo: {model_id})"
        )

    def call_llama_cpp(
        self,
        prompt: str,
        img: str | bytes | Image.Image | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_k: int = 40,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
    ) -> str:
        """
        Call Llama.cpp for text generation.

        Args:
            prompt (str): The prompt to send to the model.
            img (str | bytes | Image.Image | None, optional): Optional image data (for vision models).
            temperature (float): The temperature for sampling. Defaults to 0.1.
            max_tokens (int): Maximum tokens to generate. Defaults to 2048.
            top_k (int): Top-k sampling parameter. Defaults to 40.
            top_p (float): Top-p sampling parameter. Defaults to 0.95.
            repeat_penalty (float): Repetition penalty. Defaults to 1.1.

        Returns:
            str: The generated text response.

        Raises:
            RuntimeError: If model loading or inference fails.
        """
        try:
            # Format the full prompt with system message
            full_prompt = f"{self.sys_prompt}\n\nUser: {prompt}\n\nAssistant:"

            # Generate response
            response = self.model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                echo=False,
            )

            # Extract text from response
            if isinstance(response, dict) and "choices" in response:
                text = response["choices"][0]["text"].strip()
                return text

            logger.error("Unexpected response format from Llama.cpp")
            return ""

        except Exception as e:
            logger.error("Error during Llama.cpp inference: {}", e)
            raise RuntimeError(f"Llama.cpp inference failed: {e}")

    @staticmethod
    def ensure_model(model_id: str, repo_id: str | None = None) -> None:
        """
        Ensure that the specified GGUF model is available locally.
        If not, attempt to download it from Hugging Face.

        Args:
            model_id (str): The name of the model file (e.g., "model.gguf").
            repo_id (str | None): The Hugging Face repository ID. If None, assumes local file.
        """
        try:
            paths = load_path_env()
            cache_dir = paths.llama_cpp_cache

            # Create cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_path = cache_dir / model_id

            # Check if model already exists (direct path)
            if model_path.exists():
                logger.info(
                    "Model '{}' is already available at {}", model_id, model_path
                )
                return

            # Check HF cache structure
            if repo_id:
                resolved = resolve_hf_cache_path(cache_dir, repo_id, model_id)
                if resolved:
                    logger.info(
                        "Model '{}' is already available at {}", model_id, resolved
                    )
                    return

            if repo_id is None:
                logger.warning(
                    "Model '{}' not found locally and no repo_id provided for download",
                    model_id,
                )
                return

            logger.info(
                "Model '{}' not found. Downloading from {}...", model_id, repo_id
            )

            # Download from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_id,
                cache_dir=cache_dir,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )

            logger.info(
                "Successfully downloaded model '{}' to {}", model_id, downloaded_path
            )

        except Exception as e:
            logger.error("Failed to ensure model '{}': {}", model_id, e)
            raise
