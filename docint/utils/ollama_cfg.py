from dataclasses import dataclass, field
from pathlib import Path

import ollama
import requests
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from docint.utils.env_cfg import load_host_env, load_model_env, load_path_env

load_dotenv()


@dataclass
class OllamaPipeline:
    """
    Pipeline for summarization and image processing with the Ollama API.
    """

    ollama_host: str | None = field(default=None, init=False)
    prompt_dir: Path | None = field(default=None, init=False)
    model_id: str | None = field(default=None, init=False)
    _sys_prompt: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.model_id = load_model_env().vision_model
        self.prompt_dir = load_path_env().prompts
        self.ollama_host = load_host_env().ollama_host

    @property
    def sys_prompt(self) -> str:
        """
        Get the system prompt for the Ollama API.

        Returns:
            str: The system prompt.
        """
        if self._sys_prompt is None:
            self._sys_prompt = self.load_prompt()
            logger.info("System prompt loaded.")
        return self._sys_prompt

    def _get_ollama_health(self) -> bool:
        """
        Perform a health check by querying Ollama's /api/tags endpoint.

        Returns:
            bool: True if the Ollama server responds with model tags, False otherwise.
        """
        response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
        return response.status_code == 200 and "models" in response.json()

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

    def call_ollama_server(
        self,
        prompt: str,
        img: str | bytes | Image.Image | None = None,
        think: bool = False,
        num_ctx: int = 32768,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Call the ollama server with the given model and prompt.

        Args:
            prompt (str): The prompt to send to the model.
            img (str | bytes | Image.Image | None, optional): Optional image data to include in the prompt. Defaults to None.
            think (bool): Whether to enable "think" mode for the model. Defaults to False.
            num_ctx (int): The number of context tokens to use. Defaults to 32768.
            temperature (float): The temperature for the model's response. Defaults to 0.1.
            seed (int): The random seed for the model's response. Defaults to 42.
            stop (list[str]): A list of stop sequences for the model's response. Defaults to None.
            num_predict (int | None): The number of tokens to predict. Defaults to None.
            top_k (int | None): The top_k parameter for the model's response. Defaults to None.
            top_p (float | None): The top_p parameter for the model's response. Defaults to None.

        Returns:
            str: The response from the ollama server, or an empty string if an error occurs.

        Raises:
            RuntimeError: If the Ollama model cannot be loaded or the server is unreachable.
        """
        if not self._get_ollama_health():
            logger.error(
                "RuntimeError: Ollama server does not respond. Please ensure it is running and accessible."
            )
            raise RuntimeError(
                "Ollama server is not reachable. Please check your configuration."
            )

        # Convert PIL image to bytes if provided
        if isinstance(img, Image.Image):
            from io import BytesIO

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img = buffer.getvalue()

        # Build messages (add image only if present)
        system = {"role": "system", "content": self.sys_prompt}
        user = {"role": "user", "content": prompt, "images": []}
        if img:
            user["images"] = [img.decode("utf-8") if isinstance(img, bytes) else img]

        # Ensure model id is set for ollama library
        if not self.model_id:
            logger.error("RuntimeError: Model ID is not set.")
            raise RuntimeError("Model ID must be a valid string.")

        client = ollama.Client(host=self.ollama_host)
        response = client.chat(
            model=self.model_id,
            think=think,
            messages=[system, user],
            options={
                **(
                    {"num_ctx": num_ctx}
                    if isinstance(num_ctx, int) and num_ctx > 0
                    else {}
                ),
                "temperature": temperature,
                "seed": seed,
                "stop": stop,
                "num_predict": num_predict,
                "top_k": top_k,
                "top_p": top_p,
            },
        )
        return response["message"]["content"].strip()

    @staticmethod
    def ensure_model(model_name: str) -> None:
        """
        Ensure that the specified model is available on the Ollama server.
        If not, it attempts to pull the model.

        Args:
            model_name (str): The name of the model to check/pull.
        """
        try:
            ollama_host = load_host_env().ollama_host
            client = ollama.Client(host=ollama_host)
            models_response = client.list()
            existing_models = [m["model"] for m in models_response.get("models", [])]

            # Check if model exists
            if (
                model_name in existing_models
                or f"{model_name}:latest" in existing_models
            ):
                logger.info("Model '{}' is already available.", model_name)
                return

            logger.info("Model '{}' not found. Pulling...", model_name)

            # Stream the pull progress
            for progress in client.pull(model_name, stream=True):
                if "completed" in progress and "total" in progress:
                    percent = (progress["completed"] / progress["total"]) * 100
                    if int(percent) % 10 == 0:  # Log every 10%
                        logger.debug("Pulling {}: {:.1f}%", model_name, percent)

            logger.info("Successfully pulled model '{}'.", model_name)

        except Exception as e:
            logger.error("Failed to ensure model '{}': {}", model_name, e)
