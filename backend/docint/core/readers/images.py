import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MediaResource
from PIL import Image

from docint.utils.mimetype import get_mimetype
from docint.utils.ollama_cfg import OllamaPipeline
from loguru import logger


@dataclass
class ImageReader(BaseReader):
    """
    Image reader that utilizes the OllamaPipeline for processing images.
    """

    ollama_pipeline: OllamaPipeline = field(default_factory=OllamaPipeline, init=False)

    def _load_image(self, file_path: Path, mode: str = "RGB") -> Image.Image:
        """
        Load an image from a file path.

        Args:
            file (str | Path): The path to the image file.
            mode (str, optional): The mode to convert the image to. Defaults to "RGB".

        Returns:
            Image.Image: The loaded image.
        """
        return Image.open(file_path).convert(mode=mode)

    def _encode_img_to_base64(self, img: Image.Image, format: str = "PNG") -> str:
        """
        Encode a PIL Image to a base64 string.

        Args:
            img (Image.Image): The image to encode.
            format (str, optional): The format to save the image in. Defaults to "PNG".

        Returns:
            str: The base64 encoded string of the image.
        """
        buffer = BytesIO()
        img.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def _enrich_document(
        self, file_path: Path, text: str, source: str = "image"
    ) -> Document:
        """
        Enrich a document with metadata from the image file.

        Args:
            file_path (Path): The path to the image file.
            text (str): The text content extracted from the image.
            source (str, optional): The source type. Defaults to "image".

        Returns:
            Document: The enriched document.

        Raises:
            ValueError: If file_path is not set.
        """
        if file_path is None:
            raise ValueError("file_path is not set.")
        filename = file_path.name
        mimetype = get_mimetype(file_path)
        return Document(
            text_resource=MediaResource(text=text, mimetype=mimetype),
            metadata={
                "file_path": str(file_path),
                "file_name": filename,
                "filename": filename,
                "file_type": mimetype,
                "mimetype": mimetype,
                "source": source,
                "origin": {
                    "filename": filename,
                    "mimetype": mimetype,
                },
            },
        )

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """
        Load and process image data using the OllamaPipeline.

        Args:
            file (str | bytes): The path to the image file or image bytes.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list containing a single Document object with the processed image data.
        """
        logger.info("[ImageReader] Loading image from {}", file)
        file_path = Path(file) if not isinstance(file, Path) else file
        img = self._load_image(file_path)
        img_base64 = self._encode_img_to_base64(img)
        prompt = self.ollama_pipeline.load_prompt("describe")
        self.response = self.ollama_pipeline.call_ollama_server(
            prompt=prompt,
            img=img_base64,
        )
        return [self._enrich_document(file_path, self.response)]
