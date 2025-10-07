from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from llama_index.core.schema import Document, MediaResource
from pdf2image import convert_from_path
from PIL import Image

from docint.modules.ollama_cfg import OllamaPipeline


@dataclass
class DocScanner(OllamaPipeline):
    """
    A class to handle document scanning and OCR processing.

    Args:
        OllamaPipeline (OllamaPipeline): The base class for handling Ollama interactions.
    """

    file_path: Path | None = field(default=None)
    source: str = field(default="unknown")
    images: list[Image.Image] = field(default_factory=list, init=False)
    buffered_images: list[bytes] = field(default_factory=list, init=False)
    documents: list[Document] = field(default_factory=list, init=False)
    text: str = field(default="", init=False)

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass

    def pdf_to_img(self) -> list[Image.Image]:
        """
        Convert the loaded PDF file into a list of images.

        Returns:
            list[Image.Image]: A list of images converted from the PDF file.

        Raises:
            ValueError: If no valid PDF file is loaded.
        """
        if self.file_path is None or not self.file_path.exists():
            raise ValueError(
                "No valid PDF file loaded. Please call file_to_doc() first."
            )
        self.images = convert_from_path(self.file_path, dpi=300, fmt="png")
        return self.images

    def img_to_buffer(self) -> list[bytes]:
        """
        Convert a list of images into a list of byte buffers.

        Returns:
            list[bytes]: A list of byte buffers representing the images.
        """
        for image in self.images:
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                self.buffered_images.append(buffer.getvalue())
        return self.buffered_images

    def buffer_to_doc(
        self,
        ocr_kw: str = "ocr",
        max_workers: int = 4,
    ) -> list[Document]:
        """
        Perform OCR inference on the provided image data using a ThreadPoolExecutor for concurrency.

        Args:
            ocr_kw (str, optional): The keyword to identify the OCR prompt file. Defaults to "ocr".
            max_workers (int, optional): Maximum number of worker threads to use for concurrent processing. Defaults to 4.

        Returns:
            list[Document]: A list of Document objects containing the OCR results for each page.

        Raises:
            ValueError: If OCR inference fails for any reason.
        """
        prompt = self.load_prompt(kw=ocr_kw)
        if isinstance(self.buffered_images, bytes):
            self.buffered_images = [self.buffered_images]

        def process_page(page_num: int, image_bytes: bytes) -> Document:
            self.text = self.call_ollama_server(prompt=prompt, img=image_bytes)
            return Document(
                text_resource=MediaResource(text=self.text, mimetype="text/plain"),
                metadata={
                    "source": self.source,
                    "page": page_num,
                    "file": Path(self.file_path).name if self.file_path else "",
                },
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_page, i, image_bytes): i
                for i, image_bytes in enumerate(self.buffered_images, start=1)
                if isinstance(image_bytes, bytes)
            }
            for future in as_completed(futures):
                try:
                    self.documents.append(future.result())
                except Exception as e:
                    raise ValueError(f"OCR failed on page {futures[future]}: {e}")

        # Keep results sorted by page order
        self.documents.sort(key=lambda doc: doc.metadata.get("page", 0))
        return self.documents

    def ocr_pdf_to_doc(self) -> list[Document]:
        """
        Process a PDF file by converting it to images, performing OCR inference, and returning the results as Document objects.

        Returns:
            list[Document]: A list of Document objects containing the OCR results.
        """
        self.pdf_to_img()
        self.img_to_buffer()
        return self.buffer_to_doc()
