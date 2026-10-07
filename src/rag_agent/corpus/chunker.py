"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        elif suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        else:
         raise ValueError(f"Unsupported file type: {suffix}")

        metadata = self._infer_metadata(
            file_path,
            metadata_overrides,
        )

        chunks: list[DocumentChunk] = []

        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk["text"]

            chunk_id = VectorStoreManager.generate_chunk_id(
                metadata.source,
                chunk_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        all_chunks: list[DocumentChunk] = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path=file_path,
                    metadata_overrides=metadata_overrides,
                )
                all_chunks.extend(chunks)

            except (FileNotFoundError, ValueError) as exc:
                logger.warning(f"Skipping {file_path}: {exc}")

        return all_chunks
    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=chunk_size,
             chunk_overlap=chunk_overlap,
        )

        split_documents = text_splitter.split_documents(documents)

        chunks = []

        for doc in split_documents:
            text = doc.page_content.strip()

            if not text:
                continue

            chunks.append(
                {
                    "text": text,
                    "page": doc.metadata.get("page", 0),
                }
            )

        return chunks

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        text = file_path.read_text(encoding="utf-8")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        sections = markdown_splitter.split_text(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = []

        for section in sections:
            if len(section.page_content) > chunk_size:
                split_sections = text_splitter.split_text(section.page_content)

                for split_text in split_sections:
                    chunks.append(
                        {
                            "text": split_text,
                            "header": " > ".join(section.metadata.values()),
                        }
                    )
            else:
                chunks.append(
                    {
                        "text": section.page_content,
                        "header": " > ".join(section.metadata.values()),
                    }
                )

        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
       self,
       file_path: Path,
       overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.
        """
        stem = file_path.stem
        parts = stem.rsplit("_", 1)

        if len(parts) == 2:
           topic, difficulty = parts
        else:
           topic = stem
           difficulty = "intermediate"
           
        metadata = {
           "topic": topic,
           "difficulty": difficulty,
           "type": "concept",
           "source": file_path.name,
           "related_topics": [],
           "is_bonus": topic.lower() in {
                "som",
                "boltzmannmachine",
                "gan",
            },
        }

        if overrides:
            metadata.update(overrides)
            
        return ChunkMetadata(**metadata)