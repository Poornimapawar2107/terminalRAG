"""CLI and standalone workflow runner for testing ingestion."""

import argparse
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vector_rag.config.settings import Settings
from vector_rag.ingestion.service import DocumentService
from vector_rag.utils.logging import setup_logging

console = Console()


def run_ingest(target_path: str, force: bool = False):
    """Run ingestion workflow and print rich tables of SQLite storage and ChromaDB."""
    settings = Settings.load_from_yaml()
    setup_logging(level=settings.logging.level, log_file=settings.logging.file)

    service = DocumentService(settings=settings)

    console.print(f"[bold cyan]Ingesting path:[/bold cyan] {target_path}")
    result = service.ingest_path(target_path, force=force)

    # Ingestion Summary Panel
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="bold cyan")
    summary_table.add_column(justify="right")

    summary_table.add_row("Files Discovered:", str(result.discovered))
    summary_table.add_row("Successfully Parsed:", f"[green]{result.parsed}[/green]")
    summary_table.add_row("Skipped (Identical Hash):", f"[yellow]{result.skipped}[/yellow]")
    summary_table.add_row("Chunks Created:", str(result.chunks_created))
    summary_table.add_row("Embeddings Generated:", str(result.embeddings_generated))
    summary_table.add_row("Chroma Vectors Stored:", f"[bold green]{result.vectors_indexed}[/bold green]")
    summary_table.add_row("Total Vectors in ChromaDB:", str(service.vector_store.count()))

    console.print(
        Panel(
            summary_table,
            title="[bold green]✓ Ingestion Summary[/bold green]",
            expand=False,
        )
    )

    # Document Registry Table (SQLite)
    docs = service.list_documents()
    if docs:
        doc_table = Table(title="SQLite Document Registry (`documents` table)", show_header=True)
        doc_table.add_column("Doc ID", style="dim", max_width=12, overflow="ellipsis")
        doc_table.add_column("Filename", style="bold")
        doc_table.add_column("Type", style="cyan")
        doc_table.add_column("Size (bytes)", justify="right")
        doc_table.add_column("Pages", justify="right")
        doc_table.add_column("SHA256 Hash", style="dim", max_width=16, overflow="ellipsis")

        for d in docs:
            doc_table.add_row(
                d.document_id,
                d.filename,
                d.file_type,
                str(d.file_size),
                str(d.page_count),
                d.content_hash,
            )
        console.print(doc_table)

        # Chunks Table (SQLite)
        for d in docs:
            chunks = service.get_document_chunks(d.document_id)
            chunk_table = Table(
                title=f"SQLite Chunks for '{d.filename}' (Total: {len(chunks)})",
                show_header=True,
            )
            chunk_table.add_column("Chunk Index", justify="right", style="cyan")
            chunk_table.add_column("Page", justify="right")
            chunk_table.add_column("Chars", justify="right")
            chunk_table.add_column("Chunk Text Preview", max_width=60, overflow="ellipsis")

            for c in chunks:
                chunk_table.add_row(
                    str(c.chunk_index),
                    str(c.page or 1),
                    str(c.char_count),
                    c.text.replace("\n", " ")[:80] + "..." if len(c.text) > 80 else c.text,
                )
            console.print(chunk_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Vector RAG ingestion workflow.")
    parser.add_argument("path", help="File or directory path to ingest")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion even if hash is identical")
    args = parser.parse_args()

    run_ingest(args.path, force=args.force)
