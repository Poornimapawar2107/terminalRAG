"""Unified Rich Terminal CLI for Vector RAG."""

import argparse
import sys
from pathlib import Path
from typing import Optional

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from vector_rag import __version__
from vector_rag.config.settings import Settings
from vector_rag.generation.rag_service import RAGService
from vector_rag.ingestion.service import DocumentService
from vector_rag.retrieval.service import RetrievalService
from vector_rag.utils.errors import VectorRAGError
from vector_rag.utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger("cli.main")


def print_error_panel(error: Exception):
    """Format and print domain exceptions in a user-friendly Rich box."""
    if isinstance(error, VectorRAGError):
        title = f"[bold red]✗ {error.__class__.__name__}[/bold red]"
        body = f"[bold white]{error.message}[/bold white]"
        if error.hint:
            body += f"\n\n[dim cyan]Hint: {error.hint}[/dim cyan]"
    else:
        title = "[bold red]✗ Unexpected Error[/bold red]"
        body = f"[white]{str(error)}[/white]"

    console.print(Panel(body, title=title, border_style="red", expand=False))


def cmd_ingest(args):
    """Handle document ingestion command."""
    settings = Settings.load_from_yaml(args.config)
    setup_logging(level=settings.logging.level, log_file=settings.logging.file)
    service = DocumentService(settings=settings)

    target_path = Path(args.path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Ingesting '{target_path.name}'...", total=None)
        result = service.ingest_path(target_path, force=args.force)
        progress.update(task, completed=True)

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="bold cyan")
    summary_table.add_column(justify="right")

    summary_table.add_row("Files Discovered:", str(result.discovered))
    summary_table.add_row("Successfully Parsed:", f"[green]{result.parsed}[/green]")
    summary_table.add_row("Skipped (Identical Hash):", f"[yellow]{result.skipped}[/yellow]")
    summary_table.add_row("Chunks Created:", str(result.chunks_created))
    summary_table.add_row("Embeddings Generated:", str(result.embeddings_generated))
    summary_table.add_row("Chroma Vectors Indexed:", f"[bold green]{result.vectors_indexed}[/bold green]")
    summary_table.add_row("Total Collection Vectors:", str(service.vector_store.count()))

    console.print(
        Panel(
            summary_table,
            title="[bold green]✓ Ingestion Complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )


def cmd_query(args):
    """Handle natural language RAG query command."""
    settings = Settings.load_from_yaml(args.config)
    setup_logging(level=settings.logging.level, log_file=settings.logging.file)

    if args.vector_only:
        retrieval_service = RetrievalService(settings=settings)
        console.print(f"\n[bold cyan]Query:[/bold cyan] {args.query}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Querying vector index...", total=None)
            candidates = retrieval_service.retrieve(query=args.query, top_k=args.top_k)
            progress.update(task, completed=True)

        if not candidates:
            console.print("[yellow]No relevant chunks found.[/yellow]")
            return

        table = Table(title=f"Vector Candidates (Top-{len(candidates)})", show_header=True)
        table.add_column("Rank", justify="center", width=5)
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("Source", style="yellow", width=18)
        table.add_column("Page", justify="center", width=6)
        table.add_column("Snippet Preview", overflow="fold")

        for r, c in enumerate(candidates, start=1):
            table.add_row(
                str(r),
                f"{c.score:.4f}",
                c.filename,
                str(c.page) if c.page is not None else "-",
                c.text.replace("\n", " ")[:120] + "...",
            )
        console.print(table)
        return

    # Full RAG Pipeline
    rag_service = RAGService(settings=settings)
    console.print(f"\n[bold cyan]Query:[/bold cyan] {args.query}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Retrieving, reranking, and generating response...", total=None)
        response = rag_service.query(
            query_text=args.query,
            top_k=args.top_k,
            top_n=args.top_n,
        )
        progress.update(task, completed=True)

    # 1. Answer Panel
    console.print(
        Panel(
            response.answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # 2. Citations Table
    if response.citations:
        cite_table = Table(title="Sources & Citations", show_header=True, header_style="bold cyan")
        cite_table.add_column("Source", justify="center", style="bold cyan", width=8)
        cite_table.add_column("Document", style="yellow", width=22)
        cite_table.add_column("Page", justify="center", width=6)
        cite_table.add_column("Snippet", overflow="fold")

        for c in response.citations:
            cite_table.add_row(
                f"[{c.source_id}]",
                c.filename,
                str(c.page) if c.page is not None else "-",
                c.snippet or "-",
            )
        console.print(cite_table)
    else:
        console.print("[dim]No specific citations were referenced in the answer.[/dim]")


def cmd_list(args):
    """List all registered documents in SQLite."""
    settings = Settings.load_from_yaml(args.config)
    setup_logging(level=settings.logging.level, log_file=settings.logging.file)
    service = DocumentService(settings=settings)

    docs = service.list_documents()
    if not docs:
        console.print("[yellow]No documents registered in the system.[/yellow]")
        return

    table = Table(title=f"Registered Documents ({len(docs)} total)", show_header=True, header_style="bold blue")
    table.add_column("Doc ID", style="dim", max_width=12, overflow="ellipsis")
    table.add_column("Filename", style="bold yellow")
    table.add_column("Type", style="cyan", width=6)
    table.add_column("Size", justify="right", width=10)
    table.add_column("Pages", justify="right", width=6)
    table.add_column("SHA256 Hash", style="dim", max_width=16, overflow="ellipsis")
    table.add_column("Indexed At", style="dim")

    for d in docs:
        table.add_row(
            d.document_id,
            d.filename,
            d.file_type,
            f"{d.file_size:,} B",
            str(d.page_count),
            d.content_hash,
            d.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


def cmd_delete(args):
    """Delete a document by ID or filename."""
    settings = Settings.load_from_yaml(args.config)
    setup_logging(level=settings.logging.level, log_file=settings.logging.file)
    service = DocumentService(settings=settings)

    identifier = args.identifier.strip()
    docs = service.list_documents()

    target_doc = None
    for d in docs:
        if d.document_id == identifier or d.filename == identifier:
            target_doc = d
            break

    if not target_doc:
        console.print(f"[bold red]✗ Document '{identifier}' not found in registry.[/bold red]")
        return

    service.delete_document(target_doc.document_id)
    console.print(
        Panel(
            f"Successfully deleted [bold yellow]{target_doc.filename}[/bold yellow] and all associated chunks/vectors.",
            title="[bold green]✓ Document Deleted[/bold green]",
            border_style="green",
            expand=False,
        )
    )


def cmd_status(args):
    """Display overall system status and component configurations."""
    settings = Settings.load_from_yaml(args.config)
    service = DocumentService(settings=settings)

    docs = service.list_documents()
    vector_count = service.vector_store.count()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Vector RAG Version:", __version__)
    table.add_row("SQLite Registry:", f"{len(docs)} document(s) registered")
    table.add_row("ChromaDB Vectors:", f"{vector_count} vector chunk(s) indexed")
    table.add_row("Embedding Model:", settings.embedding.model)
    table.add_row("Reranker Model:", settings.reranker.model)
    table.add_row("Generation LLM:", settings.generation.model)
    table.add_row("Chunking Strategy:", f"{settings.chunking.strategy} (size={settings.chunking.chunk_size}, overlap={settings.chunking.chunk_overlap})")

    console.print(
        Panel(
            table,
            title="[bold blue]System Status[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )


def main():
    """Main CLI entrypoint parser."""
    parser = argparse.ArgumentParser(
        prog="vector-rag",
        description="Terminal Vector RAG: Modular, Production-Grade Question Answering CLI.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a file or directory")
    p_ingest.add_argument("path", help="File or directory path to ingest")
    p_ingest.add_argument("--force", action="store_true", help="Force re-ingestion even if unchanged")

    # Query
    p_query = subparsers.add_parser("query", help="Query the RAG system")
    p_query.add_argument("query", help="Question or query string")
    p_query.add_argument("--top-k", type=int, default=10, help="Candidate vector retrieval count")
    p_query.add_argument("--top-n", type=int, default=5, help="Reranked chunk count")
    p_query.add_argument("--vector-only", action="store_true", help="Run only vector search without LLM")

    # List
    subparsers.add_parser("list", help="List all registered documents")

    # Delete
    p_del = subparsers.add_parser("delete", help="Delete a document by ID or filename")
    p_del.add_argument("identifier", help="Document ID or filename to delete")

    # Status
    subparsers.add_parser("status", help="Show system status and component info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "ingest":
            cmd_ingest(args)
        elif args.command == "query":
            cmd_query(args)
        elif args.command == "list":
            cmd_list(args)
        elif args.command == "delete":
            cmd_delete(args)
        elif args.command == "status":
            cmd_status(args)
    except Exception as e:
        print_error_panel(e)
        logger.exception("CLI command execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
