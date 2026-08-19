"""CLI query tool inspecting two-stage retrieval (Vector Search + Cross-Encoder Reranking)."""

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vector_rag.config.settings import Settings
from vector_rag.retrieval.service import RetrievalService
from vector_rag.utils.logging import setup_logging

console = Console()


def run_query(query: str, top_k: int = 10, top_n: int = 5, vector_only: bool = False):
    """Execute search query and display comparison of Vector Candidates vs Reranked Chunks."""
    settings = Settings.load_from_yaml()
    setup_logging(level=settings.logging.level, log_file=settings.logging.file)

    service = RetrievalService(settings=settings)

    console.print(f"\n[bold cyan]User Query:[/bold cyan] {query}")

    if vector_only:
        console.print(f"[dim]Running Pure Vector Search (Top-{top_k})...[/dim]\n")
        candidates = service.retrieve(query=query, top_k=top_k)

        if not candidates:
            console.print(Panel("[yellow]No matching vector chunks found.[/yellow]", title="Results"))
            return

        table = Table(title=f"Vector Search Candidates (Top-{len(candidates)})", show_header=True)
        table.add_column("Rank", justify="center", style="cyan", width=6)
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("Filename", style="bold yellow", width=18)
        table.add_column("Page", justify="center", style="dim", width=6)
        table.add_column("Preview", overflow="fold")

        for rank, c in enumerate(candidates, start=1):
            table.add_row(
                str(rank),
                f"{c.score:.4f}",
                c.filename or "-",
                str(c.page) if c.page is not None else "-",
                c.text.replace("\n", " ")[:120] + "...",
            )
        console.print(table)
        return

    # Two-Stage Search (Vector + Cross-Encoder)
    console.print(f"[dim]Running Two-Stage Search: Top-{top_k} Vector Candidates -> Top-{top_n} Cross-Encoder Reranked...[/dim]\n")
    search_res = service.search_and_rerank(query=query, top_k=top_k, top_n=top_n)

    if not search_res.reranked:
        console.print(Panel("[yellow]No matching chunks found.[/yellow]", title="Results"))
        return

    # 1. Vector Candidates Summary Table
    cand_table = Table(
        title=f"Stage 1: Vector Candidates (Top-{len(search_res.candidates)})",
        show_header=True,
        header_style="bold blue",
    )
    cand_table.add_column("Rank", justify="center", width=5)
    cand_table.add_column("Vector Score", justify="right", style="blue", width=12)
    cand_table.add_column("Filename", style="yellow", width=16)
    cand_table.add_column("Page", justify="center", width=5)
    cand_table.add_column("Preview", overflow="fold")

    for rank, c in enumerate(search_res.candidates, start=1):
        cand_table.add_row(
            str(rank),
            f"{c.score:.4f}",
            c.filename or "-",
            str(c.page) if c.page is not None else "-",
            c.text.replace("\n", " ")[:90] + "...",
        )
    console.print(cand_table)

    # 2. Reranked Results Table
    rerank_table = Table(
        title=f"Stage 2: Cross-Encoder Reranked Results (Top-{len(search_res.reranked)})",
        show_header=True,
        header_style="bold magenta",
    )
    rerank_table.add_column("Final Rank", justify="center", style="bold magenta", width=10)
    rerank_table.add_column("Rerank Score", justify="right", style="bold green", width=12)
    rerank_table.add_column("Vector Score", justify="right", style="dim blue", width=12)
    rerank_table.add_column("Filename", style="bold yellow", width=16)
    rerank_table.add_column("Page", justify="center", width=6)
    rerank_table.add_column("Refined Chunk Text", overflow="fold")

    for rank, r in enumerate(search_res.reranked, start=1):
        rerank_table.add_row(
            str(rank),
            f"{r.rerank_score:.4f}",
            f"{r.retrieval_score:.4f}",
            r.filename or "-",
            str(r.page) if r.page is not None else "-",
            r.text.replace("\n", " "),
        )
    console.print(rerank_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Vector RAG with two-stage Vector + Reranker search.")
    parser.add_argument("query", help="Query string to search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of vector candidates (Stage 1)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of reranked results (Stage 2)")
    parser.add_argument("--vector-only", action="store_true", help="Run only Stage 1 vector search")
    args = parser.parse_args()

    run_query(args.query, top_k=args.top_k, top_n=args.top_n, vector_only=args.vector_only)
