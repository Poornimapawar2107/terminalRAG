"""Complete master RAG service orchestrating retrieval, reranking, context formatting, and LLM generation."""

from typing import Optional

from vector_rag.config.settings import Settings
from vector_rag.generation.citation import CitationExtractor
from vector_rag.generation.context import ContextBuilder
from vector_rag.generation.llm import BaseLLM, create_llm
from vector_rag.models.response import RAGResponse
from vector_rag.retrieval.service import RetrievalService
from vector_rag.utils.logging import current_request_id, get_logger, set_request_id

logger = get_logger("services.rag")


class RAGService:
    """End-to-End Vector RAG pipeline orchestrator."""

    DEFAULT_PROMPT_TEMPLATE = (
        "Reference Sources:\n"
        "{context}\n\n"
        "User Question: {query}\n\n"
        "Instructions:\n"
        "1. Answer the question directly using facts from the reference sources above.\n"
        "2. You MUST cite the source number in brackets like [1] or [2] whenever mentioning details from [Source 1], [Source 2], etc.\n\n"
        "Answer with citations:"
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_service: Optional[RetrievalService] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm: Optional[BaseLLM] = None,
        citation_extractor: Optional[CitationExtractor] = None,
    ) -> None:
        self.settings = settings or Settings.load_from_yaml()
        self.retrieval_service = retrieval_service or RetrievalService(settings=self.settings)
        self.context_builder = context_builder or ContextBuilder()
        self.llm = llm or create_llm(
            model_name=self.settings.generation.model,
            temperature=self.settings.generation.temperature,
            max_tokens=self.settings.generation.max_tokens,
        )
        self.citation_extractor = citation_extractor or CitationExtractor()

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute full RAG query:
        1. Retrieve top_k candidates from ChromaDB
        2. Rerank top_n chunks via Cross-Encoder
        3. Build ContextPackage with Source IDs
        4. Call LLM for generation
        5. Extract & correlate citations
        """
        req_id = set_request_id()
        k = top_k or self.settings.retrieval.top_k
        n = top_n or self.settings.reranker.top_n

        logger.info("Executing RAG query: '%s' (top_k=%d, top_n=%d)", query_text, k, n)

        # 1 & 2: Two-stage retrieval
        search_result = self.retrieval_service.search_and_rerank(
            query=query_text,
            top_k=k,
            top_n=n,
        )

        candidates = search_result.candidates
        reranked = search_result.reranked

        if not reranked and not candidates:
            return RAGResponse(
                query=query_text,
                answer="No relevant documents were found in the knowledge base.",
                citations=[],
                retrieved_chunks=[],
                reranked_chunks=[],
                request_id=req_id,
            )

        # Use reranked chunks if available, fallback to initial candidates
        active_chunks = reranked if reranked else candidates

        # 3. Build structured context
        context_pkg = self.context_builder.build_context(
            query=query_text,
            chunks=active_chunks,
        )
        formatted_context = context_pkg.format_prompt_context()

        # 4. Generate with LLM
        prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
            context=formatted_context,
            query=query_text,
        )
        raw_answer = self.llm.generate(prompt)

        # 5. Extract citations
        citations = self.citation_extractor.resolve_citations(
            answer_text=raw_answer,
            context_package=context_pkg,
        )

        response = RAGResponse(
            query=query_text,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=candidates,
            reranked_chunks=reranked,
            request_id=req_id,
        )
        logger.info("RAG query complete with %d citations.", len(citations))
        return response
