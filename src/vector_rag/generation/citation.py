"""Citation extraction and mapping utilities."""

import re
from typing import Dict, List, Set

from vector_rag.models.response import Citation, ContextPackage
from vector_rag.utils.logging import get_logger

logger = get_logger("generation.citation")


class CitationExtractor:
    """Parses citation bracket tags (e.g. [1], [2], [Source 1]) and resolves them against ContextPackage."""

    # Matches bracket patterns: [1], [2], [1, 2], [1][2], [Source 1], [Source 2]
    CITATION_PATTERN = re.compile(
        r"\[(?:Source\s*)?(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE
    )

    def extract_referenced_source_ids(self, text: str) -> List[int]:
        """Extract unique sorted source IDs referenced in text."""
        referenced_ids: Set[int] = set()
        matches = self.CITATION_PATTERN.findall(text)

        for match in matches:
            # Handle comma separated lists within brackets e.g. [1, 2]
            parts = match.split(",")
            for p in parts:
                cleaned = p.strip()
                if cleaned.isdigit():
                    referenced_ids.add(int(cleaned))

        return sorted(list(referenced_ids))

    def resolve_citations(
        self,
        answer_text: str,
        context_package: ContextPackage,
    ) -> List[Citation]:
        """
        Cross-reference cited IDs in generated text with the available sources in the context package.
        
        Returns:
            List of validated Citation models corresponding to cited sources.
        """
        cited_ids = self.extract_referenced_source_ids(answer_text)
        chunk_map = {c.source_id: c for c in context_package.chunks}

        resolved: List[Citation] = []
        for source_id in cited_ids:
            if source_id in chunk_map:
                chunk = chunk_map[source_id]
                snippet = (
                    chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text
                )
                resolved.append(
                    Citation(
                        source_id=source_id,
                        filename=chunk.filename,
                        page=chunk.page,
                        chunk_id=chunk.chunk_id,
                        snippet=snippet,
                    )
                )

        logger.info(
            "Resolved %d citation(s) from generated response (found source IDs: %s)",
            len(resolved),
            cited_ids,
        )
        return resolved
