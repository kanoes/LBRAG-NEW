from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .types import EvidenceBlock, SentenceAlignment


@dataclass(frozen=True)
class PromptTemplate:
    system_instruction: str
    citation_instruction: str
    answer_instruction: str


class PromptBuilder:
    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    def build(self, question: str, evidence: Sequence[EvidenceBlock], target_language: str) -> str:
        header = self._template.system_instruction.format(language=target_language)
        citation = self._template.citation_instruction
        evidence_text = self._render_evidence(evidence)
        answer_header = self._template.answer_instruction.format(language=target_language)
        return "\n\n".join(
            [
                header,
                f"Question ({target_language}): {question}",
                evidence_text,
                citation,
                answer_header,
            ]
        )

    def _render_evidence(self, evidence: Sequence[EvidenceBlock]) -> str:
        lines = ["Evidence:"]
        for block in evidence:
            pivot = block.pivot_language or block.segment.language
            meta_conf = block.metadata.get("translation_confidence", 0.0)
            header = (
                f"- [{block.segment.identifier}] {block.segment.language}->{pivot} "
                f"| weight={block.weight:.2f} | confidence={meta_conf:.2f}"
            )
            lines.append(header)
            if block.alignment:
                lines.extend(self._render_alignment(block.alignment))
            else:
                snippet = block.translated_text or block.segment.text
                lines.append(f"  • {snippet}")
        return "\n".join(lines)

    def _render_alignment(self, alignments: Sequence[SentenceAlignment]) -> Iterable[str]:
        for alignment in alignments:
            target = alignment.target_sentence
            source = alignment.source_sentence
            yield f"  • {target}"
            yield f"    ↳ Source: {source}"
