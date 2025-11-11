from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .types import EvidenceBlock


MAX_EVID_CHARS = 800

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
        body = self._render_evidence(evidence)
        answer = self._template.answer_instruction.format(language=target_language)
        return "\n\n".join([header, f"Question: {question}", body, citation, answer])

    def _render_evidence(self, evidence: Sequence[EvidenceBlock]) -> str:
        lines = ["Evidence (ranked):"]
        for block in evidence:
            base = f"[{block.segment.identifier}] ({block.segment.language})"
            text = (block.translated_text or block.segment.text).strip()
            if len(text) > MAX_EVID_CHARS:
                text = text[:MAX_EVID_CHARS] + " ...[truncated]"
            lines.append(f"- {base}: {text}")
        return "\n".join(lines)
