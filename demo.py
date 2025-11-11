import os, sys
import argparse
from dotenv import load_dotenv
from typing import Sequence

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- now import from the package "lbrag" ---
from lbrag.integrations import (
    OpenAIEmbeddingRetriever,
    OpenAIListwiseReranker,
    OpenAITranslator,
    OpenAIChatGenerator,
)
from lbrag.prompting import PromptBuilder, PromptTemplate
from lbrag.retrieval import HybridRetriever, RetrievalConfig
from lbrag.selection import TranslationSelector
from lbrag.pipeline import LBRAGPipeline, default_pivot, WeightingConfig
from lbrag.types import DocumentSegment, Query


# ---------- tiny multilingual corpus (toy) ----------
def build_corpus() -> tuple[Sequence[DocumentSegment], Sequence[DocumentSegment]]:
    # English segments (about Mount Fuji)
    en_docs = [
        DocumentSegment(
            identifier="en:fuji:wiki1",
            language="en",
            text=(
                "Mount Fuji is the highest mountain in Japan, standing at 3,776.24 meters. "
                "It is an active stratovolcano located on Honshu."
            ),
            metadata={"source": "https://en.wikipedia.org/wiki/Mount_Fuji"},
        ),
        DocumentSegment(
            identifier="en:fuji:travel",
            language="en",
            text=(
                "The elevation of Mt. Fuji is 3,776 meters above sea level. "
                "Climbing season typically runs from early July to early September."
            ),
            metadata={"source": "https://example.com/travel"},
        ),
    ]
    # Japanese segments
    ja_docs = [
        DocumentSegment(
            identifier="ja:fuji:wiki1",
            language="ja",
            text="富士山は日本最高峰であり、標高は3,776.24 mである。活火山で、本州に位置する。",
            metadata={"source": "https://ja.wikipedia.org/wiki/富士山"},
        ),
        DocumentSegment(
            identifier="ja:fuji:guide",
            language="ja",
            text="富士山の標高は3776メートル。登山シーズンは例年7月上旬から9月上旬。",
            metadata={"source": "https://example.jp/guide"},
        ),
    ]
    # (Optional) Chinese segments
    zh_docs = [
        DocumentSegment(
            identifier="zh:fuji:wiki",
            language="zh",
            text="富士山为日本最高峰，海拔3776.24米，位于本州岛，是一座活火山。",
            metadata={"source": "https://zh.wikipedia.org/wiki/富士山"},
        )
    ]
    # you can choose to include zh_docs into one of the indices or create a third retriever
    return en_docs, ja_docs + zh_docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="富士山の標高は？")
    parser.add_argument("--lang", type=str, default="ja")  # target/output language
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--budget", type=float, default=600.0)  # translation budget (approx tokens)
    parser.add_argument("--alpha", type=float, default=0.5)     # retriever dense vs rerank blend
    parser.add_argument("--model-chat", type=str, default="gpt-4o-mini")
    parser.add_argument("--model-embed", type=str, default="text-embedding-3-small")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY.")

    # ----- Build two retrievers to emulate Lq-index and EN-index -----
    en_docs, lq_docs = build_corpus()
    en_retr = OpenAIEmbeddingRetriever(documents=en_docs, embedding_model=args.model_embed)
    lq_retr = OpenAIEmbeddingRetriever(documents=lq_docs, embedding_model=args.model_embed)

    reranker = OpenAIListwiseReranker(model=args.model_chat)
    retr_cfg = RetrievalConfig(alpha=args.alpha, top_k=args.topk)
    retriever = HybridRetriever(
        retrievers={"en": en_retr, args.lang: lq_retr},
        reranker=reranker,
        config=retr_cfg,
    )

    # ----- Translator / Generator -----
    translator = OpenAITranslator(model=args.model_chat)  # uses RegexSentenceSplitter
    generator = OpenAIChatGenerator(
        model=args.model_chat,
        system_instruction="You are a careful assistant. Always answer strictly in {language}. Do not code-switch.",
    )

    # ----- Prompt template & builder -----
    template = PromptTemplate(
        system_instruction=(
            "You are a careful assistant. Always answer strictly in {language}. "
            "Do not code-switch. If evidence conflicts, summarize both before concluding."
        ),
        citation_instruction=(
            "Cite supporting evidence explicitly using [identifier] like [en:fuji:wiki1]. "
            "Do not invent citations."
        ),
        answer_instruction="Answer in {language} with concise steps if helpful.",
    )
    prompt_builder = PromptBuilder(template=template)

    # ----- Translation selector (budget & strategy) -----
    selector = TranslationSelector(budget=args.budget, min_efficiency=1e-6)

    # ----- Weighting for evidence placement -----
    weighting = WeightingConfig(beta_search=0.6, beta_alignment=0.2, beta_slots=0.2)

    # ----- Build pipeline -----
    pipeline = LBRAGPipeline(
        retriever=retriever,
        retriever_alpha=None,       # use retriever.config.alpha
        translator=translator,
        generator=generator,
        prompt_builder=prompt_builder,
        translation_selector=selector,
        weighting=weighting,
        pivot_selector=default_pivot,  # choose between L_q vs en
    )

    # ----- Run -----
    query = Query(text=args.question, language=args.lang)
    output = pipeline.run(query)

    # ----- Pretty print -----
    print("\n==================== LBRAG DEMO ====================")
    print(f"Query({args.lang}): {args.question}")
    print("----------------------------------------------------")
    print("Answer:\n", output.answer)
    print("----------------------------------------------------")
    print("Top Evidence (after weighting):")
    for i, eb in enumerate(output.evidence, 1):
        translated = "translated" if eb.translated_text else "raw"
        snippet = (eb.translated_text or eb.segment.text)[:160].replace("\n", " ")
        print(f"{i:02d}. [{eb.segment.identifier}] ({eb.segment.language}) "
              f"{translated}  weight={eb.weight:.3f}")
        print(f"     -> {snippet}...")
    print("----------------------------------------------------")
    print("Prompt (sent to generator) [first 800 chars]:")
    print(output.prompt[:800])
    print("====================================================\n")

if __name__ == "__main__":
    main()
