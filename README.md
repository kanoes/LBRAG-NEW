# LBRAG

Modular Python reference implementation for the Language-Bridged Retrieval-Augmented Generation (LBRAG) framework. The package focuses on orchestration and evaluation primitives; retrieval data sources and heavy models are provided by the user.

## Features

- Hybrid multilingual retrieval with score normalisation, language-diversity interleaving, and optional BM25/search fallbacks.
- Greedy translation budgeting with heuristic confidence estimation (length ratio, slot consistency, optional back-translation).
- Selective translation pipeline supporting greedy or Hungarian sentence alignment and slot-aware diagnostics.
- Prompt construction that enforces sentence-level citations (`[doc_id:sN]`) and weight-aware evidence ordering.
- Metrics for response language consistency (script-aware with neutral tokens) and cost-normalised bridging efficiency.
- Ready-to-use integrations for OpenAI chat/translation/embedding APIs, Qdrant vector search, and Tavily web search.

## Usage

Install the package into your environment and adapt the interfaces with your own retrievers, translators, and generators. A lightweight demonstration is available:

```bash
python examples/demo.py
```

The demo wires mock components to showcase the pipeline flow and the structure of the generated prompt and answer.

## Environment variables

Real-model integrations rely on the following variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for OpenAI chat, translation, embedding, and reranking adapters |
| `TAVILY_API_KEY` | Optional key for the Tavily search retriever |
| `QDRANT_URL` | Base URL of your Qdrant instance (defaults to `http://localhost:6333`) |
| `QDRANT_API_KEY` | Optional API key for Qdrant |

Set these before instantiating the corresponding adapters in `lbrag.integrations`.

## Extending

- Implement `Retriever`, `Reranker`, `Translator`, and `Generator` protocols with your preferred models.
- Supply `ConfidenceEstimator` implementations if you have custom translation-quality heuristics or QE models.
- Provide document metadata such as estimated translation cost or confidence to improve selection quality.
- Replace the default sentence splitter or alignment logic if you operate on languages with custom segmentation requirements.
- Leverage `PipelineConfig` and `PivotStrategy` to customise pivot-language selection policies.

## License

This repository is provided for research and prototyping purposes. Integrate external model weights and datasets in accordance with their respective licenses.
