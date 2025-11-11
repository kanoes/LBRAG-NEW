# LBRAG

Modular Python reference implementation for the Language-Bridged Retrieval-Augmented Generation (LBRAG) framework. The package focuses on orchestration and evaluation primitives; retrieval data sources and heavy models are provided by the user.

## Features

- Hybrid multilingual retrieval abstraction with pluggable dense retrievers and rerankers.
- Translation budget allocation via greedy efficiency-based selection.
- Sentence-level alignment with slot consistency estimation for selective translations.
- Prompt construction and weighted evidence handling tailored to cross-lingual QA.
- Utility metrics for response language consistency and cost-normalized bridging efficiency.

## Usage

Install the package into your environment and adapt the interfaces with your own retrievers, translators, and generators. A lightweight demonstration is available:

```bash
python demo.py
```

The demo wires mock components to showcase the pipeline flow and the structure of the generated prompt and answer.

## Extending

- Implement `Retriever`, `Reranker`, `Translator`, and `Generator` protocols with your preferred models.
- Provide document metadata such as estimated translation cost or confidence to improve selection quality.
- Replace the default sentence splitter or alignment logic if you operate on languages with custom segmentation requirements.

## License

This repository is provided for research and prototyping purposes. Integrate external model weights and datasets in accordance with their respective licenses.
