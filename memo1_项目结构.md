# memo1｜项目结构（仅 lbrag/ 与 experiments/）

## lbrag/

- **`__init__.py`**：包入口；集中导出核心类/函数，便于 `from lbrag import ...`。

- **`types.py`**：核心数据结构定义。
  - `Query`、`DocumentSegment`、`RetrievalCandidate`、`EvidenceBlock`、`TranslationRequest/Result`、`SentenceAlignment` 等。

- **`retrieval.py`**：检索与重排抽象 + 融合检索器。
  - `Retriever` / `Reranker` 协议
  - `RetrievalConfig(alpha, top_k)`
  - `HybridRetriever`：收集候选、（可选）rerank、融合排序并截断到 `top_k`。

- **`integrations.py`**：对接外部能力（OpenAI等）。
  - `OpenAIEmbeddingRetriever`：向量检索（支持本地缓存的 embedding/FAISS index；支持 `exclude_same_language` 过滤策略）
  - `OpenAIListwiseReranker`：LLM listwise rerank
  - `OpenAITranslator`：翻译（含 token_count 估算）
  - `OpenAIChatGenerator`：生成
  - `estimate_kappa(...)`：翻译质量/一致性相关的启发式评分

- **`translation.py`**：翻译辅助与对齐质量信号。
  - 句子切分（`RegexSentenceSplitter` 等）
  - `greedy_sentence_alignment`、slot match、coverage/slot consistency 估计

- **`selection.py`**：翻译预算与选择策略（LBRAG 的“选哪些证据翻译”）。
  - `TranslationCandidate`（relevance/confidence/cost/efficiency）
  - `TranslationSelector(budget=...)` 输出 `TranslationPlan`

- **`prompting.py`**：RAG prompt 构建。
  - `PromptTemplate`（system/citation/answer 三段）
  - `PromptBuilder.build(question, evidence, target_language)` 生成最终 prompt

- **`pipeline.py`**：主流程（检索→（可选）翻译/对齐→证据加权→prompt→生成）。
  - `LBRAGPipeline.run(query) -> PipelineOutput(answer, evidence, prompt)`
  - `default_pivot(...)` 选择 pivot 语言
  - `WeightingConfig` 控制 search/alignment/slots 三部分权重

- **`metrics.py`**：库内的通用指标实现（如语言一致性、CNBE 等函数级指标）。

- **`llm.py`**：LLM 客户端封装与用量统计。
  - `LLMClient`：chat / embed
  - `format_usage_summary(...)`：汇总 prompt/completion/total tokens、request_count 等

---

## experiments/

- **`scripts/experiment.py`**：实验主脚本。
  - 构建知识库文档（`samples_to_documents`）
  - 初始化四个系统：`direct / multi / cross / lbrag`
  - 采样测试 query（按 `quid` 分组后随机选取不同 `quid`，每个 `quid` 再随机选一个语言样本）
  - 逐样本评估并写出结果（answers.jsonl、metrics.json、figures）

- **`scripts/prepare_mkqa_samples.py`**：样本准备/预处理脚本（生成实验用数据）。

- **`results/<run_id>/`**：每次实验的输出目录。
  - `answers/<run_id>_answers.jsonl`：逐条样本的详细记录（含预测、指标、检索证据、prompt 等）
  - `metrics/<run_id>_metrics.json`：聚合指标（mean/std、n、用量等）
  - `figures/`：可视化图（EM/F1、RLC、Cost/CNBE、Semantic 等）

（数据与 embedding 缓存一般位于 `experiments/data/<dataset_dir>/`：数据文件 + `embeddings_*.index` / `embeddings_*.pkl` 等缓存。）
