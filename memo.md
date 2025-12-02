**`__init__.py`**（和rag实现没关系，python自己需要的一个东西）：
负责实现：把整个 LBRAG 当成一个干净的 Python 包对外暴露。（ディレクトリをパッケージとして認識させる）

详细：

集中 `from .xxx import ...`，然后在 `__all__` 里列出来。这样外面可以写 `from lbrag import LBRAGPipeline, HybridRetriever, ...`，不用关心内部模块结构。对应论文里没有具体方法，但它是“工程打包层”，让你实验脚本看起来像在用一个统一的 LBRAG 库。

---

**`integrations.py`**：
负责实现：**所有“接外部世界”的东西**——LLM、翻译 API、向量检索后端、reranker、置信度估计、以及翻译质量评分的 κ。

详细：

* `OpenAIChatGenerator`：包装 OpenAI Chat API，给 pipeline 提供真正的生成模型（论文里的“LLM 推理阶段”）。
* `OpenAITranslator`：用 Chat API 做句子级翻译，且支持 back-translation；对应论文中 CrossRAG / cRAG 里“将多语检索到的文档翻译成一个 pivot 语言（如英文）”。
* `OpenAIEmbeddingRetriever`：用 OpenAI embedding 做 dense retrieval（本地文档向量库版）。
* `TavilyRetriever`：调用 Tavily 的 web / wiki 检索（外部现成向量库版，多语言检索 MRAG）。
* `QdrantRetriever`：对接本地 / 远程 Qdrant 向量库。
* `OpenAIListwiseReranker`：让 LLM 对候选文档做 listwise rerank（对应论文说的“retriever + reranker”架构）。
* `StaticConfidenceEstimator`：一个简单的翻译置信度估计器（方便以后扩展成 learned estimator）。
* `estimate_kappa(...)`：根据长度比例、back-translation 重合度、slot 一致性计算一个类似 κ 的翻译质量分；对应论文里“alignment / slot consistency / translation quality”这类分析指标。

对应论文：

* § Retrieval（retriever & reranker）
* § Cross-lingual RAG / Translation（翻译系统）
* § Ablation / 质量分析（kappa-like 质量估计）

---

**`metrics.py`**：
负责实现：**实验评估指标，尤其是语言一致性和翻译效率指标**。

详细：

* `response_language_consistency(...)`：基于 token 集合的语言一致性（RLC 的一种实现）。
* `response_language_consistency_prob(...)`：基于语言后验分布的概率版一致性。
* `cost_normalized_bridging_efficiency(...)`：你的“翻译效率指标”，(bridged_score - baseline_score) / translation_tokens，对应论文里的“在有限翻译预算下，单位翻译成本带来的性能提升”（就是 CNBE）。

对应论文：

* § Evaluation（语言一致性、效率指标）
* § Results & Discussion 里提到的 “RLC” 和 “Cost-Normalized Bridging Efficiency / Bridging efficiency-like 指标”

---

**`pipeline.py`**：
负责实现：**LBRAG 的主 pipeline 逻辑**（真正的“多语 RAG 算法”就在这里）。

详细：

* `WeightingConfig`：控制检索得分 / 对齐覆盖率 / slot consistency 三个部分的权重，对应你论文里“同时考虑检索质量和翻译对齐质量”的设计。
* `PipelineOutput`：统一返回 answer / evidence / prompt（方便实验分析）。
* `default_pivot(...)`：根据候选文档的语言分布，决定 pivot 语言（大多数是 query 语言，否则 pivot 到 en），对应论文中“pivot 语言选择策略”。
* `LBRAGPipeline`：

  * `run(query)` 的过程就是：

    1. `retriever.retrieve(query)`：多语向量检索（MRAG）
    2. `pivot_selector` 决定 pivot 语言（默认数据驱动，或者强制 `en` 时就变成 CrossRAG）
    3. `TranslationSelector` 根据效率和预算决定哪些文档需要翻译（就是 LBRAG 的 “有限翻译预算”）
    4. 为要翻译的文档调用 `translator.translate(...)` + 句子切分 + 对齐 + slot 匹配
    5. 计算 coverage / slot consistency / kappa，用 `WeightingConfig` 合成 evidence weight
    6. `PromptBuilder.build(...)` 生成带 evidence 的 prompt
    7. `generator.generate(prompt)` 生成最终回答
  * `_translate_and_align`：是 CrossRAG / cRAG & LBRAG 的关键（文档翻译 + 对齐质量评估）。
  * `_build_untranslated_block`：没被选中翻译的文档保留原文，但有较低权重。

对应论文：

* § Methods – Multilingual RAG / Cross-lingual RAG / LBRAG 主体方法
* § In-context strategies（How we use multilingual retrieved docs + translation + alignment）
* § Results 里分析“翻译预算、对齐质量、slot consistency 对性能的影响”的地方

---

**`prompting.py`**：
负责实现：**RAG 的 prompt 结构与证据格式化**。

详细：

* `PromptTemplate`：把系统提示、引用说明、回答说明拆成三个部分，可自定义不同风格的 prompt。
* `PromptBuilder`：

  * `build(question, evidence, target_language)`：把系统提示 + 问题 + 排序后的证据 + 引用说明 + 回答说明拼成一个完整 prompt。
  * `_render_evidence(...)`：把每个 EvidenceBlock 渲染成 `- [ID] (lang) text` 形式，并截断到 `MAX_EVID_CHARS`，对应论文中 Table（RAG Prompt Template）描述的那种“Instructions + Reference Evidence + Question”的结构。

对应论文：

* § RAG Pipelines – Prompt Template 部分
* 附录中展示的 RAG prompt 模板和 evidence 展示格式

---

**`retrieval.py`**：
负责实现：**检索层的通用接口和 hybrid 多路检索器**。
详细：

* `Retriever` / `Reranker` Protocol：定义检索和重排接口，方便插拔（Cohere、OpenAI、Tavily、Qdrant 都可以挂上来）。
* `RetrievalConfig(alpha, top_k)`：控制 dense / rerank 融合权重 alpha 和 top_k。
* `HybridRetriever`：

  * 同时挂多个 retriever（不同语言、不同后端都可以），把结果 merge 在一起；
  * `_collect_candidates`：收集并按 dense_score 合并候选；
  * `_apply_reranker`：如果有 reranker，则调用 reranker 打 rerank_score，再按 `alpha * dense + (1-alpha) * rerank` 排序。

对应论文：

* § Retrieval（多语言向量检索器 + reranking）
* § Multilingual RAG（retrieval over ∪ D_i across languages）
* Ablation：检索器 / reranker 对性能的影响

---

**`selection.py`**：
负责实现：**翻译选择策略（在有限 budget 下选哪些段落翻译）**。
详细：

* `ConfidenceEstimate` / `ConfidenceEstimator`：翻译置信度估计接口（现在有个静态版在 integrations.py 里）。
* `TranslationCandidate`：把一个 DocumentSegment 附上 `relevance` / `confidence` / `cost`，并提供 `efficiency = relevance * confidence / cost`（对应论文里的“translation efficiency / bridging efficiency”概念）。
* `TranslationPlan`：记录哪些被选翻译、哪些被跳过、预算与实际花费。
* `TranslationSelector`:

  * 输入一堆 TranslationCandidate，按 `efficiency` 降序选，在预算内尽量拿到性价比最高的文档；
  * 如果还有剩余预算，会再从 skipped 里挑效率高且能塞进去的。
  * 这是 LBRAG 和传统 CrossRAG 的核心差异：CrossRAG = budget ≈ ∞；LBRAG = 有限 budget + 效率排序。

对应论文：

* § Cross-lingual RAG / LBRAG：

  * “在有限翻译预算下选择最有用的文档进行翻译”
  * “我们定义 bridging efficiency，偏好高效率候选”

---

**`translation.py`**：
负责实现：**句子切分 + 源 / 译文对齐 + slot 匹配 + 对齐质量估计**（翻译质量信号）。
详细：

* `Translator` / `SentenceSplitter` / `SupportsBackTranslation` Protocol：抽象接口。
* `SimpleSentenceSplitter` / `RegexSentenceSplitter`：简单的正则句子切分，支持中英日等功能性断句。
* `greedy_sentence_alignment(...)`：

  * 把 source_sentences 和 target_sentences 做一一 greedy 对齐（顺序匹配），生成 `SentenceAlignment` 列表；
  * 每个 alignment 里有源句、译句、slot_matches。
* `_extract_slot_matches(source, target)`：

  * 提取 source 里的 nums / dates / uppercase tokens 作为“slots”，看这些在 target 里是否保留；
  * 用作“数值、日期、实体是否正确传递”的硬信号。
* `estimate_alignment_quality(alignments, total_sentences)`：

  * `coverage` = 对齐到的句子数 / 总句数；
  * `consistency` = slot 匹配的命中率；
  * 这两个一起被 pipeline 用来算 evidence block 的权重，也参与 kappa 估计。

对应论文：

* § Methods – Cross-lingual RAG：文档级翻译 + alignment + slot consistency
* § Results – 对 alignment / slot consistency 的分析
* 附录有关“numbers / dates / named entities 在翻译中的保持情况”的讨论

---

**`types.py`**：
负责实现：**所有核心数据结构的“标准类型定义”**。
详细：

* `Query`：包含 `text` / `language` / `metadata`，代表一个用户问题（论文中的 Q）。
* `DocumentSegment`：检索到的一个文档片段，含 `identifier` / `text` / `language` / `score` / `metadata`，对应论文中的文档 d / passage。
* `RetrievalCandidate`：在检索阶段使用，包含 segment、dense_score、rerank_score，以及 `final_score(alpha)` 融合函数。
* `TranslationRequest` / `TranslationResult`：翻译接口的输入输出封装。
* `SentenceAlignment`：source_sentence / target_sentence / slot_matches，对应 alignment 结果的数据形式。
* `EvidenceBlock`：pipeline 中的“证据块”：一个 segment + 可选 translated_text + alignment + weight + metadata（有 coverage / slot consistency / kappa / token_count 等）。

对应论文：

* 整篇论文中所有符号化的对象：Q、D、docs、aligned sentences、evidence 等的工程化落地。
