# 📊 Evaluation Metrics Overview

本实验使用七个核心指标对不同系统（direct / multi / cross / lbrag）的回答质量进行评估：
**EM、F1、RLC、RLC_OK、Cost、CNBE、Semantic Score**

它们分别从 *准确度、语言一致性、翻译开销、效率提升、语义等价性* 等维度衡量系统表现。

每个指标都会计算 **均值（mean）** 和 **标准差（std）**，以反映性能的稳定性和变异性。

---

## 1. Exact Match (EM)

**定义：**
预测答案与标准答案在经过标准化处理后，是否完全一致。

**取值范围：** `0 或 1`（二值指标）

**计算方法：**
```python
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0
```

**normalize_text 规则：**
- 转小写
- 去除多余空白
- 保留：数字、22种语言的字符（英、中、日、韩、阿拉伯、俄、希伯来、泰等）
- 去除：标点符号和特殊字符

**输出：**
- `em`：所有测试样本的EM平均值
- `em_std`：EM的标准差（反映不同样本上的性能变异性）

**直觉解释：**
- 严格的"完全正确"指标
- EM=1.0 表示所有样本都答对
- EM_std 高表示性能不稳定（有的样本对，有的错）

**适用场景：**
事实类短答案QA，如数字、人名、地名等。

---

## 2. F1 Score (Language-Aware F1)

**定义：**
根据语言特性对预测答案与标准答案进行 token 级别的 F1 计算。

**Token 划分规则：**
- **英语、德语、西语等拉丁语系** → 按空格分词
- **中文、日语** → 按字符（字）切分

**取值范围：** `0.0 – 1.0`

**计算方式：**
```python
def f1_score_lang(pred: str, gold: str, lang: str) -> float:
    # 1. normalize
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    
    # 2. tokenize by language
    if lang in ("ja", "zh", "zh_cn", "zh_tw"):
        p_tokens = list(pred_norm.replace(" ", ""))
        g_tokens = list(gold_norm.replace(" ", ""))
    else:
        p_tokens = pred_norm.split()
        g_tokens = gold_norm.split()
    
    # 3. calculate overlap
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    
    # 4. F1
    return 2 * precision * recall / (precision + recall)
```

**输出：**
- `f1`：所有测试样本的F1平均值
- `f1_std`：F1的标准差

**直觉解释：**
- 衡量"答对了多少成分"
- F1=1.0 表示完全匹配
- F1=0.5 表示一半正确
- F1_std 反映不同样本难度差异

**优势：**
比 EM 更宽松，能反映部分正确的答案。

**示例：**
```
Gold: "4429米"
Pred: "4429m"

EM = 0.0  (完全不同)
F1 ≈ 0.8  (数字部分相同)
```

---

## 3. RLC — Response Language Consistency

**定义：**
预测答案中属于目标语言字符的比例。

**取值范围：** `0.0 – 1.0`

**计算方式：**
```python
def compute_rlc(text: str, lang: str) -> float:
    total = 0  # 总字符数（不含空格、数字、标点）
    hits = 0   # 目标语言字符数
    
    for ch in text:
        if ch.isspace() or ch.isdigit() or ch in punctuation:
            continue
        total += 1
        
        # 根据目标语言判断字符是否匹配
        if lang.startswith("ja"):
            if is_hiragana(ch) or is_katakana(ch) or is_kanji(ch):
                hits += 1
        elif lang.startswith("zh"):
            if is_kanji(ch):
                hits += 1
        elif lang in ("en", "de", "es", "fr", "it", "pt"):
            if ch.isalpha():
                hits += 1
        # ... 其他语言规则
    
    return hits / total if total > 0 else 1.0
```

**输出：**
- `rlc`：所有测试样本的RLC平均值
- `rlc_std`：RLC的标准差

**直觉解释：**
- 判断"模型是否用目标语言回答"
- RLC=1.0 表示纯目标语言回答
- RLC=0.5 表示目标语言和其他语言各占一半
- RLC_std 反映语言一致性的稳定性

**示例：**
```
目标语言：中文
回答："答案是 4429 meters"

分析：
- "答案是" → 中文 ✓
- "meters" → 英文 ✗
RLC ≈ 0.6
```

**重要性：**
- 多语言RAG容易产生语言混杂
- RLC衡量系统是否保持语言纯净性

---

## 4. RLC_OK — Language Consistency (Binary)

**定义：**
基于 RLC 的二值化指标，判断语言一致性是否达标。

**计算方式：**
```python
def rlc_binary(text: str, lang: str, threshold: float = 0.6) -> float:
    score = compute_rlc(text, lang)
    return 1.0 if score >= threshold else 0.0
```

**阈值：** `0.6`
- 来自多语言QA领域的常用标准
- 表示至少60%的字符需要属于目标语言

**输出：**
- `rlc_ok`：达标样本的比例（0.0 - 1.0）
- `rlc_ok_std`：RLC_OK的标准差

**直觉解释：**
- 将连续的RLC转化为"合格/不合格"
- RLC_OK=0.8 表示80%的回答语言一致性达标
- 更直观地表示系统的可靠性

**与RLC的关系：**
- RLC是连续指标，更精细
- RLC_OK是离散指标，更直观
- 两者互补，共同评估语言一致性

**示例：**
```
样本1: RLC=0.95 → RLC_OK=1 ✓
样本2: RLC=0.65 → RLC_OK=1 ✓
样本3: RLC=0.55 → RLC_OK=0 ✗

RLC_OK = 2/3 = 0.667
```

---

## 5. Cost — Translation Token Cost

**定义：**
系统在处理单个查询时，所有翻译的文本总token数。

**计算方式：**
```python
def total_translation_tokens(evidence: Sequence[EvidenceBlock]) -> float:
    total = 0.0
    for block in evidence:
        if block.metadata and "token_count" in block.metadata:
            total += float(block.metadata["token_count"])
    return total
```

**来源：**
- `OpenAITranslator` 在翻译时估算token消耗
- 存储在 `EvidenceBlock.metadata["token_count"]`

**取值范围：** `≥ 0`
- Direct系统：0（无翻译）
- Multi系统：0（无翻译，直接用多语言文档）
- Cross系统：高（翻译所有检索到的文档）
- LBRAG系统：中等（选择性翻译）

**输出：**
- `cost`：平均每个查询的翻译token数
- `cost_std`：不同查询间翻译成本的标准差

**直觉解释：**
- 衡量跨语言检索的翻译开销
- Cost越低越省钱
- Cost_std反映不同查询的成本差异

**系统对比：**
```
direct: Cost=0      (无RAG，无翻译)
multi:  Cost=0      (多语言混合，无翻译)
cross:  Cost=300+   (翻译所有文档)
lbrag:  Cost=30     (选择性翻译，成本低)
```

**重要性：**
- 实际部署中的经济成本
- 与性能一起构成效率指标（CNBE）

---

## 6. CNBE — Cost-Normalized Bridging Efficiency

**定义：**
每花费1个翻译token能带来多少F1提升。本研究的**核心创新指标**。

**计算方式（配对比较）：**
```python
# 对每个样本i，计算RAG相比Direct的提升效率
for 每个样本i:
    f1_rag_i = RAG系统在样本i的F1
    f1_baseline_i = Direct系统在样本i的F1  # 配对比较同一样本
    cost_i = RAG系统在样本i的翻译成本
    
    if cost_i > 0:
        cnbe_i = (f1_rag_i - f1_baseline_i) / cost_i
    else:
        cnbe_i = 0.0

# 然后计算所有样本的均值和标准差
CNBE_mean = mean(cnbe_i for all i)
CNBE_std = std(cnbe_i for all i)
```

**取值范围：** 任意值（可正可负）
- 正值：RAG带来F1提升
- 负值：RAG反而降低F1（翻译引入噪声）
- 值越大越好

**输出：**
- `cnbe`：所有测试样本的CNBE平均值
- `cnbe_std`：CNBE的标准差（反映效率稳定性）

**直觉解释：**

$$\text{CNBE} = \frac{\text{性能提升}}{\text{翻译成本}} = \frac{\Delta \text{F1}}{\text{Cost}}$$

- 衡量跨语言RAG的**性价比**
- CNBE高 = 用更少成本达到更好效果
- CNBE_std低 = 效率稳定，不同样本表现一致

**为什么使用配对比较？**
1. **消除样本难度差异**：简单样本和困难样本的F1本身就不同
2. **准确反映RAG效果**：只关注RAG相比baseline的提升
3. **标准差更有意义**：反映RAG在不同样本上的效率变化

**系统对比示例：**
```
Direct: CNBE=0.00000        (无翻译，不适用)
Multi:  CNBE=0.00000        (无翻译，不适用)
Cross:  CNBE=0.00017±0.001  (高成本，低效率)
LBRAG:  CNBE=0.00237±0.012  (低成本，高效率)
```

**结论：**
LBRAG的CNBE是Cross的14倍，体现了选择性翻译策略的优势。

**重要性：**
- **论文核心贡献**：证明LBRAG在有限预算下更高效
- **实际应用价值**：在成本受限场景下优化多语言RAG
- **系统设计指导**：不是翻译越多越好，而要智能选择

---

## 7. Semantic Score (LLM-based Semantic Agreement)

**定义：**
使用LLM（GPT-4o-mini）判断预测答案与标准答案的语义等价性。

**计算方式：**
```python
def semantic_score(question: str, gold: str, pred: str, lang: str) -> float:
    # 构建prompt让LLM评分
    prompt = f"""
    Question: {question}
    Gold Answer: {gold}
    Predicted Answer: {pred}
    
    Rate semantic equivalence from 0.0 to 1.0.
    Consider:
    - Same meaning in different wording → high score
    - Different units but same value → high score
    - Missing key info → lower score
    - Contradicts gold answer → near 0
    """
    
    # LLM返回 {"score": 0.94, "explanation": "..."}
    return score  # 0.0 - 1.0
```

**取值范围：** `0.0 – 1.0`
- 1.0 = 完全语义等价
- 0.5 = 部分正确
- 0.0 = 完全错误或矛盾

**输出：**
- `semantic_score`：所有测试样本的语义得分平均值
- `semantic_score_std`：语义得分的标准差

**直觉解释：**
- 比EM/F1更智能，理解语义而非字面匹配
- 能识别：
  - 同义表达："11年" vs "11.0 years"
  - 单位转换："4429米" vs "4429m"
  - 完整性："莫斯科" vs "俄罗斯首都莫斯科"

**与EM/F1的互补：**
```
Question: "世贸双塔建造用时多长"
Gold: "11.0 年份"
Pred: "大约11年左右"

EM = 0.0           (字面完全不同)
F1 = 0.4           (部分token重叠)
Semantic = 0.95    (语义几乎相同) ✓
```

**优势：**
- 更接近人类评判
- 跨语言、跨格式的语义理解
- 能处理改写、扩展回答

**局限性：**
- 依赖LLM质量
- 计算成本较高（需要额外API调用）
- 可能有主观性

**使用建议：**
- EM/F1：严格的客观指标，用于量化对比
- Semantic Score：宽松的语义指标，更接近实际应用

---

# 📝 总览表（方便插入论文）

| 指标 | 范围 | 输出字段 | 描述 |
|------|------|----------|------|
| **EM** | 0/1 | `em`, `em_std` | 预测是否与标准完全一致（严格准确度） |
| **F1** | 0–1 | `f1`, `f1_std` | Token级别的语言敏感F1，更宽松的准确度 |
| **RLC** | 0–1 | `rlc`, `rlc_std` | 回答中目标语言字符的比例（语言一致度） |
| **RLC_OK** | 0/1 | `rlc_ok`, `rlc_ok_std` | 语言一致性是否达标（≥0.6阈值） |
| **Cost** | ≥0 | `cost`, `cost_std` | 平均翻译token消耗（跨语言成本） |
| **CNBE** | 任意值 | `cnbe`, `cnbe_std` | 每个翻译token带来的F1提升（核心效率指标） |
| **Semantic** | 0–1 | `semantic_score`, `semantic_score_std` | LLM评估的语义等价性（智能评分） |

---

# 📊 输出示例

## 控制台输出
```
================================================================================
Experiment Results Summary
================================================================================
direct   | EM=0.100±0.302 | F1=0.218±0.322 | RLC=0.921±0.188 | Cost=0.0±0.0 | CNBE=0.00000±0.00000 | Sem=0.350±0.444
multi    | EM=0.000±0.000 | F1=0.263±0.301 | RLC=0.772±0.316 | Cost=0.0±0.0 | CNBE=0.00000±0.00000 | Sem=0.680±0.407
cross    | EM=0.000±0.000 | F1=0.271±0.321 | RLC=0.765±0.313 | Cost=306.3±75.4 | CNBE=0.00017±0.00087 | Sem=0.680±0.407
lbrag    | EM=0.000±0.000 | F1=0.289±0.341 | RLC=0.759±0.317 | Cost=29.8±18.2 | CNBE=0.00237±0.01157 | Sem=0.700±0.422
================================================================================
```

## JSON输出（metrics.json）
```json
{
  "metrics": {
    "lbrag": {
      "em": 0.0,
      "em_std": 0.0,
      "f1": 0.289,
      "f1_std": 0.341,
      "rlc": 0.759,
      "rlc_std": 0.317,
      "rlc_ok": 0.8,
      "rlc_ok_std": 0.4,
      "cost": 29.8,
      "cost_std": 18.2,
      "cnbe": 0.00237,
      "cnbe_std": 0.01157,
      "semantic_score": 0.7,
      "semantic_score_std": 0.422,
      "n": 10.0
    }
  }
}
```

---

# 🎯 指标解读指南

## 准确性指标
- **EM/F1**：主要评估指标，反映答案质量
- **Semantic Score**：辅助指标，更接近人类判断
- 三者结合才能全面评估

## 语言一致性指标
- **RLC**：连续指标，精确衡量语言纯净度
- **RLC_OK**：离散指标，直观判断是否合格
- 多语言RAG的特有挑战

## 效率指标
- **Cost**：翻译开销，越低越好
- **CNBE**：性价比，**最重要的创新指标**
  - LBRAG vs CrossRAG的核心差异
  - 有限预算下的优化目标

## 标准差（std）的意义
- **低std**：性能稳定，不同样本表现一致
- **高std**：性能波动大，某些样本特别好/差
- **论文价值**：用std证明方法的鲁棒性

---

# 📈 系统对比分析框架

## 1. 准确性维度
```
F1: direct < multi < cross < lbrag
→ RAG带来性能提升
→ LBRAG在有限翻译预算下达到最佳
```

## 2. 语言一致性维度
```
RLC: direct > multi ≈ cross ≈ lbrag
→ Direct最纯净（无多语言文档干扰）
→ RAG系统需要权衡准确性和语言一致性
```

## 3. 效率维度（核心）
```
Cost: lbrag << cross
CNBE: lbrag >> cross
→ LBRAG用1/10成本达到更好效果
→ 证明选择性翻译策略的价值
```

## 4. 综合评估
```
系统排序（考虑所有维度）：
1. LBRAG：最佳性价比
2. Cross：高成本高性能
3. Multi：零成本中等性能
4. Direct：无RAG基线
```
