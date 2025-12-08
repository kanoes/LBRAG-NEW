
# 📊 Evaluation Metrics Overview

本实验使用六个指标对不同系统（direct / multi / cross / lbrag）的回答质量进行评估：
 **EM、F1、RLC、RLC_OK、Cost、CNBE** 。
它们分别从 *准确度、语言一致性、翻译开销、效率提升* 等维度衡量系统表现。

---

## 1.**Exact Match (EM)**

**定义：**
预测答案与标准答案在经过标准化处理（lowercase、去标点、去多余空白等）后，是否完全一致。

**取值范围：** `0 或 1`

**计算方法：**

```python
normalize_text(pred) == normalize_text(gold)
```

若相同 → **`EM = 1`
否则 →** `EM = 0`

**直觉解释：**
判断“模型是否给出了完全正确的最终答案”。是一个严格指标，常用于开放域 QA。

---

## 2.**F1 Score (language-aware F1)**

**定义：**
根据语言特性对预测答案与标准答案进行 token 比较的 F1。

* 英语、德语、西语等 → 按空格分词
* 中文、日语 → 按“字符（字）”比较

**取值范围：** `0.0 – 1.0`

**计算方式：**

```
precision = |预测 tokens ∩ 标准 tokens| / |预测 tokens|
recall    = |预测 tokens ∩ 标准 tokens| / |标准 tokens|

F1 = 2 * precision * recall / (precision + recall)
```

**直觉解释：**
衡量“模型答得对的程度”，比 EM 更宽松：

* 完全对 → F1 = 1
* 半对 → F1 ≈ 0.5
* 完全错 → F1 = 0

F1 常用于开放域 QA，是比 EM 更鲁棒的准确度指标。

---

## 3. **RLC — Response Language Consistency**

**定义：**
预测答案中属于目标语言字符的比例。

**取值范围：** `0.0 – 1.0`

**计算方式（简化）：**

* 跳过空格、数字、标点
* 计算所有“实质字符”中：
  * 若目标语言是日语 → 平假名/片假名/汉字视为正确
  * 若目标语言是中文 → 汉字视为正确
  * 若目标语言是英语/德语/西语 → a–z 字符视为正确
  * 其它语言有 fallback 规则

```
RLC = 正确语言字符数 / 全字符数
```

**直觉解释：**
判断“模型是否真的用指定语言回答”。

例如：

* 中文回答里掺杂大量英文 → RLC 下降
* 完整纯中文回答 → RLC → 1.0

---

## 4. **RLC_OK — Language Consistency (Binary)**

**定义：**
基于 RLC 的二分类指标：

```
RLC_OK = 1  if RLC >= 0.6
RLC_OK = 0  otherwise
```

阈值  `0.6` 来自常见 multilingual QA 论文，用来判断“语言是否足够一致”。

**直觉解释：**
让语言一致性可直接统计为比例，如：
“有多少回答是语言合格的？”

---

## 5. **Cost — Translation Token Cost**

**定义：**
系统在处理查询时，所有 evidence block 的翻译 token 数之和。

来自：

```python
block.metadata["token_count"]
```

由  `OpenAITranslator.estimate_cost()` 估算。

**取值范围：** 正数（取决于翻译内容量）

**直觉解释：**
衡量系统为了跨语言检索，需要翻译多少文本。
越小越省成本。

---

## 6. **CNBE — Cost-Normalized Bridging Efficiency**

CNBE =（bridged F1 提升） /（翻译总成本）

**定义：**

```
CNBE = (F1_system – F1_direct_baseline) / translation_cost
```

其中 baseline 为 direct（无 RAG）的平均 F1。

**直觉解释：**
表示“每消耗一个翻译 token 能提升多少 F1”。
是本研究核心指标之一，用于衡量跨语言桥接的 **性价比** 。

特点：

* direct 的 CNBE = 0（无翻译）
* 若某系统以较低翻译成本实现显著提升 → CNBE 很高
* 若翻译成本高但效果差 → CNBE 很低或为负

---

# 📝 总览表（方便插入论文）

| 指标             | 范围   | 描述                                    |
| ---------------- | ------ | --------------------------------------- |
| **EM**     | 0/1    | 预测是否与标准完全一致（严格准确度）    |
| **F1**     | 0–1   | token 级别的语言敏感 F1，更宽松的准确度 |
| **RLC**    | 0–1   | 回答是否使用目标语言（语言一致度）      |
| **RLC_OK** | 0/1    | 是否达到语言一致阈值（≥0.6）           |
| **Cost**   | ≥0    | 翻译 token 消耗（跨语言成本）           |
| **CNBE**   | 任意值 | 每个翻译 token 带来的 F1 提升量         |
