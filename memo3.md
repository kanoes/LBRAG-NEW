关于数据集的获取和构建   以及实验和评估流程(我们的实验采用24种语言)

为了让RAG面对真实跨语言常见，我们要构建真正的 cross-lingual-only，简单来说也就是：

* 对于英语 query，移除 en 文档；
* 对于西语 query，移除 es 文档；
* 对于中文 query，移除 zh_cn 文档；
* …

因此做如下2件事情：

1. 在问某种语言的问题之后，检索时检索器自动屏蔽该语言的所有文档（exclude_same_language: bool = True）
2. 对于每一个问题，实验时随机抽选其一种语言进行测试（select_samples_by_quid）


memo: 目前实验中，向量检索会检索10份文件
