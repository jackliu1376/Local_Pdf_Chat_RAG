# 检索评测集

给 RAG 检索链路一个可重复的基线，避免凭感觉调参。

## 数据格式

`eval/golden_qa.jsonl`，每行一个 JSON 对象：

| 字段 | 说明 |
|---|---|
| `id` | 唯一标识 |
| `question` | 用户问题 |
| `answer` | 参考答案 |
| `relevant_snippets` | 答案在原文中的短语；评测时按去空白后的子串匹配定位相关 chunk |

## 运行

在仓库根目录执行：

```bash
# 混合检索基线（不重排）
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5

# 加重排
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5 --rerank

# 输出 JSON 报告
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5 \
  --json-out eval/results/baseline.json
```

评测只使用本地 embedding 和可选的重排模型，不调用 Ollama。

默认语料为仓库根目录 PDF + `eval/data/` 下所有支持格式；也可以用 `--corpus` 指定一个或多个文件。

中文对比使用 `.env` 中的：

```env
EMBED_MODEL_NAME=BAAI/bge-base-zh-v1.5
EMBED_QUERY_PREFIX=为这个句子生成表示以用于检索相关文章：
```

## 当前基线（8 个文件，140 个 chunk，46 条问题）

语料：仓库根目录样例 PDF + `eval/data/` 下的 6 个 PPTX 和 1 个数据库课程设计 PDF。

| 配置 | Recall@5 | MRR@5 | nDCG@5 |
|---|---|---|---|
| 英文 embedding + 混合检索 | 0.2591 | 0.2054 | 0.2021 |
| 英文 embedding + 混合检索 + BGE 重排 | 0.8141 | 0.8848 | 0.7960 |
| 中文 BGE embedding + 混合检索 | 0.8149 | 0.8116 | 0.7680 |
| 中文 BGE embedding + 混合检索 + BGE 重排 | **0.8982** | 0.8822 | **0.8446** |

结论：

- 英文 embedding 在中文语料上的候选召回是主要瓶颈，Recall@5 只有 0.2591
- 换成中文 BGE 后，不加重排就已经达到 0.8149，接近“英文 + 重排”的水平
- 中文 BGE + 重排取得最高 Recall@5 和 nDCG@5；MRR 与“英文 + 重排”基本持平
- 当前最佳配置下，46 条问题中只有 `float-bits` 仍然 miss，说明它的相关 chunk 不在 top-10 候选内

## 指标说明

- **Recall@k**：top-k 中命中的相关 chunk 占全部相关 chunk 的比例
- **MRR@k**：第一个相关 chunk 的倒数排名
- **nDCG@k**：按排名加权的相关性，越靠前的相关结果得分越高

## 已知局限

- 46 条问题覆盖 8 个文件；最佳配置下仍有 `float-bits` 等少数题目的相关 chunk 不在 top-10 候选内
- 目前只评测检索，不评测生成答案的 faithfulness 和引用正确率
- 相关 chunk 通过答案片段匹配确定；片段应尽量短且唯一，避免跨 chunk 边界
- `eval/data/` 中的课件和 PDF 是本地评测语料，是否纳入版本控制需自行判断
