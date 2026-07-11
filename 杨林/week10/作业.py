import json
import numpy as np
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "chunks" / "all_semantic.json"

STOCK_NAMES = {
    "000858": "五粮液",
    "002415": "海康威视",
    "300750": "宁德时代",
    "600519": "贵州茅台",
    "601318": "中国平安",
}


def load_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        chunks = json.load(f)

    meta_list = []
    contents = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if len(content) < 10:
            continue
        contents.append(content)
        meta_list.append({
            "content": content,
            "chunk_id": chunk.get("chunk_id", ""),
            "stock_code": chunk.get("metadata", {}).get("stock_code", ""),
            "year": chunk.get("metadata", {}).get("year", ""),
            "page_num": chunk.get("metadata", {}).get("page_num", ""),
            "section": chunk.get("metadata", {}).get("section", ""),
            "source_file": chunk.get("metadata", {}).get("source_file", ""),
        })
    return meta_list, contents


def compute_bm25_scores(query, docs):
    from rank_bm25 import BM25Okapi
    import jieba

    tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = list(jieba.cut(query))
    scores = bm25.get_scores(query_tokens)
    return scores


def search(query, meta_list, contents, top_k=5):
    scores = compute_bm25_scores(query, contents)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] < 1e-6:
            continue
        item = dict(meta_list[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results


def build_answer(query, meta_list, contents, top_k=5):
    results = search(query, meta_list, contents, top_k)
    if not results:
        return "未找到相关内容，无法回答此问题。"

    parts = ["根据检索结果，相关信息如下："]
    for i, item in enumerate(results, 1):
        stock_code = item.get("stock_code", "")
        year = item.get("year", "")
        page = item.get("page_num", "")
        company_name = STOCK_NAMES.get(stock_code, stock_code)

        label = f"[{i}] {company_name}"
        if year:
            label += f" {year}年"
        if page:
            label += f" 第{page}页"

        content = item.get("content", "")
        parts.append(f"\n{label}\n{content}")

    parts.append("\n---\n注：以上信息来自上市公司年度报告")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="本地年报问答系统（BM25关键词检索）")
    parser.add_argument("--query", type=str, default=None, help="查询问题")
    parser.add_argument("--top-k", type=int, default=5, help="返回条数")
    args = parser.parse_args()

    print("加载数据...")
    meta_list, contents = load_data(DATA_PATH)
    print(f"数据加载完成（共 {len(meta_list)} 条文档）")

    print("=" * 60)
    print("本地年报问答系统")
    print(f"数据来源: {DATA_PATH}")
    print("支持公司: 贵州茅台、五粮液、海康威视、宁德时代、中国平安")
    print("输入 'exit' 退出")
    print("=" * 60)

    def run_query(q):
        if not q.strip():
            return
        print(f"\n问题：{q}")
        print("-" * 60)
        answer = build_answer(q, meta_list, contents, args.top_k)
        try:
            print(answer)
        except UnicodeEncodeError:
            print(answer.encode("gbk", errors="replace").decode("gbk"))

    if args.query:
        run_query(args.query)
    else:
        while True:
            try:
                q = input("\n问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() == "exit":
                break
            run_query(q)


if __name__ == "__main__":
    main()
