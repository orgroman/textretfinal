import argparse
import os
import random
import statistics
from typing import Dict, List, Tuple

os.environ.setdefault(
    "JAVA_TOOL_OPTIONS",
    "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false",
)

from pyserini.pyclass import autoclass
from pyserini.search.lucene import LuceneSearcher

JSdmQueryGenerator = autoclass("io.anserini.search.query.SdmQueryGenerator")


def read_queries_tsv(path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid = (parts[0] or "").strip()
            query = (parts[1] or "").strip()
            if not qid or not query:
                continue
            queries[qid] = query
    return queries


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def average_precision(docids: List[str], rels: Dict[str, int]) -> float:
    num_rel = sum(1 for r in rels.values() if r > 0)
    if num_rel == 0:
        return 0.0
    hit = 0
    s = 0.0
    for i, d in enumerate(docids, start=1):
        if rels.get(d, 0) > 0:
            hit += 1
            s += hit / i
    return s / num_rel


def mean_ap(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], qids: List[str]) -> float:
    scores: List[float] = []
    for qid in qids:
        rels = qrels.get(qid)
        if rels is None:
            continue
        docids = run.get(qid)
        if docids is None:
            continue
        scores.append(average_precision(docids, rels))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def kfold_split(qids: List[str], k: int, seed: int) -> List[List[str]]:
    qids = list(qids)
    rng = random.Random(int(seed))
    rng.shuffle(qids)

    n = len(qids)
    k = max(2, min(int(k), n))

    fold_sizes = [n // k + (1 if i < (n % k) else 0) for i in range(k)]
    folds: List[List[str]] = []
    start = 0
    for sz in fold_sizes:
        folds.append(qids[start : start + sz])
        start += sz
    return folds


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for p in (s or "").split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for p in (s or "").split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def retrieve_docids(
    searcher: LuceneSearcher,
    queries: Dict[str, str],
    qids: List[str],
    k: int,
    threads: int,
    query_generator=None,
) -> Dict[str, List[str]]:
    qs = [queries[qid] for qid in qids]
    res = searcher.batch_search(qs, qids, k=int(k), threads=int(threads), query_generator=query_generator)
    run: Dict[str, List[str]] = {}
    for qid in qids:
        hits = res.get(qid, [])
        run[qid] = [h.docid for h in hits]
    return run


def make_sdm_configs(order_weights: List[float], unorder_weights: List[float]) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for w_o in order_weights:
        for w_u in unorder_weights:
            w_t = 1.0 - float(w_o) - float(w_u)
            if w_t <= 0.0:
                continue
            out.append((float(w_t), float(w_o), float(w_u)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--qrels", default="Files-20260104/qrels_50_Queries")
    parser.add_argument("--index", default="robust04")

    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bm25-k1", type=float, default=0.9)
    parser.add_argument("--bm25-b", type=float, default=0.4)

    parser.add_argument("--no-sdm", action="store_true")
    parser.add_argument("--sdm-order-weights", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--sdm-unorder-weights", default="0.05,0.10,0.15,0.20")

    parser.add_argument("--rm3-fixed-fb-terms", type=int, default=20)
    parser.add_argument("--rm3-fixed-fb-docs", type=int, default=5)
    parser.add_argument("--rm3-fixed-oqw", type=float, default=0.5)

    parser.add_argument("--rm3-fb-terms-grid", default="10,20,40,60")
    parser.add_argument("--rm3-fb-docs-grid", default="3,5,10,20")
    parser.add_argument("--rm3-oqw-grid", default="0.2,0.5,0.8")

    args = parser.parse_args()

    qrels = read_qrels(str(args.qrels))
    judged_qids = sorted(qrels.keys(), key=lambda x: int(x) if x.isdigit() else x)

    all_queries = read_queries_tsv(str(args.queries))
    queries = {qid: all_queries[qid] for qid in judged_qids if qid in all_queries}

    missing = [qid for qid in judged_qids if qid not in queries]
    if missing:
        raise ValueError(f"Missing {len(missing)} judged queries from queries file. Example: {missing[:3]}")

    folds = kfold_split(judged_qids, int(args.folds), int(args.seed))

    order_weights = _parse_csv_floats(str(args.sdm_order_weights))
    unorder_weights = _parse_csv_floats(str(args.sdm_unorder_weights))
    sdm_configs = make_sdm_configs(order_weights, unorder_weights)
    if not sdm_configs:
        raise ValueError("No valid SDM weight combinations. Ensure termWeight=1-order-unorder > 0.")

    rm3_fb_terms = _parse_csv_ints(str(args.rm3_fb_terms_grid))
    rm3_fb_docs = _parse_csv_ints(str(args.rm3_fb_docs_grid))
    rm3_oqw = _parse_csv_floats(str(args.rm3_oqw_grid))

    rm3_configs: List[Tuple[int, int, float]] = []
    for fb_terms in rm3_fb_terms:
        for fb_docs in rm3_fb_docs:
            for oqw in rm3_oqw:
                rm3_configs.append((int(fb_terms), int(fb_docs), float(oqw)))

    use_sdm = not bool(args.no_sdm)

    print("judged_qids", len(judged_qids))
    print("folds", len(folds), "sizes", [len(f) for f in folds])
    print("sdm_configs", len(sdm_configs))
    print("rm3_configs", len(rm3_configs))

    searcher = LuceneSearcher.from_prebuilt_index(str(args.index))
    searcher.set_bm25(float(args.bm25_k1), float(args.bm25_b))

    fold_test_maps: List[float] = []
    chosen: List[Dict[str, object]] = []

    try:
        for fold_idx, test_qids in enumerate(folds, start=1):
            test_set = set(test_qids)
            train_qids = [qid for qid in judged_qids if qid not in test_set]

            print(f"\nfold {fold_idx}/{len(folds)} train={len(train_qids)} test={len(test_qids)}")

            searcher.set_rm3(int(args.rm3_fixed_fb_terms), int(args.rm3_fixed_fb_docs), float(args.rm3_fixed_oqw))

            best_sdm = None
            best_sdm_map = -1.0
            for i, (w_t, w_o, w_u) in enumerate(sdm_configs, start=1):
                qg = JSdmQueryGenerator(float(w_t), float(w_o), float(w_u)) if use_sdm else None
                run_train = retrieve_docids(searcher, queries, train_qids, k=int(args.k), threads=int(args.threads), query_generator=qg)
                m = mean_ap(run_train, qrels, train_qids)
                if m > best_sdm_map:
                    best_sdm_map = m
                    best_sdm = (float(w_t), float(w_o), float(w_u))
                if i % max(1, len(sdm_configs) // 4) == 0 or i == len(sdm_configs):
                    print(f"  sdm {i}/{len(sdm_configs)} best_train_map={best_sdm_map:.4f}")

            if best_sdm is None:
                raise RuntimeError("SDM tuning failed")

            w_t, w_o, w_u = best_sdm
            qg_best = JSdmQueryGenerator(float(w_t), float(w_o), float(w_u)) if use_sdm else None
            print(f"  best_sdm train_map={best_sdm_map:.4f} term={w_t:.4f} order={w_o:.4f} unorder={w_u:.4f}")

            best_rm3 = None
            best_rm3_map = -1.0
            for i, (fb_terms, fb_docs, oqw) in enumerate(rm3_configs, start=1):
                searcher.set_rm3(int(fb_terms), int(fb_docs), float(oqw))
                run_train = retrieve_docids(searcher, queries, train_qids, k=int(args.k), threads=int(args.threads), query_generator=qg_best)
                m = mean_ap(run_train, qrels, train_qids)
                if m > best_rm3_map:
                    best_rm3_map = m
                    best_rm3 = (int(fb_terms), int(fb_docs), float(oqw))
                if i % max(1, len(rm3_configs) // 4) == 0 or i == len(rm3_configs):
                    print(f"  rm3 {i}/{len(rm3_configs)} best_train_map={best_rm3_map:.4f}")

            if best_rm3 is None:
                raise RuntimeError("RM3 tuning failed")

            fb_terms, fb_docs, oqw = best_rm3
            print(f"  best_rm3 train_map={best_rm3_map:.4f} fb_terms={fb_terms} fb_docs={fb_docs} oqw={oqw:.2f}")

            searcher.set_rm3(int(fb_terms), int(fb_docs), float(oqw))
            run_test = retrieve_docids(searcher, queries, test_qids, k=int(args.k), threads=int(args.threads), query_generator=qg_best)
            test_map = mean_ap(run_test, qrels, test_qids)

            fold_test_maps.append(float(test_map))
            chosen.append(
                {
                    "fold": int(fold_idx),
                    "sdm": best_sdm,
                    "rm3": best_rm3,
                    "train_map": float(best_rm3_map),
                    "test_map": float(test_map),
                }
            )

            print(f"  test_map={test_map:.4f}")

    finally:
        searcher.close()

    if not fold_test_maps:
        raise RuntimeError("No CV results")

    mean_map = statistics.mean(fold_test_maps)
    std_map = statistics.pstdev(fold_test_maps) if len(fold_test_maps) > 1 else 0.0

    print("\ncv_test_map_mean", f"{mean_map:.4f}", "cv_test_map_std", f"{std_map:.4f}")
    for c in chosen:
        print(
            "fold",
            c["fold"],
            "sdm",
            c["sdm"],
            "rm3",
            c["rm3"],
            "train_map",
            f"{float(c['train_map']):.4f}",
            "test_map",
            f"{float(c['test_map']):.4f}",
        )


if __name__ == "__main__":
    main()
