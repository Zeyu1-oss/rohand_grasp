#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总成功抓取的 pair 分布（基于 pairs_{lego}.npy rich payload）
- 扫描 *.succ.npy，读取 {lego, pair_index}
- 在 pairs_root 下加载 pairs_{lego}.npy，映射到 (p1,p2)、normals、face_ids、pair_axis、epsilon
- 统计：
  * 总成功数、每轴(pair_axis∈{0:x,1:y,2:z})分布
  * 面组合(face_ids)分布，如(0,1),(2,3),(4,5)
  * span=||p2-p1|| 分布（均值/中位/分位）
  * 每个 lego 的成功个数
- 导出：
  * CSV：逐条记录（lego, pair_idx, L, axis, eps, f1, f2, p1..., p2...）
  * NPY：midpoints (N,3)，endpoints (N,2,3)
  * JSON：summary（计数/直方/分位）
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter
from glob import glob
from typing import Dict, Tuple, Any, Optional

import numpy as np

PAIR_FILE_PATTERN = re.compile(r"pair(\d+)_ctrl(\d+)\.succ\.npy$", re.IGNORECASE)

def load_succ(path: str) -> Optional[Dict[str, Any]]:
    try:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
            obj = obj.item()
        if not isinstance(obj, dict):
            return None
        # 兼容不同键名
        lego = obj.get("lego") or obj.get("lego_id")
        pair_index = obj.get("pair_index")
        if lego is None or pair_index is None:
            # 尝试从文件名兜底解析 pair_index
            m = PAIR_FILE_PATTERN.search(os.path.basename(path))
            if m:
                pair_index = int(m.group(1))
        if lego is None or pair_index is None:
            return None
        return {"lego": str(lego), "pair_index": int(pair_index), "raw": obj}
    except Exception:
        return None

def smart_load_pairs(pairs_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(pairs_path):
        return None
    raw = np.load(pairs_path, allow_pickle=True)
    obj = raw
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        try:
            obj = raw.item()
        except Exception:
            obj = raw
    # 支持两种格式：dict payload（推荐）或直接数组
    if isinstance(obj, dict):
        # 需要至少有 pairs(K,2,3)
        if "pairs" not in obj:
            return None
        return obj
    if isinstance(obj, np.ndarray):
        # 退化：若是 (K,2,3) 的数组，也能用
        if obj.ndim == 3 and obj.shape[1:] == (2,3):
            return {"pairs": obj.astype(np.float32)}
    return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="统计成功抓取的 pair 分布")
    ap.add_argument("--succ_root", type=str, required=True,
                    help="保存 .succ.npy 的根目录（递归扫描）")
    ap.add_argument("--pairs_root", type=str, required=True,
                    help="保存 pairs_{lego}.npy 的目录")
    ap.add_argument("--out_dir", type=str, default="results/eval_stats",
                    help="输出目录（CSV/NPY/JSON）")
    ap.add_argument("--pairs_prefix", type=str, default="pairs_",
                    help="pairs 文件前缀，默认 pairs_{lego}.npy")
    ap.add_argument("--save_points", action="store_true",
                    help="保存中点云/端点云 NPY（用于可视化）")
    ap.add_argument("--csv", action="store_true",
                    help="导出逐条 CSV")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # 递归找所有 succ
    succ_files = sorted(glob(os.path.join(args.succ_root, "**", "*.succ.npy"), recursive=True))
    if not succ_files:
        print("[WARN] 没找到任何 .succ.npy")
        return

    # 缓存每个 lego 的 pairs payload
    pairs_cache: Dict[str, Dict[str, Any]] = {}

    # 收集容器
    rows = []                      # 逐条记录（导出 CSV）
    midpoints = []                 # (N,3)
    endpoints = []                 # (N,2,3)
    axis_counter = Counter()       # pair_axis 分布
    face_pair_counter = Counter()  # face 组合分布，如 '(0,1)'
    per_lego_counter = Counter()   # 每 lego 成功计数
    spans = []                     # L = ||p2-p1||

    # 为了鲁棒，face_ids/pair_axis/epsilon 可能缺失 → 用 None 替代
    for fp in succ_files:
        rec = load_succ(fp)
        if rec is None:
            continue
        lego = rec["lego"]
        k = rec["pair_index"]

        # 取 pairs_{lego}.npy
        if lego not in pairs_cache:
            ppath = os.path.join(args.pairs_root, f"{args.pairs_prefix}{lego}.npy")
            payload = smart_load_pairs(ppath)
            if payload is None:
                print(f"[WARN] 缺少/无法解析 pairs 文件：{ppath}")
                pairs_cache[lego] = None
            else:
                pairs_cache[lego] = payload

        payload = pairs_cache.get(lego)
        if not payload:
            continue

        pairs = payload.get("pairs")
        if not isinstance(pairs, np.ndarray) or pairs.ndim != 3 or pairs.shape[1:] != (2,3):
            print(f"[WARN] {lego} 的 pairs 形状异常：{getattr(pairs, 'shape', None)}")
            continue
        if not (0 <= k < pairs.shape[0]):
            print(f"[WARN] pair_index 超界：lego={lego}, k={k}, K={pairs.shape[0]}")
            continue

        p1, p2 = pairs[int(k), 0], pairs[int(k), 1]
        L = float(np.linalg.norm(p2 - p1))
        spans.append(L)
        per_lego_counter[lego] += 1

        # 可选字段
        fids = payload.get("face_ids")
        axis = payload.get("pair_axis")
        epsv = payload.get("epsilon")
        f1 = f2 = None
        ax = None
        eps = None
        if isinstance(fids, np.ndarray) and fids.shape[0] > k and fids.shape[1] == 2:
            f1, f2 = int(fids[k,0]), int(fids[k,1])
            face_pair_counter[f"({f1},{f2})"] += 1
        if isinstance(axis, np.ndarray) and axis.shape[0] > k:
            ax = int(axis[k])
            axis_counter[ax] += 1
        if isinstance(epsv, np.ndarray) and epsv.shape[0] > k:
            eps = float(epsv[k])

        # 记录点云
        endpoints.append(np.stack([p1, p2], axis=0))
        midpoints.append(0.5*(p1 + p2))

        # 行（CSV）
        rows.append({
            "lego": lego,
            "pair_index": int(k),
            "L": L,
            "axis": ax if ax is not None else "",
            "epsilon": eps if eps is not None else "",
            "f1": f1 if f1 is not None else "",
            "f2": f2 if f2 is not None else "",
            "p1x": p1[0], "p1y": p1[1], "p1z": p1[2],
            "p2x": p2[0], "p2y": p2[1], "p2z": p2[2],
        })

    total = len(rows)
    if total == 0:
        print("[INFO] 没有可用的成功样本（或 pairs 映射失败）")
        return

    # —— 汇总统计
    spans_np = np.asarray(spans, float)
    summary = {
        "total_success": int(total),
        "by_axis_counts": {str(k): int(v) for k, v in axis_counter.items()},  # k∈{0,1,2} => x/y/z
        "by_face_pairs_counts": dict(face_pair_counter),
        "by_lego_counts": {k: int(v) for k, v in per_lego_counter.items()},
        "span": {
            "mean": float(np.mean(spans_np)),
            "median": float(np.median(spans_np)),
            "p10": float(np.percentile(spans_np, 10)),
            "p90": float(np.percentile(spans_np, 90)),
            "min": float(np.min(spans_np)),
            "max": float(np.max(spans_np)),
        }
    }

    # —— 导出
    # JSON summary
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] 写出 summary.json @ {args.out_dir}")

    if args.csv:
        import csv
        csv_path = os.path.join(args.out_dir, "success_pairs.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] 写出 {csv_path}")

    if args.save_points:
        mid_np = np.asarray(midpoints, float)
        end_np = np.asarray(endpoints, float)
        np.save(os.path.join(args.out_dir, "midpoints.npy"), mid_np)
        np.save(os.path.join(args.out_dir, "endpoints.npy"), end_np)
        print(f"[OK] 写出 midpoints.npy / endpoints.npy @ {args.out_dir}")

    print("\n=== Summary ===")
    print(f"Total successes: {summary['total_success']}")
    print(f"Axis counts (0:x,1:y,2:z): {summary['by_axis_counts']}")
    print(f"Span mean/median: {summary['span']['mean']:.6f} / {summary['span']['median']:.6f}")
    print(f"Span min/max: {summary['span']['min']:.6f} ~ {summary['span']['max']:.6f}")

if __name__ == "__main__":
    main()
