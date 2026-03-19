# build_gnn_sample.py
"""
功能：
  对一组 items.txt / limits.txt：
    1) 调用 reduction.py 执行 Reduction（去孤立点 + 必选点）
    2) (可选) 调用 CHILS.py 在缩减图上求 MIS 初解作为标签
    3) 按 GNN 格式构造 6维特征 (x, edge_index, y)
    4) 保存为 npz

特征设计 (6维):
  0: log1p(weight)        <- 自身权重(对数)
  1: log1p(degree)        <- 度数(对数)
  2: sqrt(degree)         <- 度数平方根
  3: log1p(nbr_w_sum)     <- 邻居权重和(对数)
  4: log1p(nbr_w_mean)    <- 邻居权重均值(对数)
  5: log1p(nbr_w_max)     <- 邻居最大权重(对数)
"""

import os
import numpy as np

# ---- 依赖外部模块 ----
from reduction import (
    load_items_from_file,
    load_limits_from_file,
    ProblemInstance,
    Reducer,
)

from CHILS import (
    chils_initial_solution,
    build_adj_list,
)


def build_training_graph_for_one_instance(
    items_path: str,
    limits_path: str,
    out_npz_path: str,
    max_outer_iter: int = 50,
    random_seed: int = 42,
    generate_label: bool = True
):

    print(f"\n[Build] 处理: {os.path.basename(items_path)}")

    if not os.path.exists(items_path):
        raise FileNotFoundError(f"items.txt 不存在: {items_path}")

    # ---------- 1. 加载 & Reduction ----------
    items, key2idx = load_items_from_file(items_path)
    groups = load_limits_from_file(limits_path, key2idx)
    instance = ProblemInstance(items, groups)

    reducer = Reducer(instance)
    reduced_instance = reducer.run()
    n_reduced = len(reduced_instance.items)
    
    # ---------- 2. 准备基础数据 ----------
    # 原始大数值权重
    raw_weights = np.array([it[2] for it in reduced_instance.items], dtype=np.float32)
    
    adj = build_adj_list(reduced_instance)
    raw_degrees = np.array([len(adj[i]) for i in range(n_reduced)], dtype=np.float32)

    # ---------- 3. 构建 6维特征 (关键修改：Log归一化) ----------
    print("[Feature] 构建 6维 Log 归一化特征...")
    
    # 为了防止内存爆炸，我们用简单的循环或列表推导来计算邻居特征
    nbr_w_sum = np.zeros(n_reduced, dtype=np.float32)
    nbr_w_max = np.zeros(n_reduced, dtype=np.float32)
    
    for i in range(n_reduced):
        neighbors = list(adj[i])
        if neighbors:
            # 获取邻居的原始权重
            nw = raw_weights[neighbors]
            nbr_w_sum[i] = np.sum(nw)
            nbr_w_max[i] = np.max(nw)
        else:
            nbr_w_sum[i] = 0.0
            nbr_w_max[i] = 0.0
            
    # 防止除以0
    nbr_w_mean = np.divide(nbr_w_sum, raw_degrees, out=np.zeros_like(nbr_w_sum), where=raw_degrees!=0)

    # === 核心：全部取 log1p，把 10^7 变成 ~16.0 ===
    f0 = np.log1p(raw_weights)
    f1 = np.log1p(raw_degrees)
    f2 = np.sqrt(raw_degrees)
    f3 = np.log1p(nbr_w_sum)
    f4 = np.log1p(nbr_w_mean)
    f5 = np.log1p(nbr_w_max)

    x = np.stack([f0, f1, f2, f3, f4, f5], axis=1).astype(np.float32)
    print(f"  -> 特征矩阵形状: {x.shape}, 最大值: {np.max(x):.2f} (已归一化)")

    # ---------- 4. (可选) CHILS 生成标签 ----------
    sol_reduced = []
    if generate_label:
        # print(f"  -> 运行 CHILS (iter={max_outer_iter})...")
        sol_reduced = chils_initial_solution(
            reduced_instance,
            max_outer_iter=max_outer_iter,
            random_seed=random_seed,
        )
    
    y = np.zeros(n_reduced, dtype=np.int64)
    if generate_label and sol_reduced:
        for idx in sol_reduced:
            if 0 <= idx < n_reduced:
                y[idx] = 1

    # ---------- 5. 构造 edge_index ----------
    edges = []
    for u in range(n_reduced):
        for v in adj[u]:
            edges.append((u, v))
            edges.append((v, u)) # 双向

    if edges:
        edges_np = np.unique(np.array(edges, dtype=np.int64), axis=0)
        edge_index = edges_np.T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    # ---------- 6. 保存 ----------
    # 必须保存 raw_weights，供求解器计算真实的 Obj
    new2old = np.array([reducer.new2old[i] for i in range(n_reduced)], dtype=np.int64)
    preselected = np.array(reducer.preselected, dtype=np.int64)

    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez(
        out_npz_path,
        x=x,                 # 6维归一化特征
        weights=raw_weights, # 原始真实权重 (1e7 级别)
        edge_index=edge_index,
        y=y,
        new2old=new2old,
        preselected=preselected,
    )
    print(f"[Done] 已保存 .npz (E={edge_index.shape[1]})")

if __name__ == "__main__":
    # 测试用
    pass