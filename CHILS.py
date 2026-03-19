# CHILS.py
import os
import random
from typing import List, Set

# 从 reduction.py 导入内容
from reduction import (
    load_items_from_file,
    load_limits_from_file,
    ProblemInstance,
    Reducer,
)

# ============================================================
# 1. 基于 groups 构造邻接表（给 Python-CHILS 用）
# ============================================================

def build_adj_list(instance: ProblemInstance):
    """
    根据 groups 构造图的邻接表：
      - 节点：0..n-1
      - 每个 group 视作 clique，在组内两两连边
    返回：
      adj: List[set]，adj[u] 是与 u 相邻的节点集合
    """
    n = len(instance.items)
    groups = instance.groups

    adj: List[Set[int]] = [set() for _ in range(n)]
    for g in groups:
        for i in range(len(g)):
            u = g[i]
            for j in range(i + 1, len(g)):
                v = g[j]
                if u == v:
                    continue
                adj[u].add(v)
                adj[v].add(u)
    return adj


# ============================================================
# 2. Python 版 CHILS：加权贪心 + 简单局部搜索
# ============================================================

def chils_initial_solution(instance: ProblemInstance,
                           max_outer_iter: int = 50,
                           random_seed: int = 42):
    """
    CHILS  MWIS 初解：
      1) 加权贪心构造一个独立集
      2) 通过简单的 1 对 k 交换局部搜索提升总权重

    返回：
      solution: List[int]  独立集节点索引（基于 reduced_instance 的索引）
    """
    random.seed(random_seed)

    items = instance.items
    n = len(items)
    weights = [it[2] for it in items]  # weight = 第三个元素

    # 1. 构造邻接表
    adj = build_adj_list(instance)
    degrees = [len(adj[i]) for i in range(n)]

    # -------- Step 1: 加权贪心构造初解 --------
    remaining = set(range(n))
    solution = set()

    while remaining:
        best_node = None
        best_score = -1.0
        for u in remaining:
            # CHILS 思路：度越小、权重越大越优先
            score = weights[u] / (1.0 + degrees[u])
            if score > best_score:
                best_score = score
                best_node = u

        if best_node is None:
            break

        solution.add(best_node)
        # 删掉自己和所有邻居，保证独立集
        to_remove = {best_node} | adj[best_node]
        remaining -= to_remove

    def total_weight(sol_set):
        return sum(weights[u] for u in sol_set)

    print(f"[CHILS-Py] 贪心初解：|S| = {len(solution)}, weight = {total_weight(solution):.3f}")

    # -------- Step 2: 简单局部搜索（1 对 k 交换） --------
    best_solution = set(solution)
    best_w = total_weight(best_solution)

    for it in range(max_outer_iter):
        improved = False

        # 只在“未选中的点”中尝试引入高权点
        candidates = list(set(range(n)) - best_solution)
        candidates.sort(key=lambda u: weights[u], reverse=True)

        for u in candidates:
            # 与当前解中冲突的节点
            conflict_nodes = [v for v in adj[u] if v in best_solution]

            if not conflict_nodes:
                # 完全不冲突，直接加入
                new_solution = set(best_solution)
                new_solution.add(u)
            else:
                # 删除与 u 冲突的所有点，再加入 u（1 对 k 交换）
                new_solution = set(best_solution)
                for v in conflict_nodes:
                    new_solution.remove(v)
                new_solution.add(u)

            new_w = total_weight(new_solution)
            if new_w > best_w:
                best_solution = new_solution
                best_w = new_w
                improved = True
                break  # 找到一次改进就结束本轮，重新开始下一轮

        print(f"[CHILS-Py] 迭代 {it+1}/{max_outer_iter}, 当前 weight = {best_w:.3f}, |S| = {len(best_solution)}")

        if not improved:
            print("[CHILS-Py] 未找到更优邻域解，局部搜索结束。")
            break

    final_solution = sorted(best_solution)
    print(f"[CHILS-Py] 最终解：|S| = {len(final_solution)}, weight = {best_w:.3f}")
    return final_solution


# ============================================================
# 3. 主入口：Reduction + CHILS-Py + 还原到原始 items
# ============================================================

