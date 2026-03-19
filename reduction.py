import os
from typing import List, Dict, Tuple, Set
import numpy as np
from typing import List, Dict, Set

# =========================
# 1. 读取真实 items.txt
# =========================

def load_items_from_file(items_path: str):
    """
    读取真实 items.txt：
        #n sky+std+addon ...   （首行）
        F1624 G2768985589193642496: 4961.6677
        ...

    返回：
        items: List[Tuple[str, str, float]]
            [(fiber_id, star_id, weight), ...]
        key2idx: Dict[str, int]
            "F1624 G2768..." -> item_index
    """
    items: List[Tuple[str, str, float]] = []
    key2idx: Dict[str, int] = {}

    with open(items_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # 跳过 header: "#n sky+std+addon ..."
                first = False
                continue

            # 形如 "F1624 G2768...: 4961.66"
            # 先按 ":" 切开
            try:
                left, w_str = line.split(":")
            except ValueError:
                # 防御性：如果这一行不是合法的，就跳过
                continue

            w = float(w_str.strip())

            # left 部分再按空格拆成 fiber_id, star_id
            parts = left.strip().split()
            if len(parts) != 2:
                continue
            fiber_id, star_id = parts
            key = f"{fiber_id} {star_id}"

            idx = len(items)
            items.append((fiber_id, star_id, w))
            key2idx[key] = idx

    return items, key2idx


# =========================
# 2. 读取真实 limits.txt
# =========================

def load_limits_from_file(limits_path: str, key2idx: Dict[str, int]):
    """
    读取真实 limits.txt：
        #n sky+std+addon ...
        F1624 G...,F1625 G...
        E1524 G...

    每一行代表一个约束集合（组），组内 item 用逗号分隔。

    返回：
        groups: List[List[int]]  （每个 group 是若干 item_index）
    """
    groups: List[List[int]] = []

    with open(limits_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # 跳过 header
                first = False
                continue

            # 一行可能是：
            #   F1624 G...,F1625 G...
            #   或  F1624 G...
            parts = line.split(",")
            group_indices: List[int] = []

            for token in parts:
                token = token.strip()
                if not token:
                    continue
                # token 形如 "F1624 G2768..."
                key = token
                if key in key2idx:
                    group_indices.append(key2idx[key])
                else:
                    # limits 里可能出现 items.txt 中没有的组合，直接忽略即可
                    pass

            if group_indices:
                groups.append(group_indices)

    return groups


# =========================
# 3. ProblemInstance & Reduction
# =========================

class ProblemInstance:
    def __init__(self, items, groups):
        """
        items: List[(fiber_id, star_id, weight)]
        groups: List[List[item_index]]
        """
        self.items = items
        self.groups = groups


class Reducer:
    def __init__(self, instance: ProblemInstance):
        self.original = instance
        self.preselected: List[int] = []   # 必选的原始索引
        self.new2old: Dict[int, int] = {}  # 映射

    def run(self) -> ProblemInstance:
        items = self.original.items
        groups = self.original.groups
        n = len(items)
        
        print(f"[Reduction] 原始规模: {n} 节点, {len(groups)} 组约束")

        # 1. 提取权重向量
        weights = np.array([it[2] for it in items], dtype=np.float32)
        
        # 2. 构建邻接表 (为了速度，先用 set，后续优化可用 CSR)
        adj = [set() for _ in range(n)]
        for g in groups:
            for i in range(len(g)):
                u = g[i]
                for j in range(i + 1, len(g)):
                    v = g[j]
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # 3. 计算“邻居权重和” (Vectorized check preparation)
        #    规则：如果 w[u] >= sum(w[v] for v in N(u))，则必选 u
        neighbor_weight_sum = np.zeros(n, dtype=np.float32)
        for u in range(n):
            for v in adj[u]:
                neighbor_weight_sum[u] += weights[v]
        
        # 4. 执行“权重支配”剪裁
        #    这包含了 Degree-0 (sum=0) 和 强 Degree-1/High-Weight 情况
        #    注意：这里做一个简化处理，只做一轮静态检查，避免递归修改图导致 Python 慢
        dominated_nodes = np.where(weights > neighbor_weight_sum)[0]
        
        # 这里的逻辑需要小心：如果两个相邻节点都满足条件（不可能，因为 w_u > w_v + others 且 w_v > w_u + others 矛盾），
        # 所以这步是安全的并行操作。
        
        preselected_set = set(dominated_nodes)
        
        # 确定要移除的节点：必选点 + 它们的邻居
        to_remove = set()
        for u in preselected_set:
            to_remove.add(u)
            to_remove.update(adj[u])
            
        self.preselected = list(preselected_set)
        
        print(f"[Reduction] 权重支配规则选中: {len(self.preselected)} 个 (包含孤立点)")
        print(f"[Reduction] 连带移除邻居后，共减少: {len(to_remove)} 个节点")

        # 5. 重建图 (Mapping)
        new_items = []
        old2new = {}
        
        for old_idx in range(n):
            if old_idx not in to_remove:
                new_idx = len(new_items)
                new_items.append(items[old_idx])
                old2new[old_idx] = new_idx
                self.new2old[new_idx] = old_idx
                
        # 重建 Groups
        new_groups = []
        for g in groups:
            # 过滤掉已移除的节点，并映射到新 ID
            new_g = [old2new[idx] for idx in g if idx in old2new]
            # 只有大小 >= 2 的组才构成有效约束
            # (大小为 1 的组意味着该点没有冲突了，其实也可以不存，等 GNN 处理)
            if len(new_g) >= 2:
                new_groups.append(new_g)

        reduced_instance = ProblemInstance(new_items, new_groups)
        print(f"[Reduction] 最终剩余: {len(new_items)} 节点")
        
        return reduced_instance

# =========================
# 4. 导出 graph 文件（可选）
# =========================

def export_to_graph_file(instance: ProblemInstance, graph_path: str):
    """
    将 ProblemInstance 导出为 KaMIS / CHILS 格式的 .graph 文件：
        p edge N M
        u v   (0-based 索引)
    """
    items = instance.items
    groups = instance.groups
    n = len(items)

    # 生成边集（group = clique）
    edges = set()
    for g in groups:
        for i in range(len(g)):
            u = g[i]
            for j in range(i + 1, len(g)):
                v = g[j]
                if u == v:
                    continue
                a, b = min(u, v), max(u, v)
                edges.add((a, b))

    m = len(edges)

    with open(graph_path, "w") as f:
        f.write(f"p edge {n} {m}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")

    print(f"[Export] 已输出图文件: {graph_path}")
    print(f"[Export] 节点数 = {n}, 边数 = {m}")


# =========================
# 5. 入口：对真实 items/limits 做 Reduction
# =========================
