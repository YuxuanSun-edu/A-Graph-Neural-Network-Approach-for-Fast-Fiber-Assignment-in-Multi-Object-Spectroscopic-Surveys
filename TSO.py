import sys
import os
import time
import re
import csv
import threading
import psutil
import numpy as np
import networkx as nx
from collections import defaultdict

# ================= 自检与依赖 =================
print(f">>> 论文复现版 Solver (高性能流网络版) 启动... (PID: {os.getpid()})")

# ================= 1. 内存监控模块 =================
class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.keep_running = True
        self.max_memory = 0
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)

    def run(self):
        while self.keep_running:
            try:
                current_mem = self.process.memory_info().rss / (1024 * 1024)
                if current_mem > self.max_memory:
                    self.max_memory = current_mem
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.keep_running = False

    def get_peak_usage(self):
        return max(0, self.max_memory - self.start_memory)

# ================= 2. 数据加载类 (兼容新旧数据) =================
class ProblemInstance:
    def __init__(self, items_path, limits_path):
        self.weights = {}      
        self.nodes = []
        self.key_to_id = {} 
        self.id_to_key = {}
        self.node_info = {} 
        self.conflict_adj = defaultdict(set)
        
        # 辅助映射：StarID -> [NodeID list]
        self.star_id_to_nodes = defaultdict(list)
        
        self._load_data(items_path, limits_path)
        
    def _load_data(self, items_path, limits_path):
        next_id = 0
        
        # --- 1. 读取 Items ---
        with open(items_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#"): continue
                line = line.strip()
                if not line: continue
                
                key_str = None
                weight = 0.0
                
                # 智能解析权重和Key
                if ":" in line:
                    parts = line.split(":")
                    key_str = parts[0].strip()
                    try: weight = float(parts[1].strip())
                    except: continue
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try: weight = float(parts[-1].strip())
                        except: continue
                        key_str = " ".join(parts[:-1]).strip()
                
                if key_str:
                    if key_str not in self.key_to_id:
                        nid = next_id
                        self.key_to_id[key_str] = nid
                        self.id_to_key[nid] = key_str
                        self.weights[nid] = weight
                        self.nodes.append(nid)
                        next_id += 1
                        
                        # 智能提取 Fiber 和 Star
                        tokens = key_str.split()
                        # 假设倒数第一个是 StarID，倒数第二个是 FiberID
                        if len(tokens) >= 2:
                            fiber_part = tokens[-2]
                            star_part = tokens[-1]
                        else:
                            fiber_part = "Unk"
                            star_part = tokens[0] if tokens else "Unk"
                        
                        self.node_info[nid] = {
                            'fiber': fiber_part,
                            'star': star_part,
                            'weight': weight
                        }
                        self.star_id_to_nodes[star_part].append(nid)

        # --- 2. 读取 Limits ---
        if os.path.exists(limits_path):
            with open(limits_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("#"): continue
                    line = line.strip()
                    if not line: continue
                    
                    if "," in line: raw_keys = [k.strip() for k in line.split(",")]
                    else: raw_keys = line.split()
                    
                    group_node_ids = []
                    for k in raw_keys:
                        # 尝试1: 完整匹配
                        if k in self.key_to_id:
                            group_node_ids.append(self.key_to_id[k])
                        # 尝试2: StarID 后缀匹配
                        else:
                            sub = k.split()
                            if sub:
                                s_id = sub[-1]
                                if s_id in self.star_id_to_nodes:
                                    group_node_ids.extend(self.star_id_to_nodes[s_id])
                    
                    # 建立冲突边
                    group_node_ids = list(set(group_node_ids))
                    if len(group_node_ids) >= 2:
                        for i in range(len(group_node_ids)):
                            for j in range(i+1, len(group_node_ids)):
                                u, v = group_node_ids[i], group_node_ids[j]
                                self.conflict_adj[u].add(v)
                                self.conflict_adj[v].add(u)
        
        print(f"   -> [Graph] Nodes: {len(self.nodes)}, Conflict Edges: {sum(len(v) for v in self.conflict_adj.values())//2}")

# ================= 3. 论文核心算法: Netflow + Retreat =================
def solve_paper_baseline(instance: ProblemInstance):
    """
    复现思路：
    Stage 1: 构建二部图网络流 (Source -> Star -> Fiber -> Sink)。
             这一步解决“分配”问题，忽略物理碰撞。
    Stage 2: 基于优先级的 Retreat。
             这一步解决“冲突”问题，丢弃低分目标。
    """
    
    # --- Stage 1: 构建流网络 ---
    print("   -> [Stage 1] 构建网络流图...")
    G = nx.DiGraph()
    SOURCE = 'SOURCE'
    SINK = 'SINK'
    
    # 我们不预先 add_node，直接 add_edge 更快
    edge_to_nid = {}
    
    # 为了避免字符串拼接带来的巨大内存开销，我们使用元组或整数作为中间节点
    # 但为了逻辑清晰，这里还是用字符串
    
    for nid in instance.nodes:
        info = instance.node_info[nid]
        # 加前缀防止重名
        star_node = f"S:{info['star']}"
        fiber_node = f"F:{info['fiber']}"
        
        # Source -> Star (Capacity 1)
        G.add_edge(SOURCE, star_node, capacity=1.0)
        
        # Star -> Fiber (Capacity 1)
        # 这条边代表选择了这个观测 Item
        G.add_edge(star_node, fiber_node, capacity=1.0)
        edge_to_nid[(star_node, fiber_node)] = nid
        
        # Fiber -> Sink (Capacity 1)
        G.add_edge(fiber_node, SINK, capacity=1.0)

    print(f"   -> [Stage 1] 开始求解 MaxFlow (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})...")
    
    t_flow_start = time.time()
    try:
        # 【核心修改】使用 preflow_push，它在大规模图上比 edmonds_karp 快得多
        # 也可以尝试 flow_func=nx.algorithms.flow.shortest_augmenting_path
        flow_value, flow_dict = nx.maximum_flow(
            G, SOURCE, SINK, 
            flow_func=nx.algorithms.flow.preflow_push
        )
    except Exception as e:
        print(f"   [Error] MaxFlow 失败: {e}")
        return set(), 0.0
    
    print(f"   -> [Stage 1] MaxFlow 耗时: {time.time() - t_flow_start:.2f}s")

    # --- 提取 Stage 1 结果 ---
    stage1_solution = []
    # 遍历流字典提取结果
    # flow_dict[u][v] == 1.0 表示选中
    # 只需要看 Star -> Fiber 的边
    for u, neighbors in flow_dict.items():
        if u.startswith("S:"): # 只看 Star 节点发出的流
            for v, flow in neighbors.items():
                if flow > 0.9 and v.startswith("F:"):
                    if (u, v) in edge_to_nid:
                        stage1_solution.append(edge_to_nid[(u, v)])

    print(f"   -> [Stage 1] 初步分配: {len(stage1_solution)} 个观测 (含冲突)")

    # --- Stage 2: Retreat (冲突消解) ---
    # 按照权重从大到小排序，贪心保留
    print("   -> [Stage 2] 执行 Retreat (冲突消解)...")
    
    stage1_solution.sort(key=lambda x: instance.weights[x], reverse=True)
    
    final_solution = set()
    blocked_nodes = set() # 用 set 查重比遍历快
    
    for nid in stage1_solution:
        # 检查是否与已选中的点有冲突
        # 注意：这里只需要检查 nid 的冲突邻居是否已经在 final_solution 中
        is_conflict = False
        
        # 快速检查
        # 只要 nid 的任何邻居被选中了，nid 就不能选
        # 优化：遍历 conflict_adj[nid]
        for neighbor in instance.conflict_adj[nid]:
            if neighbor in final_solution:
                is_conflict = True
                break
        
        if not is_conflict:
            final_solution.add(nid)
            
    total_w = sum(instance.weights[n] for n in final_solution)
    return final_solution, total_w

# ================= 4. 保存与主程序 =================
def save_solution(filename, solution_set, total_weight, instance):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"#n sky+std+addon None obj {len(solution_set)} weight {total_weight:.4f}\n")
        for nid in solution_set:
            key = instance.id_to_key[nid]
            f.write(f"{key}: Selected\n")

def main():
    base_dir = r"C:\Users\89328\Desktop\demo\match_map_3600"
    csv_file_path = os.path.join(base_dir, "PaperBaseline_Performance.csv")
    
    if not os.path.exists(base_dir):
        print(f"❌ 目录不存在: {base_dir}")
        return

    # 排除结果文件，只找 items.txt
    items_files = [f for f in os.listdir(base_dir) if f.endswith("items.txt") and not f.startswith("result") and not f.endswith("result.txt")]
    print(f"✅ 发现 {len(items_files)} 个任务。\n")
    
    print(f"{'UID':<25} | {'Weight':<12} | {'Time(s)':<8} | {'Mem(MB)':<8}")
    print("-" * 65)
    
    # 写入表头
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['UID', 'Algorithm', 'Time(s)', 'Obj(Weight)', 'Mem(MB)'])
    
    for fname in items_files:
        uid = fname.replace(" items.txt", "")
        # 跳过之前的测试文件，或者指定特定文件
        # if "20260117" not in uid: continue 

        items_path = os.path.join(base_dir, fname)
        limits_path = os.path.join(base_dir, fname.replace("items.txt", "limits.txt"))
        
        # 容错：有些 limits 文件可能名字不一样
        if not os.path.exists(limits_path):
             limits_path = os.path.join(base_dir, f"{uid}limits.txt")
        
        try:
            problem = ProblemInstance(items_path, limits_path)
        except Exception as e:
            print(f"⚠️ {uid} 加载失败: {e}")
            continue
            
        if not problem.nodes:
            print(f"⚠️ {uid} 无有效节点，跳过")
            continue

        monitor = MemoryMonitor()
        monitor.start()
        t0 = time.time()
        
        sol, w = solve_paper_baseline(problem)
        
        elapsed = time.time() - t0
        monitor.stop()
        monitor.join()
        peak_mem = monitor.get_peak_usage()
        
        # 保存结果
        save_name = f"{uid}_PaperBaseline_result.txt"
        save_solution(os.path.join(base_dir, save_name), sol, w, problem)
        
        print(f"{uid:<25} | {w:<12.1f} | {elapsed:<8.3f} | {peak_mem:<8.2f}")
        
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([uid, 'Netflow+Retreat', f"{elapsed:.4f}", f"{w:.4f}", f"{peak_mem:.2f}"])

    print(f"\n✅ 完成。")

if __name__ == "__main__":
    main()