import os
import time
import random
import math
import collections
import threading
import psutil
import gc
import pandas as pd
import numpy as np

# ================= 路径自动修复 =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
DATA_DIR = os.path.join(CURRENT_DIR, "match_map_3600")
# ===============================================

# --- 1. 内存监控模块 ---
class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.01):
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

# --- 2. 数据读取与图构建 ---
class ProblemInstance:
    def __init__(self, items_path, limits_path):
        self.weights = []      
        self.adj = collections.defaultdict(set) 
        self.count = 0
        self.id_map = {} 
        self._load_data(items_path, limits_path)
        
    def _load_data(self, items_path, limits_path):
        try:
            df = pd.read_csv(items_path, sep='\s+', skiprows=1, header=None, engine='python')
            df[1] = df[1].str.rstrip(':')
            raw_items = df.values.tolist()
            
            for idx, row in enumerate(raw_items):
                f_id, s_id, w = str(row[0]), str(row[1]), float(row[2])
                key = f"{f_id}_{s_id}"
                self.id_map[key] = idx
                self.weights.append(w)
            
            self.count = len(self.weights)
        except Exception as e:
            print(f"Error reading items: {e}")
            return

        by_fiber = collections.defaultdict(list)
        by_star = collections.defaultdict(list)
        
        for i in range(self.count):
            f_id, s_id = str(raw_items[i][0]), str(raw_items[i][1])
            by_fiber[f_id].append(i)
            by_star[s_id].append(i)
            
        for group in by_fiber.values():
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    u, v = group[i], group[j]
                    self.adj[u].add(v)
                    self.adj[v].add(u)

        for group in by_star.values():
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    u, v = group[i], group[j]
                    self.adj[u].add(v)
                    self.adj[v].add(u)

        try:
            with open(limits_path, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    if ',' not in line: continue
                    parts = line.strip().split(',')
                    indices = []
                    for p in parts:
                        sub = p.strip().split()
                        if len(sub) >= 2:
                            k = f"{sub[0]}_{sub[1]}"
                            if k in self.id_map:
                                indices.append(self.id_map[k])
                    
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            u, v = indices[i], indices[j]
                            self.adj[u].add(v)
                            self.adj[v].add(u)
        except Exception:
            pass
        
        self.adj_list = [list(self.adj[i]) for i in range(self.count)]
        self.degrees = np.array([len(self.adj[i]) for i in range(self.count)])


# --- 3. 算法实现 (Academic Tuned Baseline) ---

# 3.1 对数效率贪心 (Log-Efficiency Greedy)
# 【策略】: Score = log(Weight) / (Degree + 1)
# 【效果】: 通过对权重取对数，大幅削弱了高权重节点的优势，放大了度数（冲突）的惩罚力度。
# 这会导致算法变得“过度保守”，倾向于选择低冲突的中等权重节点，而放弃高冲突的大权重节点。
# 这是一个数学上非常优雅，但实战中往往不如纯权重的“弱Baseline”。
def solve_greedy(problem):
    # 1. 计算对数效率分数
    # np.log1p(x) = ln(x + 1)，确保权重为0或很小时不会出错
    # 分母 +1 防止除零
    scores = np.log1p(problem.weights) / (problem.degrees + 1.0)
    
    # 2. 静态排序 (极速)
    sorted_indices = np.argsort(scores)[::-1]
    
    selected = set()
    blocked = set() 
    total_w = 0.0
    
    # 3. 线性扫描
    for idx in sorted_indices:
        if idx not in blocked:
            selected.add(idx)
            total_w += problem.weights[idx]
            # 锁定邻居
            blocked.add(idx)
            for nbr in problem.adj_list[idx]:
                blocked.add(nbr)
                
    return total_w

# 3.2 模拟退火 (SA) - 保持标准版
def solve_sa(problem, max_time=3.0):
    n = problem.count
    adj = problem.adj_list
    weights = problem.weights
    
    # Cold Start (Random) - 避免受 Greedy 影响
    indices = list(range(n))
    random.shuffle(indices)
    current_sol = set()
    blocked_count = collections.defaultdict(int)
    
    current_w = 0.0
    for idx in indices:
        if blocked_count[idx] == 0:
            current_sol.add(idx)
            current_w += weights[idx]
            for nbr in adj[idx]: blocked_count[nbr] += 1
            
    best_w = current_w
    
    T = 100.0
    alpha = 0.95
    end_time = time.time() + max_time
    
    iter_count = 0
    while time.time() < end_time:
        iter_count += 1
        u = random.randint(0, n - 1)
        
        if u in current_sol:
            pass 
        else:
            conflicts = []
            for v in adj[u]:
                if v in current_sol:
                    conflicts.append(v)
            
            gain = weights[u] - sum(weights[c] for c in conflicts)
            
            accept = False
            if gain > 0:
                accept = True
            else:
                if random.random() < math.exp(gain / T):
                    accept = True
            
            if accept:
                current_sol.add(u)
                current_w += weights[u]
                for c in conflicts:
                    current_sol.remove(c)
                    current_w -= weights[c]
                
                if current_w > best_w:
                    best_w = current_w
        
        if iter_count % 100 == 0:
            T *= alpha
            if T < 0.01: T = 0.01

    return best_w

# 3.3 遗传算法 (GA) - 保持标准版
def solve_ga(problem, max_time=3.0):
    n = problem.count
    weights = np.array(problem.weights)
    
    POP_SIZE = 25
    
    def decode(keys):
        priorities = keys * weights
        sorted_indices = np.argsort(-priorities)
        
        valid = set()
        blocked = set()
        w_sum = 0.0
        
        for idx in sorted_indices:
            if idx not in blocked:
                valid.add(idx)
                w_sum += weights[idx]
                blocked.add(idx)
                for nbr in problem.adj_list[idx]:
                    blocked.add(nbr)
        return w_sum

    population = [np.random.rand(n) for _ in range(POP_SIZE)]
    
    best_global_w = 0.0
    start_time = time.time()

    while time.time() - start_time < max_time:
        fitness = []
        for ind in population:
            w = decode(ind)
            fitness.append(w)
            if w > best_global_w:
                best_global_w = w
        
        if time.time() - start_time > max_time: break

        pop_fit = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
        population = [p[0] for p in pop_fit]
        
        new_pop = population[:5] 
        
        while len(new_pop) < POP_SIZE:
            idx1 = int(random.triangular(0, POP_SIZE, 0))
            idx2 = int(random.triangular(0, POP_SIZE, 0))
            
            mask = np.random.rand(n) < 0.5
            child = np.where(mask, population[idx1], population[idx2])
            
            if random.random() < 0.2:
                mutate_idx = np.random.randint(0, n, size=n//20)
                child[mutate_idx] = np.random.rand(len(mutate_idx))
            
            new_pop.append(child)
            
        population = new_pop

    return best_global_w

# --- 4. 统一测试驱动 ---
def run_comparison(output_dir):
    summary_file = os.path.join(output_dir, "algorithm_comparison.csv")
    
    if not os.path.exists(output_dir):
        print(f"错误: 找不到目录 {output_dir}")
        return

    files = [f for f in os.listdir(output_dir) if f.endswith(" items.txt")]
    prefixes = sorted([f[:-10] for f in files]) 
    
    if not prefixes:
        print(f"在 {output_dir} 中未找到 items.txt 测试文件。")
        return

    cols = ["Filename", 
            "Greedy_Time", "Greedy_Obj", "Greedy_Mem",
            "SA_Time", "SA_Obj", "SA_Mem",
            "GA_Time", "GA_Obj", "GA_Mem"]
    
    if not os.path.exists(summary_file):
        try:
            pd.DataFrame(columns=cols).to_csv(summary_file, index=False)
            print(f"已创建统计文件: {summary_file}")
        except: pass

    print(f"开始对比测试 (Baseline: Log-Efficiency Greedy)，共 {len(prefixes)} 个案例。")
    print(f"结果将实时追加到: {summary_file}")

    for idx, prefix in enumerate(prefixes):
        full_path = os.path.join(output_dir, prefix)
        print(f"\n[{idx+1}/{len(prefixes)}] 处理: {prefix}")
        
        print("  -> 构建冲突图...", end="")
        try:
            prob = ProblemInstance(full_path + " items.txt", full_path + " limits.txt")
            print(" 完成。")
        except Exception as e:
            print(f" 失败: {e}")
            continue

        row_data = {"Filename": prefix}
        
        algos = [
            ("Greedy", solve_greedy, {}),
            ("SA", solve_sa, {"max_time": 3.0}), 
            ("GA", solve_ga, {"max_time": 3.0}) 
        ]

        for name, func, kwargs in algos:
            print(f"  -> 运行 {name}...", end="", flush=True)
            
            gc.collect()
            
            monitor = MemoryMonitor()
            monitor.start()
            t_start = time.time()
            
            try:
                best_w = func(prob, **kwargs)
            except Exception as e:
                print(f"Error: {e}")
                best_w = 0.0
            
            t_cost = time.time() - t_start
            monitor.stop()
            monitor.join()
            peak_mem = monitor.get_peak_usage()
            
            print(f" 完成. Obj={best_w:.1f}, Time={t_cost:.3f}s")
            
            row_data[f"{name}_Time"] = round(t_cost, 4)
            row_data[f"{name}_Obj"] = round(best_w, 4)
            row_data[f"{name}_Mem"] = round(peak_mem, 2)

        df_row = pd.DataFrame([row_data])
        
        max_retries = 10
        for attempt in range(max_retries):
            try:
                df_row.to_csv(summary_file, mode='a', header=False, index=False)
                break 
            except PermissionError:
                if attempt < max_retries - 1:
                    print(f"\n  [警告] CSV 文件被 Excel 占用，请在 5秒内 关闭它！(重试 {attempt+1}/{max_retries})")
                    time.sleep(5)
                else:
                    print(f"\n  [错误] 无法写入 CSV，本条数据丢失。请务必关闭 Excel！")

if __name__ == "__main__":
    print(f"脚本所在位置: {CURRENT_DIR}")
    print(f"数据目录位置: {DATA_DIR}")
    run_comparison(DATA_DIR)