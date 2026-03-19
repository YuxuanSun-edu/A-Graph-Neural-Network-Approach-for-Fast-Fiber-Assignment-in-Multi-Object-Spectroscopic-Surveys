# pipeline.py 
import os
import re
import argparse
import time
import torch
import gc
import csv
import psutil
import threading
import numpy as np
import fnmatch
from concurrent.futures import ProcessPoolExecutor

# 引入功能
from build_gnn_sample import build_training_graph_for_one_instance
from train import run_training_pipeline
from main_gnn_solver import solve_one_npz, load_gnn_model

# ================= 智能配置 =================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "match_map_3600")
MODELS_DIR = os.path.join(CURRENT_SCRIPT_DIR, "models")
# ===========================================

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

# --- 2. 辅助函数 ---
def find_data_pairs(data_dir, pattern="*"):
    if not os.path.exists(data_dir): return []
    all_files = os.listdir(data_dir)
    filtered_files = fnmatch.filter(all_files, pattern)
    
    items_map, limits_map = {}, {}
    p_items = re.compile(r"^(.*)\s+items\.txt$")
    p_limits = re.compile(r"^(.*)\s+limits\.txt$")
    
    for f in filtered_files:
        m_i = p_items.match(f)
        if m_i: items_map[m_i.group(1)] = os.path.join(data_dir, f)
        m_l = p_limits.match(f)
        if m_l: limits_map[m_l.group(1)] = os.path.join(data_dir, f)
        
    ids = sorted(list(set(items_map.keys()) & set(limits_map.keys())))
    return [(uid, items_map[uid], limits_map[uid]) for uid in ids]

def extract_weight_from_file(filepath):
    if not os.path.exists(filepath): return 0.0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            m = re.search(r"weight\s+([\d.E+-]+)", line)
            if m: return float(m.group(1))
    except: pass
    return 0.0

def worker_build_graph(args):
    uid, items_p, limits_p, npz_p = args
    if os.path.exists(npz_p): return
    try:
        build_training_graph_for_one_instance(
            items_path=items_p,
            limits_path=limits_p,
            out_npz_path=npz_p,
            max_outer_iter=0,
            random_seed=42,
            generate_label=True
        )
    except Exception as e:
        print(f"[Build Error] {uid}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=str, default="") 
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--pattern", type=str, default="*", help="Filename pattern")
    parser.add_argument("--effort", type=str, default="normal") 
    args = parser.parse_args()

    data_dir = args.dir
    model_path = args.model
    if not model_path:
        # 如果没指定，优先找 best_mis_gnn.pt，没有则找最新的
        default_model = os.path.join(MODELS_DIR, "best_mis_gnn.pt")
        if os.path.exists(default_model):
            model_path = default_model
        elif os.path.exists(MODELS_DIR):
            pts = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
            if pts:
                pts.sort(key=os.path.getmtime)
                model_path = pts[-1]
            else:
                model_path = "best_mis_gnn.pt"
        else:
            model_path = "best_mis_gnn.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed_dir = os.path.join(data_dir, "processed")
    results_dir = os.path.join(data_dir, "results_gnn")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    csv_report_path = os.path.join(data_dir, "GNN_Performance.csv")

    pairs = find_data_pairs(data_dir, args.pattern)
    print(f">>> [Pipeline] 模式: {'训练' if args.train else '推理'} | 筛选: '{args.pattern}' | 任务数: {len(pairs)}")
    
    if not pairs: return

    # ========================================================
    # Phase 1: 多进程并行建图
    # ========================================================
    print(f"\n>>> [Phase 1] 并行构建图数据 (CPU)...")
    t0 = time.time()
    
    tasks = []
    for uid, items_p, limits_p in pairs:
        npz_p = os.path.join(processed_dir, f"{uid}_reduced_train.npz")
        tasks.append((uid, items_p, limits_p, npz_p))
    
    workers = min(4, os.cpu_count()) 
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(worker_build_graph, tasks))
        
    print(f">>> [Phase 1] 建图完成！耗时: {time.time()-t0:.2f}s")

    # ========================================================
    # Phase 2: 模型训练 (可选)
    # ========================================================
    if args.train:
        print("\n>>> [Phase 2] 模型训练/微调...")
        run_training_pipeline(
            npz_dir=processed_dir, 
            save_path=model_path if os.path.exists(model_path) else os.path.join(MODELS_DIR, "best_mis_gnn.pt"), 
            max_epochs=50
        )
        
        # 重新扫描最新模型
        if os.path.exists(MODELS_DIR):
            pts = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
            if pts:
                pts.sort(key=os.path.getmtime)
                model_path = pts[-1]

    # ========================================================
    # Phase 3: 批量 GPU 求解 & 生成报表
    # ========================================================
    print(f"\n>>> [Phase 3] 批量 GNN 求解与报表生成...")
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，跳过推理。")
        return

    print(f"  -> Loading Model: {model_path}")
    
    # 找一个存在的npz来确定维度
    sample_npz = ""
    for t in tasks:
        if os.path.exists(t[3]):
            sample_npz = t[3]
            break
            
    if not sample_npz:
        print("没有可用的 npz 文件，跳过求解。")
        return
        
    try:
        data_tmp = np.load(sample_npz)
        in_dim = data_tmp["x"].shape[1]
        loaded_model = load_gnn_model(model_path, in_dim=in_dim, device=device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 初始化 CSV
    with open(csv_report_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['UID', 'Time(s)', 'Obj(Weight)', 'Mem(MB)'])

    print(f"  -> 正在写入报表: {csv_report_path}")
    print(f"  -> {'UID':<25} | {'Weight':<15} | {'Time(s)':<8} | {'Mem(MB)':<8}")
    print("-" * 65)

    SAMPLES = 1000 
    
    for uid, items_p, limits_p in pairs:
        npz_p = os.path.join(processed_dir, f"{uid}_reduced_train.npz")
        out_txt = os.path.join(results_dir, f"{uid}_solution.txt")
        
        if not os.path.exists(npz_p): continue

        gc.collect()
        torch.cuda.empty_cache()

        monitor = MemoryMonitor()
        monitor.start()
        t_start_one = time.time()
        
        weight_val = 0.0
        success = False
        
        try:
            solve_one_npz(
                npz_path=npz_p,
                items_path=items_p,
                model_or_path=loaded_model,
                outfile=out_txt,
                samples=SAMPLES,
                device=device
            )
            success = True
        except Exception as e:
            print(f"  [Fail] {uid}: {e}")

        t_cost = time.time() - t_start_one
        monitor.stop()
        monitor.join()
        peak_mem = monitor.get_peak_usage()

        if success:
            weight_val = extract_weight_from_file(out_txt)

        weight_str = f"{weight_val:.1f}"
        time_str = f"{t_cost:.3f}"
        mem_str = f"{peak_mem:.2f}"

        print(f"  -> {uid:<25} | {weight_str:<15} | {time_str:<8} | {mem_str:<8}")

        with open(csv_report_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([uid, time_str, weight_str, mem_str])

    print(f"\n>>> [Pipeline] 全部完成！")
    print(f">>> 完整报表已保存至: {csv_report_path}")

if __name__ == "__main__":
    main()