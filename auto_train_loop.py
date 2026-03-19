import os
import time
import subprocess
import shutil
import glob
import re
import sys
import datetime
import random
import fnmatch

# 引入数据生成器
import data_generator 

# ================= 核心配置 =================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_SCRIPT_DIR  # 假设就在当前目录

TRAIN_DIR = os.path.join(CURRENT_SCRIPT_DIR, "train_data")
EVAL_DIR = os.path.join(CURRENT_SCRIPT_DIR, "match_map_3600")
MODELS_DIR = os.path.join(CURRENT_SCRIPT_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_mis_gnn.pt") 
BACKUP_PATH = os.path.join(MODELS_DIR, "best_mis_gnn.pt.bak")

PIPELINE = os.path.join(CURRENT_SCRIPT_DIR, "pipeline.py")

# 策略配置
TOTAL_LOOPS = 1000       
BATCH_SIZE = 2           
MAX_DATASET_SIZE = 100   
# ========================================================

def run_cmd(cmd_list):
    print(f"[CMD] {' '.join(cmd_list)}")
    try: 
        subprocess.run(cmd_list, check=True, cwd=CURRENT_SCRIPT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"[Error] 命令执行失败: {e}")
        # 不抛出异常，保证循环继续

def cleanup_processed_cache(d, pattern="*"):
    processed_dir = os.path.join(d, "processed")
    if not os.path.exists(processed_dir): return
    target_pattern = f"{pattern}_reduced_train.npz"
    files = os.listdir(processed_dir)
    for f in files:
        if fnmatch.fnmatch(f, target_pattern):
            try: os.remove(os.path.join(processed_dir, f))
            except: pass

def clean_old_synthetic_sources():
    if not os.path.exists(TRAIN_DIR): return
    syn_items = sorted(glob.glob(os.path.join(TRAIN_DIR, "202* items.txt")))
    if len(syn_items) > MAX_DATASET_SIZE:
        num_to_delete = len(syn_items) - MAX_DATASET_SIZE
        for f in syn_items[:num_to_delete]:
            try:
                os.remove(f)
                limits_f = f.replace(" items.txt", " limits.txt")
                if os.path.exists(limits_f): os.remove(limits_f)
                base = os.path.basename(f).replace(" items.txt", "")
                res = os.path.join(TRAIN_DIR, "results_gnn", f"{base}_solution.txt")
                if os.path.exists(res): os.remove(res)
            except: pass

def get_latest_model_from_models_dir():
    if not os.path.exists(MODELS_DIR): return None
    files = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    if not files: return None
    files.sort(key=os.path.getmtime)
    return files[-1]

# ============================================================
# 主流程
# ============================================================
def main():
    print("==================================================")
    print(f"🚀 启动: 自动训练循环")
    print(f"📂 模型路径: {MODEL_PATH}")
    print("==================================================")
    
    # 1. 初始化环境
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    total_start = time.time()

    for i in range(1, TOTAL_LOOPS + 1):
        loop_start = time.time()
        print(f"\n>>> [Loop {i}/{TOTAL_LOOPS}] 正在生成数据并训练...")
        
        # 备份当前最佳模型 (用于回滚)
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH, BACKUP_PATH)

        # --- 步骤 1: 生成模拟数据 ---
        print(f"  [1/4] 生成模拟数据 ({BATCH_SIZE} 组)...")
        for k in range(BATCH_SIZE):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            prefix = f"{timestamp}_{i}_{k}"
            data_generator.generate_dataset(TRAIN_DIR, prefix)
        
        clean_old_synthetic_sources()

        # --- 步骤 2: 教师生成标签 (Inference) ---
        # 这里的逻辑是：用上一轮最好的模型，给新生成的数据打标签(生成 solution.txt)
        if os.path.exists(MODEL_PATH):
            print(f"  [2/4] 运行求解器 (生成/更新标签)...")
            # 注意：这里会生成 solution.txt，作为后续训练的 Ground Truth
            run_cmd(["python", PIPELINE, "--dir", TRAIN_DIR, "--model", MODEL_PATH])
        else:
            print(f"  [2/4] ⚠️ 无可用模型，跳过推理 -> 将使用 CHILS 算法生成冷启动标签...")

        # --- 步骤 3: 学生训练 (Train) ---
        print(f"  [3/4] 开始训练...")
        
        # 必须清理训练集的 .npz 缓存，强迫 pipeline 重新读取刚才生成的 solution.txt
        cleanup_processed_cache(TRAIN_DIR, "202*") 
        
        # 启动训练
        # --train 模式下，pipeline 会调用 train.py，最终把模型保存回 MODELS_DIR/best_mis_gnn.pt
        # 因为我们上面的 MODEL_PATH 已经指向那里了，所以会自动覆盖
        run_cmd(["python", PIPELINE, "--dir", TRAIN_DIR, "--model", MODEL_PATH, "--train"])

        # --- 步骤 4: 维护 ---
        # 检查是否生成了新模型 (其实 train.py 已经覆盖了，这里主要是为了打印确认)
        if os.path.exists(MODEL_PATH):
            print(f"  ✅ 模型已更新: {MODEL_PATH}")
        else:
            print("  ⚠️ 本轮未产出模型 (可能是首轮冷启动或出错)")

        loop_time = time.time() - loop_start
        print(f"✅ Loop {i} 完成 | 耗时: {loop_time:.2f}s | 总耗时: {(time.time() - total_start)/60:.1f} min")

if __name__ == "__main__":
    main()