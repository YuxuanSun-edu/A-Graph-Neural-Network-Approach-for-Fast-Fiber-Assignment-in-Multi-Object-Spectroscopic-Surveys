# data_generator.py (基于 LAMOST 真实物理参数重构版)
import time
import random
import collections
import os
import math
import numpy as np
from scipy.spatial import KDTree

# =======================================================
# 1. LAMOST 物理参数配置 (单位: mm)
# 参考论文: Real-time Collision-free Motion Planning...
# =======================================================
FOCAL_PLANE_DIAMETER = 1750.0       # 焦面直径 1.75m 
FOCAL_PLANE_RADIUS = FOCAL_PLANE_DIAMETER / 2.0
UNIT_DIST = 25.6                    # 单元中心间距 25.6mm 
OBS_RADIUS = 16.5                   # 观测半径 16.5mm 
AVG_STARS_PER_UNIT = 4              # 平均每个单元覆盖4颗星 

# 机械碰撞阈值：
# 论文提到光纤头会有机械干涉 。
# 虽然论文未给出明确的碰撞直径数值，但根据 25.6mm 的间距和图示，
# 设定一个保守的物理安全距离 (例如 6mm，即两颗星距离小于此值时，不同光纤同时观测会打架)
COLLISION_THRESHOLD = 6.0           

# =======================================================
# 2. 核心生成逻辑
# =======================================================

def generate_lamost_fibers():
    """
    生成符合 LAMOST 焦面的六边形蜂窝状光纤单元布局
    返回: fibers (list of [x, y, id])
    """
    fibers = []
    
    # 六边形网格生成逻辑
    # 垂直行间距 = D * sin(60°)
    dy = UNIT_DIST * math.sqrt(3) / 2
    dx = UNIT_DIST
    
    # 覆盖范围略大于焦面以确保边缘覆盖
    num_rows = int(FOCAL_PLANE_DIAMETER / dy) + 2
    num_cols = int(FOCAL_PLANE_DIAMETER / dx) + 2
    
    # 以 (0,0) 为中心生成
    start_y = - (num_rows * dy) / 2
    start_x = - (num_cols * dx) / 2
    
    fiber_id_counter = 1
    
    for row in range(num_rows):
        y = start_y + row * dy
        # 奇数行水平偏移半个间距
        offset = (dx / 2) if (row % 2 == 1) else 0
        
        for col in range(num_cols):
            x = start_x + col * dx + offset
            
            # 只有在焦面圆内的单元才保留
            dist_to_center = math.sqrt(x**2 + y**2)
            if dist_to_center <= FOCAL_PLANE_RADIUS:
                fibers.append([x, y, fiber_id_counter])
                fiber_id_counter += 1
                
    return fibers

def generate_random_stars(total_stars):
    """
    在焦面圆内生成均匀分布的星体
    """
    stars = []
    count = 0
    while count < total_stars:
        # 在外接正方形内生成
        x = random.uniform(-FOCAL_PLANE_RADIUS, FOCAL_PLANE_RADIUS)
        y = random.uniform(-FOCAL_PLANE_RADIUS, FOCAL_PLANE_RADIUS)
        
        # 拒绝采样：只保留圆内的
        if x**2 + y**2 <= FOCAL_PLANE_RADIUS**2:
            # 权重模拟：2000 ~ 5000 (保持你之前的逻辑)
            weight = random.uniform(2000.0, 5000.0)
            stars.append([x, y, weight, count]) # count 作为 star_id
            count += 1
    return stars

def build_candidates_and_conflicts(fibers, stars):
    """
    构建候选观测关系 (Candidates) 和 冲突组 (Limits)
    """
    fiber_data = np.array(fibers) # [[x, y, id], ...]
    star_data = np.array(stars)   # [[x, y, w, id], ...]
    
    fiber_coords = fiber_data[:, 0:2]
    star_coords = star_data[:, 0:2]
    
    # 1. 使用 KDTree 加速距离查询
    # tree_fiber = KDTree(fiber_coords) # 不查 Fiber tree，因为观测半径是针对 Fiber 的
    tree_star = KDTree(star_coords)
    
    candidates = [] # [fiber_id, star_id, weight]
    
    # --- A. 构建候选关系 (Assignments) ---
    # 遍历每个 Fiber，找出其 OBS_RADIUS 内的 Stars
    # 使用 query_ball_point 批量查找
    indices_list = tree_star.query_ball_point(fiber_coords, r=OBS_RADIUS)
    
    for i, star_indices in enumerate(indices_list):
        f_id = int(fiber_data[i][2])
        for s_idx in star_indices:
            s_data = star_data[s_idx]
            s_id = int(s_data[3])
            w = float(s_data[2])
            candidates.append([f_id, s_id, w])
            
    # --- B. 构建冲突 (Limits) ---
    limit_groups = []
    group_keys = set()
    
    def add_group(g_items):
        if len(g_items) > 1:
            # 排序并转 tuple 以去重
            g_sorted = sorted(list(set(g_items)))
            key = tuple(g_sorted)
            if key not in group_keys:
                group_keys.add(key)
                limit_groups.append(g_sorted)

    # 1. "一星一纤" 冲突：同一个 Star 被多个 Fiber 覆盖
    star_to_fibers = collections.defaultdict(list)
    for row in candidates:
        f_id, s_id, _ = row
        star_to_fibers[s_id].append((f_id, s_id))
    
    for s_id, items in star_to_fibers.items():
        add_group(items)
        
    # 2. "一纤一星" 冲突：同一个 Fiber 覆盖多个 Stars
    fiber_to_stars = collections.defaultdict(list)
    for row in candidates:
        f_id, s_id, _ = row
        fiber_to_stars[f_id].append((f_id, s_id))
        
    for f_id, items in fiber_to_stars.items():
        add_group(items)
        
    # 3. "机械碰撞" 冲突 (Physical Collisions)：
    # 如果两个 Star 距离太近 (< COLLISION_THRESHOLD)，
    # 且它们分别被分配给了 *不同* 的 Fiber (且这两个 Fiber 是邻居)，
    # 那么这两条观测指令不能同时存在。
    # 简化处理：只要两个 Star 距离过近，它们对应的所有 Candidate 组合都视作潜在冲突
    
    # 找出所有距离 < COLLISION_THRESHOLD 的星体对
    collision_pairs = tree_star.query_pairs(r=COLLISION_THRESHOLD)
    
    for s1_idx, s2_idx in collision_pairs:
        s1_id = int(star_data[s1_idx][3])
        s2_id = int(star_data[s2_idx][3])
        
        # 获取这两个星体涉及的所有候选任务
        # s1 对应的任务列表: [(f_a, s1), (f_b, s1)...]
        tasks_1 = star_to_fibers.get(s1_id, [])
        tasks_2 = star_to_fibers.get(s2_id, [])
        
        if tasks_1 and tasks_2:
            # 两两组合形成互斥对
            for t1 in tasks_1:
                for t2 in tasks_2:
                    # t1 和 t2 不能共存
                    # 只有当它们属于不同光纤时才构成物理碰撞 (同光纤的情况已被"一纤一星"覆盖)
                    if t1[0] != t2[0]: 
                        add_group([t1, t2])

    return candidates, limit_groups

# =======================================================
# 3. 文件写入辅助
# =======================================================
def write_candidates(output_filename, result_list):
    list_len = len(result_list)
    total_weight = sum(row[2] for row in result_list)
    with open(output_filename, 'w', encoding='utf-8') as f:
        # 保持与旧格式兼容的 Header
        f.write(f"#n sky+std+addon None obj {list_len} weight {total_weight:E}\n")
        for row in result_list:
            # F{fiber_id} G{star_id}: {weight}
            f.write(f"F{int(row[0])} G{int(row[1])}: {float(row[2]):.6f}\n")

def write_conflicts(output_filename, limit_groups):
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("#n sky+std+addon None obj None weight None \n")
        for group in limit_groups:
            parts = [f"F{int(f)} G{int(s)}" for f, s in group]
            f.write(", ".join(parts) + "\n")

# =======================================================
# 4. 对外接口 (auto_train_loop.py 调用)
# =======================================================
def generate_dataset(output_dir, prefix):
    """
    生成一组数据并保存到 output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 随机种子设为当前时间微秒，保证每次生成都不一样
    seed_val = int(time.time() * 1000) % (2 ** 31 - 1)
    np.random.seed(seed_val)
    random.seed(seed_val)
    
    # 1. 生成物理光纤盘 (约 4000 个)
    fibers = generate_lamost_fibers()
    num_fibers = len(fibers)
    
    # 2. 决定星体数量 (依据论文: 平均每个单元 4 颗星 -> 约 16000 颗)
    # 这里加一点随机波动 (15000 ~ 17000)
    num_stars = random.randint(15000, 17000)
    
    print(f"[Generator] {prefix} | 物理建模: 焦面1.75m, {num_fibers}单元, {num_stars}目标 ...")
    
    # 3. 生成星体
    stars = generate_random_stars(num_stars)
    
    # 4. 计算关系与冲突
    candidates, limit_groups = build_candidates_and_conflicts(fibers, stars)
    
    # 5. 写入文件
    items_file = os.path.join(output_dir, f"{prefix} items.txt")
    limits_file = os.path.join(output_dir, f"{prefix} limits.txt")
    
    write_candidates(items_file, candidates)
    write_conflicts(limits_file, limit_groups)
    
    return items_file, limits_file

if __name__ == "__main__":
    # 简单测试
    generate_dataset("./test_gen", "test_physics")