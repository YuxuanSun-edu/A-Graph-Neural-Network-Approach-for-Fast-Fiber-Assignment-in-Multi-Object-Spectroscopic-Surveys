import torch
import re
import torch.nn.functional as F
from torch import Tensor
import numpy as np

# ============================================================
# 1. GPU 并行采样
# ============================================================
@torch.no_grad()
def parallel_gumbel_sampling(logits: Tensor, edge_index: Tensor, num_samples: int, temperature=1.0):
    device = logits.device
    N = logits.numel()
    
    # 生成噪声
    U = torch.rand((num_samples, N), device=device)
    g = -torch.log(-torch.log(U + 1e-10) + 1e-10)
    
    # Logits 广播
    scores = (logits.unsqueeze(0) / temperature + g) # [S, N]

    active_mask = torch.ones((num_samples, N), dtype=torch.bool, device=device)
    selected_mask = torch.zeros((num_samples, N), dtype=torch.bool, device=device)
    src, dst = edge_index 

    for _ in range(100): 
        if not active_mask.any():
            break

        current_scores = scores.clone()
        current_scores[~active_mask] = -1e9 

        s_src = current_scores[:, src] 
        s_dst = current_scores[:, dst]
        
        lost_mask = (s_src < s_dst)
        
        # 修正：使用 batch_offset 处理并行索引
        batch_offset = torch.arange(num_samples, device=device).view(-1, 1) * N
        flat_src = (src.unsqueeze(0) + batch_offset).view(-1)
        
        is_suppressed_flat = torch.zeros(num_samples * N, dtype=torch.bool, device=device)
        flat_lost = lost_mask.view(-1)
        
        if flat_lost.any():
            mask_indices = flat_src[flat_lost]
            is_suppressed_flat.index_fill_(0, mask_indices, True)
            
        is_suppressed = is_suppressed_flat.view(num_samples, N)
        
        newly_selected = active_mask & (~is_suppressed)
        if newly_selected.sum() == 0:
            break
            
        selected_mask |= newly_selected
        
        s_selected = newly_selected[:, src]
        to_remove_flat = torch.zeros(num_samples * N, dtype=torch.bool, device=device)
        flat_dst = (dst.unsqueeze(0) + batch_offset).view(-1)
        flat_s_selected = s_selected.view(-1)
        
        if flat_s_selected.any():
            remove_indices = flat_dst[flat_s_selected]
            to_remove_flat.index_fill_(0, remove_indices, True)
            
        to_remove = to_remove_flat.view(num_samples, N)
        active_mask &= ~(newly_selected | to_remove)

    return selected_mask

# ============================================================
# 2. 冲突过滤
# ============================================================
@torch.no_grad()
def gpu_conflict_filter(selected: Tensor, edge_index: Tensor, scores: Tensor):
    src, dst = edge_index
    s_u = selected[:, src]
    s_v = selected[:, dst]
    conflict = s_u & s_v 

    if conflict.sum() == 0:
        return selected

    score_u = scores[:, src]
    score_v = scores[:, dst]
    keep_u = score_u >= score_v
    
    mask_u_bad = conflict & (~keep_u)
    mask_v_bad = conflict & keep_u
    
    device = selected.device
    S, N = selected.shape
    
    batch_offset = torch.arange(S, device=device).view(-1, 1) * N
    flat_src = (src.unsqueeze(0) + batch_offset).view(-1)
    flat_dst = (dst.unsqueeze(0) + batch_offset).view(-1)
    
    kill_flat = torch.zeros(S * N, dtype=torch.bool, device=device)
    
    if mask_u_bad.any():
        indices_u = flat_src[mask_u_bad.view(-1)]
        kill_flat.index_fill_(0, indices_u, True)
        
    if mask_v_bad.any():
        indices_v = flat_dst[mask_v_bad.view(-1)]
        kill_flat.index_fill_(0, indices_v, True)
        
    kill_mask = kill_flat.view(S, N)
    return selected & (~kill_mask)

# ============================================================
# 3. 局部搜索
# ============================================================
@torch.no_grad()
def gpu_local_search(selected: Tensor, edge_index: Tensor, weights: Tensor, steps: int = 50):
    S, N = selected.shape
    src, dst = edge_index
    device = selected.device

    w = weights.unsqueeze(0)
    scores_expanded = w.expand(S, -1)
    
    best_selected = selected.clone()
    best_total_scores = (selected * w).sum(dim=1)
    
    # 引入 Restart 机制的计数器
    no_improve_count = 0

    for i in range(steps):
        s_src = selected[:, src] 
        contrib = torch.zeros((S, N), device=device)
        val = s_src * w[:, src] 
        contrib.scatter_add_(1, dst.unsqueeze(0).expand(S, -1), val)
        
        raw_gain = w - contrib
        candidates = (~selected) & (raw_gain > 1e-5)
        
        if candidates.sum() == 0:
            no_improve_count += 1
        else:
            selected = selected | candidates
            selected = gpu_conflict_filter(selected, edge_index, scores_expanded)
            
            current_scores = (selected * w).sum(dim=1)
            improved = current_scores > best_total_scores
            
            if improved.any():
                mask_imp = improved.unsqueeze(1).expand(-1, N)
                best_selected = torch.where(mask_imp, selected, best_selected)
                best_total_scores = torch.where(improved, current_scores, best_total_scores)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Restart 机制
        if no_improve_count >= 20:
            perturb = torch.rand_like(selected.float()) < 0.05
            selected = selected ^ (perturb & selected)
            no_improve_count = 0

    return best_selected

# ============================================================
# 4. 迭代搜索
# ============================================================
@torch.no_grad()
def gpu_iterated_local_search(selected, edge_index, weights, iter_cycles=20):
    device = selected.device
    S, N = selected.shape
    
    current_best = gpu_local_search(selected, edge_index, weights, steps=20)
    current_score = (current_best * weights.unsqueeze(0)).sum(dim=1)
    
    best_global = current_best.clone()
    best_global_score = current_score.clone()
    
    perturb_rate = 0.15
    
    for i in range(iter_cycles):
        perturb_mask = (torch.rand((S, N), device=device) < perturb_rate) & current_best
        candidate = current_best.clone()
        candidate[perturb_mask] = False 
        
        candidate = gpu_local_search(candidate, edge_index, weights, steps=15)
        new_score = (candidate * weights.unsqueeze(0)).sum(dim=1)
        
        improved = new_score >= current_score 
        if improved.any():
            mask_imp = improved.unsqueeze(1).expand(-1, N)
            current_best = torch.where(mask_imp, candidate, current_best)
            current_score = torch.where(improved, new_score, current_score)
            
            global_imp = new_score > best_global_score
            if global_imp.any():
                mask_g = global_imp.unsqueeze(1).expand(-1, N)
                best_global = torch.where(mask_g, candidate, best_global)
                best_global_score = torch.where(global_imp, new_score, best_global_score)

    return best_global

# ============================================================
# 5. 主入口 (修复了维度错误)
# ============================================================
@torch.no_grad()
def sample_and_refine_for_graph(model, x, edge_index, weights, samples=64, device="cuda"):
    device = torch.device(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    weights = weights.to(device)
    
    deg = torch.zeros(x.size(0), device=device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
    
    model.eval()
    logits = model(x, edge_index).view(-1)
    
    masks_list = []

    # --- 策略 1: 多维确定性启发式 (循环执行，避免维度堆叠报错) ---
    # 分别运行 4 种贪心，确保算法保底
    
    heuristics_list = [
        weights / (deg + 1.0),           # 经典
        weights,                         # 唯权重
        weights / torch.sqrt(deg + 1.0), # 轻惩罚
        weights / (deg.pow(2) + 1.0)     # 重惩罚
    ]
    
    for h_score in heuristics_list:
        # 单次采样，确定性贪心
        raw_h = parallel_gumbel_sampling(h_score, edge_index, 1, temperature=1e-5)
        mask_h = gpu_conflict_filter(raw_h, edge_index, weights.unsqueeze(0))
        mask_h = gpu_local_search(mask_h, edge_index, weights, steps=20)
        masks_list.append(mask_h)

    # --- 策略 2: GNN 引导 ---
    n_gnn = int(samples * 0.4)
    if n_gnn > 0:
        raw_gnn = parallel_gumbel_sampling(logits, edge_index, n_gnn, temperature=0.5)
        scores_gnn = logits.unsqueeze(0).expand(n_gnn, -1)
        mask_gnn = gpu_conflict_filter(raw_gnn, edge_index, scores_gnn)
        mask_gnn = gpu_local_search(mask_gnn, edge_index, weights, steps=50)
        masks_list.append(mask_gnn)

    # --- 策略 3: 激进搜索 ---
    n_ils = max(0, samples - 4 - n_gnn)
    if n_ils > 0:
        raw_ils = parallel_gumbel_sampling(logits, edge_index, n_ils, temperature=1.2)
        scores_ils = logits + 0.3 * torch.log1p(weights)
        mask_ils = gpu_conflict_filter(raw_ils, edge_index, scores_ils.unsqueeze(0).expand(n_ils, -1))
        mask_ils = gpu_iterated_local_search(mask_ils, edge_index, weights, iter_cycles=20)
        masks_list.append(mask_ils)

    # --- 汇总 ---
    all_masks = torch.cat(masks_list, dim=0)
    final_values = (all_masks * weights.unsqueeze(0)).sum(dim=1)
    
    best_idx = torch.argmax(final_values)
    best_mask = all_masks[best_idx]
    best_score = float(final_values[best_idx])
    
    return best_mask.cpu(), best_score

# ============================================================
# 6. 辅助函数
# ============================================================
def restore_original_solution(new2old, preselected, chosen_mask):
    import numpy as np
    if isinstance(chosen_mask, torch.Tensor):
        chosen_mask = chosen_mask.numpy()
    chosen_mask = np.asarray(chosen_mask).astype(bool)
    chosen_reduced = np.where(chosen_mask)[0]
    mapped = [int(new2old[i]) for i in chosen_reduced]
    return sorted(set(mapped) | set(preselected))

def write_observation_plan(items_path, solution_indices, output_path):
    with open(items_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = ""
    start = 0
    if len(lines) > 0 and lines[0].strip().startswith("#"):
        header = lines[0].strip()
        start = 1
    selected = []
    tot_w = 0.0
    cnt = 0
    max_idx = len(lines) - start
    for idx in solution_indices:
        if 0 <= idx < max_idx:
            line = lines[idx + start]
            selected.append(line)
            cnt += 1
            try:
                parts = line.split(":")
                if len(parts) > 1: tot_w += float(parts[1].strip())
            except: pass
    if header:
        header = re.sub(r"obj \d+", f"obj {cnt}", header)
        header = re.sub(r"weight [\d.E+-]+", f"weight {tot_w:.8E}", header)
        header += "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        if header: f.write(header)
        f.writelines(selected)