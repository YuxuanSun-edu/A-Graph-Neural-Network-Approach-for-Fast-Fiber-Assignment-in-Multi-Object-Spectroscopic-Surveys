# main_gnn_solver.py 
import os
import argparse
import torch
import numpy as np

from model import MISScoreGNN
from gnn_sampler_gpu import (
    sample_and_refine_for_graph,
    restore_original_solution,
    write_observation_plan,
)

# ------------------------------------------------------------
# 加载模型
# ------------------------------------------------------------
def load_gnn_model(model_path, in_dim, device):
    # 使用 weights_only=True 加载，更安全且符合新版 PyTorch 规范
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except:
        ckpt = torch.load(model_path, map_location=device)

    hidden_dim = ckpt.get("hidden_dim", 64)
    num_layers = ckpt.get("num_layers", 3)
    dropout = ckpt.get("dropout", 0.2)

    model = MISScoreGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    state = ckpt["model_state_dict"]
    
    # 兼容性加载：自动过滤维度不匹配的参数
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    if len(pretrained_dict) < len(state):
        print(f"[Warning] 模型维度不匹配，仅加载了 {len(pretrained_dict)}/{len(state)} 层参数")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------
# 分批求解逻辑
# ------------------------------------------------------------
def run_sampling_in_batches(model, x, edge_index, weights, total_samples, batch_size, device="cuda"):
    best_global_mask = None
    best_global_score = -float('inf')

    # 计算需要的批次
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    # 显存预热/清理
    torch.cuda.empty_cache()

    for i in range(num_batches):
        current_samples = min(batch_size, total_samples - i * batch_size)
        try:
            mask, score = sample_and_refine_for_graph(
                model=model,
                x=x,
                edge_index=edge_index,
                weights=weights,
                samples=current_samples,
                device=device,
            )
            
            if score > best_global_score:
                best_global_score = score
                best_global_mask = mask
                
            # 及时释放显存
            del mask
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Warning] 显存不足 (Batch Size {batch_size})，尝试减半重试...")
                torch.cuda.empty_cache()
                # 简单的错误恢复：如果真的爆显存了（极小概率），递归调用自己，但 batch_size 减半
                if batch_size > 16:
                    return run_sampling_in_batches(model, x, edge_index, weights, total_samples, batch_size // 2, device)
                else:
                    raise e
            else:
                raise e

    if best_global_mask is None:
        return torch.zeros(weights.size(0), dtype=torch.bool).cpu(), 0.0

    return best_global_mask, best_global_score


# ------------------------------------------------------------
# 核心求解入口
# ------------------------------------------------------------
def solve_one_npz(npz_path, items_path, model_or_path, outfile, samples, device):
    # 1. 加载数据
    data = np.load(npz_path)
    
    x = torch.from_numpy(data["x"]).float().to(device)
    edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
    
    if 'weights' in data:
        weights = torch.from_numpy(data['weights']).float().to(device)
    else:
        weights = x[:, 0]

    new2old = data["new2old"]
    preselected = data["preselected"]

    # 2. 加载模型
    if isinstance(model_or_path, str):
        in_dim = x.size(1)
        model = load_gnn_model(model_or_path, in_dim=in_dim, device=device)
    else:
        model = model_or_path

    # 3. 性能配置 
    N = x.size(0)
    
    effective_samples = samples 
    
    if N > 100000:
        # 只有当节点数超过 10万 这种超大图时，才稍微保守一点
        SAFE_BATCH_SIZE = 1024
    else:
        # 对于 2~3万节点的 LAMOST 数据，直接拉满
        SAFE_BATCH_SIZE = 2048 

    # 4. 运行采样
    best_mask, best_score = run_sampling_in_batches(
        model=model,
        x=x,
        edge_index=edge_index,
        weights=weights,
        total_samples=effective_samples,
        batch_size=SAFE_BATCH_SIZE,
        device=device
    )

    # 5. 还原输出
    full_idx = restore_original_solution(
        new2old=new2old,
        preselected=preselected,
        chosen_mask=best_mask.cpu().numpy(),
    )
    write_observation_plan(items_path, full_idx, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--outfile", required=True)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    solve_one_npz(
        npz_path=args.npz,
        items_path=args.items,
        model_or_path=args.model,
        outfile=args.outfile,
        samples=args.samples,
        device=device,
    )

if __name__ == "__main__":
    main()