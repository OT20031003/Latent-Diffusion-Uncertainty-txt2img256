import argparse
import os
import sys
import json
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
import cv2

# ==========================================
# 自分自身のディレクトリをパスに追加
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Face Utils Import ---
try:
    from face_utils import FaceParser
    FACE_PARSING_AVAILABLE = True
except ImportError as e:
    FACE_PARSING_AVAILABLE = False

# --- 外部ライブラリのインポート ---
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: 'lpips' not found. LPIPS metric will be skipped.")

try:
    from DISTS_pytorch import DISTS
    DISTS_AVAILABLE = True
except ImportError:
    DISTS_AVAILABLE = False
    print("Warning: 'DISTS_pytorch' not found. DISTS metric will be skipped.")

try:
    from facenet_pytorch import InceptionResnetV1
    IDLOSS_AVAILABLE = True
except ImportError:
    IDLOSS_AVAILABLE = False
    print("Warning: 'facenet-pytorch' not found. ID Loss will be skipped.")

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: 'pytorch-fid' not found. FID calculation will be skipped.")


def set_seed(seed):
    """再現性のためのシード値設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        print("missing keys:", u)
        print("unexpected keys:", m)
    model.cuda()
    model.eval()
    return model

def add_awgn_channel(latent, snr_db):
    """AWGNチャネルのシミュレーション"""
    sig_power = torch.mean(latent ** 2, dim=(1, 2, 3), keepdim=True)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(latent) * noise_std
    noisy_latent = latent + noise
    return noisy_latent

def calculate_psnr(img1, img2):
    """PSNR計算"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_id_loss(img1, img2, net):
    """ID Loss計算"""
    img1_in = F.interpolate((img1 * 2) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
    img2_in = F.interpolate((img2 * 2) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
    with torch.no_grad():
        emb1 = net(img1_in)
        emb2 = net(img2_in)
    cosine_sim = F.cosine_similarity(emb1, emb2)
    return (1.0 - cosine_sim).item()

def compute_structural_uncertainty(uncertainty_map, latents_pass1=None, alpha=0.3):
    """構造的不確実性の計算"""
    if uncertainty_map is None: return None
    kernel_size = 5
    sigma = 2.0
    channels = uncertainty_map.shape[1]
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*3.14159*variance)) * torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1).to(uncertainty_map.device)

    pad = (kernel_size - 1) // 2
    smoothed_unc = F.conv2d(uncertainty_map, gaussian_kernel, padding=pad, groups=channels)

    b_sz = smoothed_unc.shape[0]
    min_val = smoothed_unc.view(b_sz, -1).min(dim=1, keepdim=True)[0].view(b_sz, 1, 1, 1)
    max_val = smoothed_unc.view(b_sz, -1).max(dim=1, keepdim=True)[0].view(b_sz, 1, 1, 1)
    norm_unc = (smoothed_unc - min_val) / (max_val - min_val + 1e-8)

    noise = torch.rand_like(norm_unc)
    weighted_map = (1 - alpha) * norm_unc + alpha * noise
    return weighted_map

def compute_edge_map(images, target_hw):
    """エッジマップ計算"""
    gray = (images + 1.0) / 2.0
    gray = 0.299 * gray[:, 0] + 0.587 * gray[:, 1] + 0.114 * gray[:, 2]
    gray = gray.unsqueeze(1) 

    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=images.device, dtype=torch.float32).view(1, 1, 3, 3)
    k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=images.device, dtype=torch.float32).view(1, 1, 3, 3)

    gx = F.conv2d(gray, k_x, padding=1)
    gy = F.conv2d(gray, k_y, padding=1)

    magnitude = torch.sqrt(gx**2 + gy**2)
    edge_map = F.interpolate(magnitude, size=target_hw, mode='bilinear', align_corners=False)
    return edge_map

def create_retransmission_mask(uncertainty_map, rate):
    if uncertainty_map is None: return None
    spatial_unc = torch.mean(uncertainty_map, dim=1, keepdim=True)
    b, c, h, w = spatial_unc.shape
    mask = torch.zeros_like(spatial_unc)
    for i in range(b):
        flat = spatial_unc[i].flatten()
        k = int(flat.numel() * rate)
        if k > 0:
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            mask[i] = (spatial_unc[i] >= threshold).float()
    return mask

def create_hybrid_mask(uncertainty_map, edge_map, rate, alpha=0.7, beta=0.3):
    if uncertainty_map is None or edge_map is None: return None
    spatial_unc = torch.mean(uncertainty_map, dim=1, keepdim=True)
    
    def normalize(x):
        b_sz = x.shape[0]
        min_v = x.view(b_sz, -1).min(dim=1, keepdim=True)[0].view(b_sz, 1, 1, 1)
        max_v = x.view(b_sz, -1).max(dim=1, keepdim=True)[0].view(b_sz, 1, 1, 1)
        return (x - min_v) / (max_v - min_v + 1e-8)

    norm_unc = normalize(spatial_unc)
    norm_edge = normalize(edge_map)
    combined_score = alpha * norm_unc + beta * norm_edge

    b, c, h, w = combined_score.shape
    mask = torch.zeros_like(combined_score)
    for i in range(b):
        flat = combined_score[i].flatten()
        k = int(flat.numel() * rate)
        if k > 0:
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            mask[i] = (combined_score[i] >= threshold).float()
    return mask

def create_semantic_weighted_mask(binary_mask, parsing_map, retransmission_rate):
    if binary_mask.ndim == 3: mask_latent = np.mean(binary_mask, axis=0)
    else: mask_latent = binary_mask
        
    h_lat, w_lat = mask_latent.shape
    mask_latent = (mask_latent > 0.5).astype(np.float32)
    parsing_map_latent = cv2.resize(parsing_map, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)

    weights = {
        4: 6.0, 5: 6.0, 2: 5.0, 3: 5.0, 10: 5.0, 11: 5.0, 12: 5.0, 13: 5.0, 1: 2.0, 17: 0.8, 0: 0.1
    }
    default_weight = 0.5
    weight_map = np.full_like(mask_latent, default_weight)
    for cls_id, w in weights.items():
        weight_map[parsing_map_latent == cls_id] = w
        
    noise = np.random.rand(h_lat, w_lat) * 0.01
    weighted_score = mask_latent * weight_map + noise
    
    flat_scores = weighted_score.flatten()
    k = int(flat_scores.size * retransmission_rate)
    final_mask = np.zeros_like(mask_latent, dtype=np.float32)
    if k > 0:
        threshold_idx = flat_scores.size - k
        threshold = np.partition(flat_scores, threshold_idx)[threshold_idx]
        final_mask = (weighted_score >= threshold).astype(np.float32)
        final_mask = final_mask * mask_latent
    return final_mask

def create_semantic_only_mask(shape, parsing_map, retransmission_rate):
    h_lat, w_lat = shape
    parsing_map_latent = cv2.resize(parsing_map, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
    weights = {
        4: 6.0, 5: 6.0, 2: 5.0, 3: 5.0, 10: 5.0, 11: 5.0, 12: 5.0, 13: 5.0, 1: 2.0, 17: 0.8, 0: 0.1
    }
    default_weight = 0.5
    score_map = np.full((h_lat, w_lat), default_weight, dtype=np.float32)
    for cls_id, w in weights.items():
        score_map[parsing_map_latent == cls_id] = w
    
    noise = np.random.rand(h_lat, w_lat) * 0.01
    score_map += noise
    
    flat_scores = score_map.flatten()
    k = int(flat_scores.size * retransmission_rate)
    final_mask = np.zeros_like(score_map, dtype=np.float32)
    if k > 0:
        threshold_idx = flat_scores.size - k
        threshold = np.partition(flat_scores, threshold_idx)[threshold_idx]
        final_mask = (score_map >= threshold).astype(np.float32)
    return final_mask

def create_random_mask(shape, rate, device):
    b, c, h, w = shape
    mask = torch.zeros(shape, device=device)
    for i in range(b):
        rnd = torch.rand((1, h, w), device=device)
        flat = rnd.flatten()
        k = int(flat.numel() * rate)
        if k > 0:
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            mask[i] = (rnd >= threshold).float()
    return mask

def save_mask_tensor(mask, batch_files, batch_out_dir, prefix):
    for j, fname in enumerate(batch_files):
        m = mask[j, 0].cpu().numpy()
        m_img = (m * 255).astype(np.uint8)
        img = Image.fromarray(m_img, mode='L')
        save_path = os.path.join(batch_out_dir, str(j), f"{os.path.splitext(fname)[0]}_{prefix}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

def evaluate_and_save(x_rec, batch_input, batch_files, batch_out_dir, suffix, 
                      loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_save_dir=None):
    gt_01 = torch.clamp((batch_input + 1.0) / 2.0, 0.0, 1.0)
    rec_01 = torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)
    results = {}
    for j, fname in enumerate(batch_files):
        img_data = (rec_01[j].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        save_path = os.path.join(batch_out_dir, str(j), f"{os.path.splitext(fname)[0]}_{suffix}{os.path.splitext(fname)[1]}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        if fid_save_dir: img.save(os.path.join(fid_save_dir, fname))
        metrics = {"psnr": calculate_psnr(gt_01[j:j+1], rec_01[j:j+1]).item()}
        if loss_fn_lpips: metrics["lpips"] = loss_fn_lpips(gt_01[j:j+1]*2-1, rec_01[j:j+1]*2-1).item()
        if loss_fn_dists: metrics["dists"] = loss_fn_dists(gt_01[j:j+1], rec_01[j:j+1]).item()
        if loss_fn_id: metrics["id_loss"] = calculate_id_loss(gt_01[j:j+1], rec_01[j:j+1], loss_fn_id)
        results[fname] = metrics
    return results

def main():
    parser = argparse.ArgumentParser(description="DiffCom Retransmission Simulation (Text2Img)")
    parser.add_argument("--input_dir", type=str, default="input_dir")
    parser.add_argument("--output_dir", type=str, default="results_text2img/")
    parser.add_argument("--snr", type=float, default=-15.0)
    parser.add_argument("-r","--retransmission_rate", type=float, default=0.2)
    # Default Config & Checkpoint for Text2Img
    parser.add_argument("--config", type=str, default="models/ldm/text2img256/config.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/text2img256/model.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--face_model_path", type=str, default="models/face_parsing/79999_iter.pth")
    parser.add_argument("--struct_alpha", type=float, default=0.0)
    parser.add_argument("--hybrid_alpha", type=float, default=0.7)
    parser.add_argument("--hybrid_beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    
    # Text2Img用: scale (Unconditional Guidance Scale)
    # プロンプトが空の場合、条件付きと条件なしが等しくなるため、理論上scaleは影響しませんが
    # コードの整合性のために残します。
    parser.add_argument("--scale", type=float, default=5.0)

    available_methods = ["structural", "raw", "wacv", "semantic", "semantic_only", "random", "edge_rec", "edge_gt", "hybrid"]
    parser.add_argument("--target_methods", nargs='+', default=["all"], 
                        help=f"Select methods to run. Options: {', '.join(available_methods)} or 'all'.")
    
    args = parser.parse_args()

    # --- 通信シミュレーション用の強制設定 ---
    # 受信側は画像の内容（プロンプト）を知らないため、常に空文字列（無条件）として処理する
    print("\n[Simulation Mode] Prompt forced to empty string ('') for unconditional conditioning.")
    forced_prompt = "" 
    # -------------------------------------

    method_name_map = {
        "structural": "pass2_structural",
        "raw": "pass2_raw",
        "wacv": "pass2_wacv",
        "semantic": "pass2_semantic",
        "semantic_only": "pass2_semantic_only",
        "random": "pass2_random",
        "edge_rec": "pass2_edge_rec",
        "edge_gt": "pass2_edge_gt",
        "hybrid": "pass2_hybrid"
    }

    if "all" in args.target_methods:
        target_keys = set(method_name_map.values())
    else:
        target_keys = set()
        for m in args.target_methods:
            if m in method_name_map:
                target_keys.add(method_name_map[m])
            else:
                print(f"Warning: Unknown method '{m}' ignored.")
    
    print(f"Target Methods: {sorted(list(target_keys))}")

    set_seed(args.seed)
    
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    # FaceParserロード処理（Text2Imgでは顔以外も含むため、エラーで止まらないように注意）
    use_semantic = ("pass2_semantic" in target_keys) or ("pass2_semantic_only" in target_keys)
    face_parser = None
    if use_semantic:
        if FACE_PARSING_AVAILABLE and os.path.exists(args.face_model_path):
            print(f"Loading FaceParser for semantic weighting (Sender-Side)...")
            face_parser = FaceParser(args.face_model_path, device='cuda')
        else:
            print("Warning: Semantic method selected but dependencies missing or model not found. Skipping semantic methods.")
            target_keys.discard("pass2_semantic")
            target_keys.discard("pass2_semantic_only")

    loss_fn_lpips = lpips.LPIPS(net='alex').cuda().eval() if LPIPS_AVAILABLE else None
    loss_fn_dists = DISTS().cuda().eval() if DISTS_AVAILABLE else None
    loss_fn_id = InceptionResnetV1(pretrained='vggface2').cuda().eval() if IDLOSS_AVAILABLE else None

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    experiment_dir = os.path.join(args.output_dir, f"snr{args.snr}dB_rate{args.retransmission_rate}_alpha{args.struct_alpha}")
    os.makedirs(experiment_dir, exist_ok=True)
    fid_root = os.path.join(experiment_dir, "fid_images")
    
    eval_keys = ["gt", "pass1"] + sorted(list(target_keys))
    fid_dirs = {k: os.path.join(fid_root, k) for k in eval_keys}
    if FID_AVAILABLE: [os.makedirs(d, exist_ok=True) for d in fid_dirs.values()]
    
    all_results = {}
    json_path = os.path.join(experiment_dir, f"metrics.json")
    calc_wacv = "pass2_wacv" in target_keys

    try:
        with torch.no_grad():
            for i in range(0, len(image_files), args.batch_size):
                batch_idx = i // args.batch_size
                current_batch_seed = args.seed + batch_idx
                set_seed(current_batch_seed)

                batch_files = image_files[i : i + args.batch_size]
                actual_bs = len(batch_files)
                batch_out_dir = os.path.join(experiment_dir, f"batch{batch_idx}")
                os.makedirs(batch_out_dir, exist_ok=True)

                batch_tensors = []
                for fname in batch_files:
                    img = Image.open(os.path.join(args.input_dir, fname)).convert("RGB").resize((256, 256), Image.BICUBIC)
                    batch_tensors.append(torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
                if actual_bs < args.batch_size: batch_tensors.extend([batch_tensors[-1]] * (args.batch_size - actual_bs))
                batch_input = torch.stack(batch_tensors).cuda()

                # Text2Img用 Conditioning 生成 (常に空文字列)
                c = None
                uc = None
                if hasattr(model, "get_learned_conditioning"):
                    prompts = [forced_prompt] * args.batch_size
                    c = model.get_learned_conditioning(prompts)
                    # Unconditional Conditioningも同じ空文字列
                    uc = model.get_learned_conditioning([""] * args.batch_size)

                # GT & First Stage
                valid_batch_input = batch_input[:actual_bs]
                for j, fname in enumerate(batch_files):
                    gt_img = (torch.clamp((valid_batch_input[j] + 1.0) / 2.0, 0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    save_path = os.path.join(batch_out_dir, str(j), f"{os.path.splitext(fname)[0]}_gt{os.path.splitext(fname)[1]}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    Image.fromarray(gt_img).save(save_path)
                    if FID_AVAILABLE: Image.fromarray(gt_img).save(os.path.join(fid_dirs["gt"], fname))

                z0 = model.encode_first_stage(batch_input)
                z0 = z0.mode() if not isinstance(z0, torch.Tensor) else z0
                z_received_pass1 = add_awgn_channel(z0, args.snr)

                # Pass 1 Sampling
                samples_pass1, history_pass1 = sampler.sample_awgn(
                    S=args.ddim_steps, 
                    batch_size=args.batch_size, 
                    shape=z0.shape[1:], 
                    noisy_latent=z_received_pass1, 
                    snr_db=args.snr, 
                    eta=0.0,
                    calc_wacv_uncertainty=calc_wacv,
                    conditioning=c,  # 強制的に空文字列
                    unconditional_guidance_scale=args.scale,
                    unconditional_conditioning=uc
                )
                
                temporal_uncertainty_map = None
                wacv_uncertainty_map = None
                if history_pass1:
                    for h_name, h_map in history_pass1:
                        if h_name == "temporal": temporal_uncertainty_map = h_map
                        elif h_name == "wacv": wacv_uncertainty_map = h_map
                
                uncertainty_map_for_sem = temporal_uncertainty_map

                x_rec_pass1 = model.decode_first_stage(samples_pass1)
                pass1_results = evaluate_and_save(x_rec_pass1[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass1", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass1"] if FID_AVAILABLE else None)

                # === エッジマップ計算 ===
                need_edge = any(k in target_keys for k in ["pass2_edge_rec", "pass2_hybrid"])
                need_edge_gt = any(k in target_keys for k in ["pass2_edge_gt", "pass2_hybrid"])
                
                edge_map_rec = None
                if need_edge:
                    edge_map_rec = compute_edge_map(x_rec_pass1, (z0.shape[2], z0.shape[3]))
                
                edge_map_gt = None
                if need_edge_gt:
                    edge_map_gt = compute_edge_map(batch_input, (z0.shape[2], z0.shape[3]))

                # === Structural Uncertainty Map ===
                need_struct = any(k in target_keys for k in ["pass2_structural", "pass2_hybrid"])
                map_struct = None
                if need_struct:
                    map_struct = compute_structural_uncertainty(temporal_uncertainty_map, samples_pass1, alpha=args.struct_alpha)

                method_candidates = [
                    ("pass2_structural", lambda: create_retransmission_mask(map_struct, args.retransmission_rate)),
                    ("pass2_raw", lambda: create_retransmission_mask(temporal_uncertainty_map, args.retransmission_rate)),
                    ("pass2_wacv", lambda: create_retransmission_mask(wacv_uncertainty_map, args.retransmission_rate)),
                    ("pass2_edge_rec", lambda: create_retransmission_mask(edge_map_rec, args.retransmission_rate)),
                    ("pass2_edge_gt", lambda: create_retransmission_mask(edge_map_gt, args.retransmission_rate)),
                    ("pass2_hybrid", lambda: create_hybrid_mask(map_struct, edge_map_gt, args.retransmission_rate, args.hybrid_alpha, args.hybrid_beta))
                ]
                
                active_methods = [m for m in method_candidates if m[0] in target_keys]

                results_batch = {"pass1": pass1_results}
                
                for key, mask_gen in active_methods:
                    if args.retransmission_rate > 0.0:
                        set_seed(current_batch_seed + 1000)
                        mask = mask_gen()
                        if mask is None: continue
                        
                        # === ファイル名調整 (Hybridの場合パラメータ付与) ===
                        file_suffix = key
                        mask_prefix = f"mask_{key.replace('pass2_', '')}"

                        if key == "pass2_hybrid":
                            param_str = f"_a{args.hybrid_alpha}_b{args.hybrid_beta}"
                            file_suffix += param_str
                            mask_prefix += param_str
                        # ========================================================
                        
                        save_mask_tensor(mask, batch_files, batch_out_dir, mask_prefix)
                        
                        s, _ = sampler.sample_inpainting_awgn(
                            S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                            noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask, x0=z0, eta=0.0,
                            conditioning=c, # 空文字列
                            unconditional_guidance_scale=args.scale,
                            unconditional_conditioning=uc
                        )
                        results_batch[key] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, file_suffix, loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs[key] if FID_AVAILABLE else None)

                # =================================================================
                # Semantic Weighted Method
                # =================================================================
                if "pass2_semantic" in target_keys and args.retransmission_rate > 0.0 and uncertainty_map_for_sem is not None and face_parser:
                    set_seed(current_batch_seed + 1000)
                    mask_sem_list = []
                    for j in range(actual_bs):
                        candidate_rate = min(1.0, args.retransmission_rate * 3.0)
                        u_map = uncertainty_map_for_sem[j:j+1]
                        mask_rec = create_retransmission_mask(u_map, candidate_rate) 
                        
                        p = face_parser.get_parsing_map(valid_batch_input[j:j+1])
                        m_np = create_semantic_weighted_mask(
                            mask_rec.cpu().numpy().squeeze(), 
                            p, 
                            args.retransmission_rate 
                        )
                        m = torch.from_numpy(m_np).unsqueeze(0).unsqueeze(0).float()
                        mask_sem_list.append(m)
                    
                    mask_sem = torch.cat(mask_sem_list, dim=0)
                    save_mask_tensor(mask_sem, batch_files, batch_out_dir, "mask_semantic_weighted")
                    
                    if mask_sem.shape[0] < args.batch_size:
                        mask_sem = torch.cat([mask_sem, mask_sem[-1:].repeat(args.batch_size-actual_bs, 1, 1, 1)], dim=0)
                    mask_sem = mask_sem.cuda()
                        
                    s, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                        noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask_sem, x0=z0, eta=0.0,
                        conditioning=c, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                    )
                    results_batch["pass2_semantic"] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_semantic", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass2_semantic"] if FID_AVAILABLE else None)

                # =================================================================
                # Semantic ONLY Method
                # =================================================================
                if "pass2_semantic_only" in target_keys and args.retransmission_rate > 0.0 and face_parser:
                    set_seed(current_batch_seed + 1000)
                    mask_sem_list = []
                    for j in range(actual_bs):
                        p = face_parser.get_parsing_map(valid_batch_input[j:j+1])
                        m_np = create_semantic_only_mask(
                            (z0.shape[2], z0.shape[3]), # (H, W)
                            p, 
                            args.retransmission_rate
                        )
                        m = torch.from_numpy(m_np).unsqueeze(0).unsqueeze(0).float()
                        mask_sem_list.append(m)
                    
                    mask_sem = torch.cat(mask_sem_list, dim=0)
                    save_mask_tensor(mask_sem, batch_files, batch_out_dir, "mask_semantic_only")
                    
                    if mask_sem.shape[0] < args.batch_size:
                        mask_sem = torch.cat([mask_sem, mask_sem[-1:].repeat(args.batch_size-actual_bs, 1, 1, 1)], dim=0)
                    mask_sem = mask_sem.cuda()
                        
                    s, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                        noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask_sem, x0=z0, eta=0.0,
                        conditioning=c, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                    )
                    results_batch["pass2_semantic_only"] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_semantic_only", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass2_semantic_only"] if FID_AVAILABLE else None)

                if "pass2_random" in target_keys and args.retransmission_rate > 0.0:
                    set_seed(current_batch_seed + 1000)
                    mask_r = create_random_mask((z0.shape[0], 1, z0.shape[2], z0.shape[3]), args.retransmission_rate, z0.device)
                    save_mask_tensor(mask_r, batch_files, batch_out_dir, "mask_rand")
                    s, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                        noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask_r, x0=z0, eta=0.0,
                        conditioning=c, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                    )
                    results_batch["pass2_random"] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_random", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass2_random"] if FID_AVAILABLE else None)

                for f in batch_files:
                    all_results[f] = {k: {"metrics": results_batch[k].get(f)} for k in results_batch.keys() if results_batch.get(k)}
                gc.collect(); torch.cuda.empty_cache()

        method_labels = {
            "pass1": "Pass 1 (Initial)", 
            "pass2_structural": "Pass 2 (Struct/Smooth)", 
            "pass2_raw": "Pass 2 (Raw/Temporal)", 
            "pass2_wacv": "Pass 2 (WACV/ScoreVar)",
            "pass2_semantic": "Pass 2 (Semantic/Weighted)", 
            "pass2_semantic_only": "Pass 2 (Semantic/Only)",
            "pass2_random": "Pass 2 (Random)",
            "pass2_edge_rec": "Pass 2 (Edge/Rec)",
            "pass2_edge_gt": "Pass 2 (Edge/GT-Oracle)",
            "pass2_hybrid": "Pass 2 (Hybrid U+E)"
        }
        
        executed_methods = [m for m in method_labels.keys() if m == "pass1" or m in target_keys]

        metric_keys = ["psnr", "lpips", "dists", "id_loss"]
        averages = {m: {met: np.mean([all_results[f][m]["metrics"][met] for f in all_results if all_results[f].get(m) and all_results[f][m].get("metrics")]) for met in metric_keys} for m in executed_methods}
        
        fids = {}
        if FID_AVAILABLE:
            for m in executed_methods:
                if m in fid_dirs and os.path.exists(fid_dirs[m]) and len(os.listdir(fid_dirs[m])) > 0:
                    try:
                        fids[m] = fid_score.calculate_fid_given_paths([fid_dirs["gt"], fid_dirs[m]], 50, 'cuda', 2048)
                    except Exception as e:
                        fids[m] = None
                else:
                    fids[m] = None

        best_scores = {}
        for met in metric_keys:
             valid_values = [averages[m][met] for m in executed_methods if m in averages and not np.isnan(averages[m][met])]
             if valid_values:
                 best_scores[met] = (max if met == "psnr" else min)(valid_values)
             else:
                 best_scores[met] = -1

        valid_fids = [v for v in fids.values() if v is not None]
        best_fid = min(valid_fids) if valid_fids else None

        GREEN, BOLD, RESET = "\033[92m", "\033[1m", "\033[0m"
        print("\n" + "="*120 + f"\n  SUMMARY (N={len(all_results)})\n" + "="*120)
        print(f"{'Method':<30} | {'PSNR':<10} | {'LPIPS':<10} | {'DISTS':<10} | {'ID Loss':<10} | {'FID':<10}")
        print("-" * 120)
        for m in executed_methods:
            name = method_labels[m]
            if m not in averages: continue
            row = f"{name:<30}"
            for met in metric_keys:
                val = averages[m][met]
                s = f"{val:.4f}"
                is_best = abs(val - best_scores[met]) < 1e-7 if best_scores[met] != -1 else False
                row += f" | {GREEN}{BOLD}*{s}*{RESET}" if is_best else f" | {s:<10}"
            f_val = fids.get(m)
            if f_val is not None:
                row += f" | {GREEN}{BOLD}*{f_val:.4f}*{RESET}" if best_fid and abs(f_val - best_fid) < 1e-7 else f" | {f_val:.4f}  "
            else: row += f" | {'N/A':<10}"
            print(row)
        print("-" * 120)
        with open(json_path, 'w') as f: json.dump({"summary": {"averages": averages, "fid": fids}}, f, indent=4)
    except Exception as e: print(f"Error: {e}"); raise e

if __name__ == "__main__": main()