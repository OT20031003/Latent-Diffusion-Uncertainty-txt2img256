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

# --- Mask2Former Import (General Segmentation) ---
try:
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
    MASK2FORMER_AVAILABLE = True
except ImportError:
    MASK2FORMER_AVAILABLE = False
    print("Warning: 'transformers' (Mask2Former) not found. General segmentation will be skipped.")

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

# --- CLIP Import ---
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: 'transformers' (CLIP) not found. CLIP Score will be skipped.")


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

def calculate_clip_similarity(img1, img2, model, processor):
    """CLIP Image-to-Image Similarity 計算"""
    i1 = (img1.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    i2 = (img2.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    image1_pil = Image.fromarray(i1)
    image2_pil = Image.fromarray(i2)

    inputs = processor(images=[image1_pil, image2_pil], return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    similarity = (outputs[0] @ outputs[1].T).item()
    return similarity

def compute_structural_uncertainty(uncertainty_map, latents_pass1=None, alpha=0.3):
    """構造的不確実性の計算 (Baseline)"""
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

# =========================================================
#  新手法: Segment-based Feedback (SBF)
# =========================================================
def create_segment_feedback_mask(feedback_mask_binary, segmentation_map, rate):
    """
    【SBF】セグメントごとの不確実性平均に基づく再送マスク生成
    - feedback_mask_binary: 受信側からのFB (0 or 1) [B, 1, H_lat, W_lat]
    - segmentation_map: 送信側のセグメンテーション (Pixel Level) [B, H_img, W_img] (numpy or tensor)
    - rate: 再送率 (Budget)
    """
    if feedback_mask_binary is None or segmentation_map is None:
        return None

    b, _, h_lat, w_lat = feedback_mask_binary.shape
    device = feedback_mask_binary.device
    final_mask = torch.zeros_like(feedback_mask_binary)
    
    # ピクセル数予算
    total_pixels = h_lat * w_lat
    budget = int(total_pixels * rate)
    if budget == 0:
        return final_mask

    for i in range(b):
        # 1. セグメンテーションマップを潜在空間サイズにリサイズ
        # segmentation_map[i] が TensorかNumpyかを確認して統一
        seg = segmentation_map[i]
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        
        # リサイズ (Nearest Neighbor)
        seg_lat = cv2.resize(seg, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
        
        # Feedback Mask (0 or 1)
        f_mask = feedback_mask_binary[i, 0].cpu().numpy()
        
        # ユニークなクラスIDを取得 (背景含む)
        unique_ids = np.unique(seg_lat)
        
        segment_stats = []
        for class_id in unique_ids:
            # マスク領域
            mask_area = (seg_lat == class_id)
            size = np.sum(mask_area)
            if size == 0: continue
            
            # 領域内のFB(1)の数
            ones_count = np.sum(f_mask[mask_area])
            
            # 平均不確実性 (1の割合)
            avg_uncertainty = ones_count / size
            
            segment_stats.append({
                'id': class_id,
                'avg': avg_uncertainty,
                'mask': mask_area,
                'size': size
            })
            
        # 2. 平均不確実性が高い順にソート
        segment_stats.sort(key=lambda x: x['avg'], reverse=True)
        
        # 3. 予算配分
        current_usage = 0
        batch_mask = np.zeros((h_lat, w_lat), dtype=np.float32)
        
        for stat in segment_stats:
            # 平均スコアが0 (FBで誰も再送を希望していない) ならスキップするか検討
            # ここでは「受信側が少しでも不安なら」拾うため、0より大きければ候補
            if stat['avg'] <= 0.0:
                continue

            if current_usage + stat['size'] <= budget:
                # まるごと追加
                batch_mask[stat['mask']] = 1.0
                current_usage += stat['size']
            else:
                # 予算オーバーする場合の Pruning (間引き)
                remaining = budget - current_usage
                if remaining > 0:
                    # このセグメント内で、元のFBが1だった場所を優先的に採用
                    mask_area = stat['mask']
                    overlap = (mask_area & (f_mask > 0.5)) # FBで1だった場所
                    
                    overlap_indices = np.argwhere(overlap)
                    non_overlap_indices = np.argwhere(mask_area & (~(f_mask > 0.5)))
                    
                    # 優先順位: FBが1の場所 -> FBが0の場所
                    candidates = []
                    if len(overlap_indices) > 0:
                        candidates.extend([tuple(x) for x in overlap_indices])
                    if len(non_overlap_indices) > 0:
                        candidates.extend([tuple(x) for x in non_overlap_indices])
                        
                    # 詰める
                    for r, c in candidates[:remaining]:
                        batch_mask[r, c] = 1.0
                    
                current_usage = budget # 満タン
                break # 終了
        
        final_mask[i, 0] = torch.from_numpy(batch_mask).to(device)

    return final_mask

def create_hybrid_mask(uncertainty_map, edge_map, rate, alpha=0.7, beta=0.3):
    """既存のHybrid手法"""
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

def create_smart_hybrid_mask(uncertainty_mask_binary, edge_map, rate, kernel_size=3):
    """Smart Hybrid"""
    if uncertainty_mask_binary is None or edge_map is None: return None
    
    def normalize(x):
        b_sz = x.shape[0]
        min_v = x.view(b_sz, -1).min(dim=1, keepdim=True)[0].view(b_sz, 1, 1, 1)
        max_v = x.view(b_sz, -1).max(dim=1, keepdim=True)[0].view(b_sz, 1, 1, 1)
        return (x - min_v) / (max_v - min_v + 1e-8)

    norm_edge = normalize(edge_map)
    score = uncertainty_mask_binary * norm_edge
    padding = kernel_size // 2
    dilated_score = F.max_pool2d(score, kernel_size=kernel_size, stride=1, padding=padding)
    
    b, c, h, w = dilated_score.shape
    mask = torch.zeros_like(dilated_score)
    for i in range(b):
        flat = dilated_score[i].flatten()
        k = int(flat.numel() * rate)
        if k > 0:
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            mask[i] = (dilated_score[i] >= threshold).float()
    return mask

def create_oracle_error_mask(latent_rec, latent_gt, rate):
    """Oracle Error Mask"""
    diff = torch.abs(latent_rec - latent_gt)
    error_map = torch.mean(diff, dim=1, keepdim=True)
    b, c, h, w = error_map.shape
    mask = torch.zeros_like(error_map)
    for i in range(b):
        flat = error_map[i].flatten()
        k = int(flat.numel() * rate)
        if k > 0:
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            mask[i] = (error_map[i] >= threshold).float()
    return mask

def create_semantic_weighted_mask(binary_mask, parsing_map, retransmission_rate):
    """(Existing) Semantic Weighted"""
    if binary_mask.ndim == 3: mask_latent = np.mean(binary_mask, axis=0)
    else: mask_latent = binary_mask
        
    h_lat, w_lat = mask_latent.shape
    mask_latent = (mask_latent > 0.5).astype(np.float32)
    parsing_map_latent = cv2.resize(parsing_map, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)

    weights = {4: 6.0, 5: 6.0, 2: 5.0, 3: 5.0, 10: 5.0, 11: 5.0, 12: 5.0, 13: 5.0, 1: 2.0, 17: 0.8, 0: 0.1}
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
    """(Existing) Semantic Only"""
    h_lat, w_lat = shape
    parsing_map_latent = cv2.resize(parsing_map, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
    weights = {4: 6.0, 5: 6.0, 2: 5.0, 3: 5.0, 10: 5.0, 11: 5.0, 12: 5.0, 13: 5.0, 1: 2.0, 17: 0.8, 0: 0.1}
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
    """Random Mask"""
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
                      loss_fn_lpips, loss_fn_dists, loss_fn_id, 
                      clip_scorer=None, fid_save_dir=None):
    gt_01 = torch.clamp((batch_input + 1.0) / 2.0, 0.0, 1.0)
    rec_01 = torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)
    results = {}
    
    model_clip, processor_clip = None, None
    if clip_scorer:
        model_clip, processor_clip = clip_scorer

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
        if model_clip and processor_clip:
            metrics["clip"] = calculate_clip_similarity(gt_01[j:j+1], rec_01[j:j+1], model_clip, processor_clip)

        results[fname] = metrics
    return results

def main():
    global CLIP_AVAILABLE
    parser = argparse.ArgumentParser(description="DiffCom Retransmission Simulation (Text2Img)")
    parser.add_argument("--input_dir", type=str, default="input_dir")
    parser.add_argument("--output_dir", type=str, default="results_text2img/")
    parser.add_argument("--snr", type=float, default=-3.0)
    parser.add_argument("-r","--retransmission_rate", type=float, default=0.2)
    parser.add_argument("--config", type=str, default="models/ldm/text2img256/config.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/text2img256/model.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--face_model_path", type=str, default="models/face_parsing/79999_iter.pth")
    parser.add_argument("--mask2former_model", type=str, default="facebook/mask2former-swin-base-coco-panoptic")
    parser.add_argument("--struct_alpha", type=float, default=0.0)
    parser.add_argument("--hybrid_alpha", type=float, default=0.7)
    parser.add_argument("--hybrid_beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=5.0)

    # "sbf" (Segment Based Feedback) を追加
    available_methods = ["structural", "raw", "wacv", "semantic", "semantic_only", "random", 
                         "edge_rec", "edge_gt", "hybrid", "smart_hybrid", "oracle", "sbf"]
    parser.add_argument("--target_methods", nargs='+', default=["all"], 
                        help=f"Select methods. Options: {', '.join(available_methods)}")
    
    args = parser.parse_args()

    print("\n[Simulation Mode] Prompt forced to empty string ('') for unconditional conditioning.")
    forced_prompt = "" 

    method_name_map = {
        "structural": "pass2_structural",
        "raw": "pass2_raw",
        "wacv": "pass2_wacv",
        "semantic": "pass2_semantic",
        "semantic_only": "pass2_semantic_only",
        "random": "pass2_random",
        "edge_rec": "pass2_edge_rec",
        "edge_gt": "pass2_edge_gt",
        "hybrid": "pass2_hybrid",
        "smart_hybrid": "pass2_smart_hybrid",
        "oracle": "pass2_oracle",
        "sbf": "pass2_sbf"  # 新規追加
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

    # --- Segmentation Model Loading ---
    face_parser = None
    mask2former_processor = None
    mask2former_model = None

    # FaceParser (Fallback or Legacy)
    if "pass2_semantic" in target_keys or "pass2_semantic_only" in target_keys:
        if FACE_PARSING_AVAILABLE and os.path.exists(args.face_model_path):
            print(f"Loading FaceParser for semantic weighting...")
            face_parser = FaceParser(args.face_model_path, device='cuda')

    # Mask2Former (For SBF)
    if "pass2_sbf" in target_keys:
        if MASK2FORMER_AVAILABLE:
            try:
                print(f"Loading Mask2Former for SBF ({args.mask2former_model})...")
                mask2former_processor = Mask2FormerImageProcessor.from_pretrained(args.mask2former_model)
                mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(args.mask2former_model).cuda()
                mask2former_model.eval()
            except Exception as e:
                print(f"Failed to load Mask2Former: {e}. SBF will try to use FaceParser as fallback.")
        else:
            print("Mask2Former not available. SBF will try to use FaceParser as fallback.")

    # Metrics
    loss_fn_lpips = lpips.LPIPS(net='alex').cuda().eval() if LPIPS_AVAILABLE else None
    loss_fn_dists = DISTS().cuda().eval() if DISTS_AVAILABLE else None
    loss_fn_id = InceptionResnetV1(pretrained='vggface2').cuda().eval() if IDLOSS_AVAILABLE else None

    clip_scorer = None
    if CLIP_AVAILABLE:
        print("Loading CLIP Model (openai/clip-vit-base-patch32)...")
        try:
            local_clip_path = "/mnt/d/ai_models/clip-vit-base-patch32"
            print(f"Loading CLIP from: {local_clip_path}")
            clip_model = CLIPModel.from_pretrained(local_clip_path).cuda().eval()
            clip_processor = CLIPProcessor.from_pretrained(local_clip_path)
            clip_scorer = (clip_model, clip_processor)
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            CLIP_AVAILABLE = False


    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    experiment_dir = os.path.join(args.output_dir, f"snr{args.snr}dB_rate{args.retransmission_rate}_alpha{args.struct_alpha}")
    os.makedirs(experiment_dir, exist_ok=True)
    fid_root = os.path.join(experiment_dir, "fid_images")
    
    eval_keys = ["gt", "pass1"] + sorted(list(target_keys))
    fid_dirs = {k: os.path.join(fid_root, k) for k in eval_keys}
    if FID_AVAILABLE: [os.makedirs(d, exist_ok=True) for d in fid_dirs.values()]
    
    all_results = {}
    json_filename = f"metrics_snr{args.snr}_rate{args.retransmission_rate}_alpha{args.struct_alpha}.json"
    json_path = os.path.join(experiment_dir, json_filename)
    
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

                c = None
                uc = None
                if hasattr(model, "get_learned_conditioning"):
                    prompts = [forced_prompt] * args.batch_size
                    c = model.get_learned_conditioning(prompts)
                    uc = model.get_learned_conditioning([""] * args.batch_size)

                # GT Save
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
                    S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                    noisy_latent=z_received_pass1, snr_db=args.snr, eta=0.0,
                    calc_wacv_uncertainty=calc_wacv, conditioning=c,
                    unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                )
                
                temporal_uncertainty_map = None
                wacv_uncertainty_map = None
                if history_pass1:
                    for h_name, h_map in history_pass1:
                        if h_name == "temporal": temporal_uncertainty_map = h_map
                        elif h_name == "wacv": wacv_uncertainty_map = h_map
                
                uncertainty_map_for_sem = temporal_uncertainty_map
                x_rec_pass1 = model.decode_first_stage(samples_pass1)
                
                pass1_results = evaluate_and_save(
                    x_rec_pass1[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass1", 
                    loss_fn_lpips, loss_fn_dists, loss_fn_id, clip_scorer, 
                    fid_dirs["pass1"] if FID_AVAILABLE else None
                )

                # Maps Preparation
                need_edge = any(k in target_keys for k in ["pass2_edge_rec", "pass2_hybrid"])
                need_edge_gt = any(k in target_keys for k in ["pass2_edge_gt", "pass2_hybrid", "pass2_smart_hybrid"])
                edge_map_rec = compute_edge_map(x_rec_pass1, (z0.shape[2], z0.shape[3])) if need_edge else None
                edge_map_gt = compute_edge_map(batch_input, (z0.shape[2], z0.shape[3])) if need_edge_gt else None

                need_struct = any(k in target_keys for k in ["pass2_structural", "pass2_hybrid", "pass2_smart_hybrid"])
                map_struct = compute_structural_uncertainty(temporal_uncertainty_map, samples_pass1, alpha=args.struct_alpha) if need_struct else None

                # Base masks for hybrids
                smart_hybrid_feedback_mask = None
                if "pass2_smart_hybrid" in target_keys and map_struct is not None:
                    candidate_rate = min(1.0, args.retransmission_rate * 3.0)
                    smart_hybrid_feedback_mask = create_retransmission_mask(map_struct, candidate_rate)

                # Standard Methods
                method_candidates = [
                    ("pass2_structural", lambda: create_retransmission_mask(map_struct, args.retransmission_rate)),
                    ("pass2_raw", lambda: create_retransmission_mask(temporal_uncertainty_map, args.retransmission_rate)),
                    ("pass2_wacv", lambda: create_retransmission_mask(wacv_uncertainty_map, args.retransmission_rate)),
                    ("pass2_edge_rec", lambda: create_retransmission_mask(edge_map_rec, args.retransmission_rate)),
                    ("pass2_edge_gt", lambda: create_retransmission_mask(edge_map_gt, args.retransmission_rate)),
                    ("pass2_hybrid", lambda: create_hybrid_mask(map_struct, edge_map_gt, args.retransmission_rate, args.hybrid_alpha, args.hybrid_beta)),
                    ("pass2_smart_hybrid", lambda: create_smart_hybrid_mask(smart_hybrid_feedback_mask, edge_map_gt, args.retransmission_rate)),
                    ("pass2_oracle", lambda: create_oracle_error_mask(samples_pass1, z0, args.retransmission_rate))
                ]
                
                results_batch = {"pass1": pass1_results}
                
                # --- Run Standard Methods ---
                for key, mask_gen in method_candidates:
                    if key in target_keys and args.retransmission_rate > 0.0:
                        set_seed(current_batch_seed + 1000)
                        mask = mask_gen()
                        if mask is None: continue
                        
                        file_suffix = key
                        mask_prefix = f"mask_{key.replace('pass2_', '')}"
                        if key == "pass2_hybrid":
                            param_str = f"_a{args.hybrid_alpha}_b{args.hybrid_beta}"
                            file_suffix += param_str
                            mask_prefix += param_str
                        
                        save_mask_tensor(mask, batch_files, batch_out_dir, mask_prefix)
                        s, _ = sampler.sample_inpainting_awgn(
                            S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                            noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask, x0=z0, eta=0.0,
                            conditioning=c, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                        )
                        results_batch[key] = evaluate_and_save(
                            model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, file_suffix, 
                            loss_fn_lpips, loss_fn_dists, loss_fn_id, clip_scorer, fid_dirs[key] if FID_AVAILABLE else None
                        )

                # =================================================================
                # 【新規追加】 SBF: Segment-based Feedback Retransmission
                # =================================================================
                if "pass2_sbf" in target_keys and args.retransmission_rate > 0.0 and uncertainty_map_for_sem is not None:
                    set_seed(current_batch_seed + 1000)
                    
                    # 1. 送信側セグメンテーションマップ作成 (Mask2Former > FaceParser)
                    seg_maps = []
                    if mask2former_model and mask2former_processor:
                        # Mask2Former Inference
                        # 画像をPILにしてProcessorへ
                        pil_images = []
                        for b_idx in range(actual_bs):
                            # [-1,1] -> [0,255] uint8 -> PIL
                            t_img = valid_batch_input[b_idx]
                            img_np = (torch.clamp((t_img + 1.0) / 2.0, 0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            pil_images.append(Image.fromarray(img_np))
                        
                        inputs = mask2former_processor(images=pil_images, return_tensors="pt").to(mask2former_model.device)
                        m2f_out = mask2former_model(**inputs)
                        
                        # Panoptic Segmentation Result Retrieval
                        # target_sizes is needed to resize back to original
                        target_sizes = [(256, 256) for _ in range(actual_bs)]
                        results_m2f = mask2former_processor.post_process_panoptic_segmentation(m2f_out, target_sizes=target_sizes)
                        
                        for r in results_m2f:
                            seg_maps.append(r["segmentation"]) # Tensor [H, W]
                            
                    elif face_parser:
                        # FaceParser Inference
                        for b_idx in range(actual_bs):
                            p = face_parser.get_parsing_map(valid_batch_input[b_idx:b_idx+1])
                            seg_maps.append(p)
                    else:
                        print("Warning: No segmentation model available for SBF. Skipping.")
                        
                    
                    if len(seg_maps) > 0:
                        # 2. 受信側フィードバックマスク生成 (Feedback Binary)
                        # ここでは不確実性マップの上位 rate 分を「1」とするマスクをフィードバックとして想定
                        # SBFはこの「バラバラの1」を「セグメント単位」に整形する
                        fb_mask = create_retransmission_mask(uncertainty_map_for_sem, args.retransmission_rate)
                        
                        # 3. SBFマスク生成
                        sbf_mask = create_segment_feedback_mask(fb_mask, seg_maps, args.retransmission_rate)
                        
                        if sbf_mask is not None:
                            save_mask_tensor(sbf_mask, batch_files, batch_out_dir, "mask_sbf")
                            s, _ = sampler.sample_inpainting_awgn(
                                S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                                noisy_latent=z_received_pass1, snr_db=args.snr, mask=sbf_mask, x0=z0, eta=0.0,
                                conditioning=c, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                            )
                            results_batch["pass2_sbf"] = evaluate_and_save(
                                model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_sbf", 
                                loss_fn_lpips, loss_fn_dists, loss_fn_id, clip_scorer, fid_dirs["pass2_sbf"] if FID_AVAILABLE else None
                            )

                # =================================================================
                # Existing Semantic Methods (Legacy FaceParser)
                # =================================================================
                if "pass2_semantic" in target_keys and args.retransmission_rate > 0.0 and uncertainty_map_for_sem is not None and face_parser:
                    set_seed(current_batch_seed + 1000)
                    mask_sem_list = []
                    for j in range(actual_bs):
                        candidate_rate = min(1.0, args.retransmission_rate * 3.0)
                        u_map = uncertainty_map_for_sem[j:j+1]
                        mask_rec = create_retransmission_mask(u_map, candidate_rate) 
                        p = face_parser.get_parsing_map(valid_batch_input[j:j+1])
                        m_np = create_semantic_weighted_mask(mask_rec.cpu().numpy().squeeze(), p, args.retransmission_rate)
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
                    results_batch["pass2_semantic"] = evaluate_and_save(
                        model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_semantic", 
                        loss_fn_lpips, loss_fn_dists, loss_fn_id, clip_scorer, fid_dirs["pass2_semantic"] if FID_AVAILABLE else None
                    )

                if "pass2_semantic_only" in target_keys and args.retransmission_rate > 0.0 and face_parser:
                    set_seed(current_batch_seed + 1000)
                    mask_sem_list = []
                    for j in range(actual_bs):
                        p = face_parser.get_parsing_map(valid_batch_input[j:j+1])
                        m_np = create_semantic_only_mask((z0.shape[2], z0.shape[3]), p, args.retransmission_rate)
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
                    results_batch["pass2_semantic_only"] = evaluate_and_save(
                        model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_semantic_only", 
                        loss_fn_lpips, loss_fn_dists, loss_fn_id, clip_scorer, fid_dirs["pass2_semantic_only"] if FID_AVAILABLE else None
                    )

                if "pass2_random" in target_keys and args.retransmission_rate > 0.0:
                    set_seed(current_batch_seed + 1000)
                    mask_r = create_random_mask((z0.shape[0], 1, z0.shape[2], z0.shape[3]), args.retransmission_rate, z0.device)
                    save_mask_tensor(mask_r, batch_files, batch_out_dir, "mask_rand")
                    s, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], 
                        noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask_r, x0=z0, eta=0.0,
                        conditioning=c, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                    )
                    results_batch["pass2_random"] = evaluate_and_save(
                        model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_random", 
                        loss_fn_lpips, loss_fn_dists, loss_fn_id, clip_scorer, fid_dirs["pass2_random"] if FID_AVAILABLE else None
                    )

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
            "pass2_hybrid": "Pass 2 (Hybrid U+E)",
            "pass2_smart_hybrid": "Pass 2 (Smart Hybrid AND)",
            "pass2_oracle": "Pass 2 (Oracle Error)",
            "pass2_sbf": "Pass 2 (SBF/SegFeedback)" # 新規
        }
        
        executed_methods = [m for m in method_labels.keys() if m == "pass1" or m in target_keys]

        metric_keys = ["psnr", "lpips", "dists", "id_loss", "clip"]
        averages = {m: {met: np.mean([all_results[f][m]["metrics"][met] for f in all_results if all_results[f].get(m) and all_results[f][m].get("metrics") and met in all_results[f][m]["metrics"]]) for met in metric_keys} for m in executed_methods}
        
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
                 if met in ["psnr", "clip"]:
                     best_scores[met] = max(valid_values)
                 else:
                     best_scores[met] = min(valid_values)
             else:
                 best_scores[met] = -1

        valid_fids = [v for v in fids.values() if v is not None]
        best_fid = min(valid_fids) if valid_fids else None

        GREEN, BOLD, RESET = "\033[92m", "\033[1m", "\033[0m"
        print("\n" + "="*135 + f"\n  SUMMARY (N={len(all_results)})\n" + "="*135)
        print(f"{'Method':<30} | {'PSNR':<10} | {'LPIPS':<10} | {'DISTS':<10} | {'ID Loss':<10} | {'CLIP':<10} | {'FID':<10}")
        print("-" * 135)
        for m in executed_methods:
            name = method_labels[m]
            if m not in averages: continue
            row = f"{name:<30}"
            for met in metric_keys:
                if met not in averages[m]: 
                    row += f" | {'N/A':<10}"
                    continue
                val = averages[m][met]
                s = f"{val:.4f}"
                is_best = abs(val - best_scores[met]) < 1e-7 if best_scores[met] != -1 else False
                row += f" | {GREEN}{BOLD}*{s}*{RESET}" if is_best else f" | {s:<10}"
            f_val = fids.get(m)
            if f_val is not None:
                row += f" | {GREEN}{BOLD}*{f_val:.4f}*{RESET}" if best_fid and abs(f_val - best_fid) < 1e-7 else f" | {f_val:.4f}  "
            else: row += f" | {'N/A':<10}"
            print(row)
        print("-" * 135)

        output_data = {"summary": {"averages": {}, "fid": {}}}

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    output_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing JSON ({e}). Creating new one.")
        
        if "summary" not in output_data: output_data["summary"] = {}
        if "averages" not in output_data["summary"]: output_data["summary"]["averages"] = {}
        if "fid" not in output_data["summary"]: output_data["summary"]["fid"] = {}

        output_data["summary"]["averages"].update(averages)
        output_data["summary"]["fid"].update(fids)

        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"Metrics saved to: {json_path}")

    except Exception as e: print(f"Error: {e}"); raise e

if __name__ == "__main__": main()