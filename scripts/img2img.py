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

# ==========================================
# 自分自身のディレクトリをパスに追加
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Face Utils Import ---
try:
    from face_utils import FaceParser, create_semantic_uncertainty_mask
    FACE_PARSING_AVAILABLE = True
except ImportError as e:
    FACE_PARSING_AVAILABLE = False
    print(f"Warning: 'face_utils' not found or failed to import: {e}")
    print("Semantic retransmission will be skipped.")

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
    """
    不確実性マップの平滑化＋確率的ノイズ混合
    """
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

def save_mask_tensor(mask_tensor, batch_files, batch_out_dir, suffix, target_size=(256, 256)):
    if mask_tensor is None: return
    mask_resized = F.interpolate(mask_tensor, size=target_size, mode='nearest')
    mask_np = mask_resized.cpu().numpy()
    for j, fname in enumerate(batch_files):
        if j >= len(mask_np): break
        m = mask_np[j].squeeze()
        save_path = os.path.join(batch_out_dir, str(j), f"{os.path.splitext(fname)[0]}_{suffix}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray((m * 255).astype(np.uint8), mode='L').save(save_path)

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
    parser = argparse.ArgumentParser(description="DiffCom Retransmission Simulation")
    parser.add_argument("--input_dir", type=str, default="input_dir")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--snr", type=float, default=-15.0)
    parser.add_argument("-r","--retransmission_rate", type=float, default=0.2)
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--face_model_path", type=str, default="models/face_parsing/79999_iter.pth")
    parser.add_argument("--struct_alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    face_parser = FaceParser(args.face_model_path, device='cuda') if FACE_PARSING_AVAILABLE and os.path.exists(args.face_model_path) else None
    loss_fn_lpips = lpips.LPIPS(net='alex').cuda().eval() if LPIPS_AVAILABLE else None
    loss_fn_dists = DISTS().cuda().eval() if DISTS_AVAILABLE else None
    loss_fn_id = InceptionResnetV1(pretrained='vggface2').cuda().eval() if IDLOSS_AVAILABLE else None

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    experiment_dir = os.path.join(args.output_dir, f"snr{args.snr}dB_rate{args.retransmission_rate}_alpha{args.struct_alpha}_seed{args.seed}")
    os.makedirs(experiment_dir, exist_ok=True)
    fid_root = os.path.join(experiment_dir, "fid_images")
    fid_dirs = {k: os.path.join(fid_root, k) for k in ["gt", "pass1", "pass2_structural", "pass2_raw", "pass2_semantic", "pass2_random"]}
    if FID_AVAILABLE: [os.makedirs(d, exist_ok=True) for d in fid_dirs.values()]
    
    all_results = {}
    json_path = os.path.join(experiment_dir, f"metrics.json")

    try:
        with torch.no_grad():
            for i in range(0, len(image_files), args.batch_size):
                batch_files = image_files[i : i + args.batch_size]
                actual_bs = len(batch_files)
                batch_idx = i // args.batch_size
                batch_out_dir = os.path.join(experiment_dir, f"batch{batch_idx}")
                os.makedirs(batch_out_dir, exist_ok=True)

                batch_tensors = []
                for fname in batch_files:
                    img = Image.open(os.path.join(args.input_dir, fname)).convert("RGB").resize((256, 256), Image.BICUBIC)
                    batch_tensors.append(torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
                if actual_bs < args.batch_size: batch_tensors.extend([batch_tensors[-1]] * (args.batch_size - actual_bs))
                batch_input = torch.stack(batch_tensors).cuda()

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

                # Pass 1
                samples_pass1, history_pass1 = sampler.sample_awgn(S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], noisy_latent=z_received_pass1, snr_db=args.snr, eta=0.0)
                uncertainty_map = history_pass1[0][1] if history_pass1 else None
                x_rec_pass1 = model.decode_first_stage(samples_pass1)
                pass1_results = evaluate_and_save(x_rec_pass1[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass1", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass1"] if FID_AVAILABLE else None)

                # Pass 2 Methods
                methods = [
                    ("pass2_structural", lambda: create_retransmission_mask(compute_structural_uncertainty(uncertainty_map, samples_pass1, alpha=args.struct_alpha), args.retransmission_rate)),
                    ("pass2_raw", lambda: create_retransmission_mask(uncertainty_map, args.retransmission_rate))
                ]
                
                results_batch = {"pass1": pass1_results}
                for key, mask_gen in methods:
                    if args.retransmission_rate > 0.0 and uncertainty_map is not None:
                        mask = mask_gen()
                        save_mask_tensor(mask, batch_files, batch_out_dir, f"mask_{key.split('_')[1]}")
                        s, _ = sampler.sample_inpainting_awgn(S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask, x0=z0, eta=0.0)
                        results_batch[key] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, key, loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs[key] if FID_AVAILABLE else None)

                # Semantic
                if args.retransmission_rate > 0.0 and uncertainty_map is not None and face_parser:
                    mask_sem_list = []
                    for j in range(actual_bs):
                        p = face_parser.get_parsing_map(x_rec_pass1[j:j+1])
                        m = torch.from_numpy(create_semantic_uncertainty_mask(uncertainty_map[j].cpu().numpy().squeeze(), p, args.retransmission_rate)).unsqueeze(0).unsqueeze(0).float()
                        mask_sem_list.append(m)
                    mask_sem = torch.cat(mask_sem_list, dim=0)
                    save_mask_tensor(mask_sem, batch_files, batch_out_dir, "mask_semantic")
                    mask_sem_p = torch.cat([mask_sem, mask_sem[-1:].repeat(args.batch_size-actual_bs, 1, 1, 1)], dim=0).cuda()
                    s, _ = sampler.sample_inpainting_awgn(S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask_sem_p, x0=z0, eta=0.0)
                    results_batch["pass2_semantic"] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_semantic", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass2_semantic"] if FID_AVAILABLE else None)

                # Random
                if args.retransmission_rate > 0.0:
                    mask_r = create_random_mask((z0.shape[0], 1, z0.shape[2], z0.shape[3]), args.retransmission_rate, z0.device)
                    save_mask_tensor(mask_r, batch_files, batch_out_dir, "mask_rand")
                    s, _ = sampler.sample_inpainting_awgn(S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:], noisy_latent=z_received_pass1, snr_db=args.snr, mask=mask_r, x0=z0, eta=0.0)
                    results_batch["pass2_random"] = evaluate_and_save(model.decode_first_stage(s)[:actual_bs], valid_batch_input, batch_files, batch_out_dir, "pass2_random", loss_fn_lpips, loss_fn_dists, loss_fn_id, fid_dirs["pass2_random"] if FID_AVAILABLE else None)

                for f in batch_files:
                    all_results[f] = {k: {"metrics": results_batch[k].get(f)} for k in ["pass1", "pass2_structural", "pass2_raw", "pass2_semantic", "pass2_random"] if results_batch.get(k)}
                gc.collect(); torch.cuda.empty_cache()

        # Summary & Highlighting
        method_labels = {"pass1": "Pass 1 (Initial)", "pass2_structural": "Pass 2 (Struct/Smooth)", "pass2_raw": "Pass 2 (Raw/NoSmooth)", "pass2_semantic": "Pass 2 (Semantic/Face)", "pass2_random": "Pass 2 (Random)"}
        metric_keys = ["psnr", "lpips", "dists", "id_loss"]
        averages = {m: {met: np.mean([all_results[f][m]["metrics"][met] for f in all_results if all_results[f].get(m) and all_results[f][m].get("metrics")]) for met in metric_keys} for m in method_labels}
        
        fids = {}
        if FID_AVAILABLE:
            for m in method_labels.keys():
                fids[m] = fid_score.calculate_fid_given_paths([fid_dirs["gt"], fid_dirs[m]], 50, 'cuda', 2048) if os.path.exists(fid_dirs[m]) and len(os.listdir(fid_dirs[m]))>0 else None

        best_scores = {met: (max if met == "psnr" else min)([averages[m][met] for m in method_labels if not np.isnan(averages[m][met])]) for met in metric_keys}
        valid_fids = [v for v in fids.values() if v is not None]
        best_fid = min(valid_fids) if valid_fids else None

        GREEN, BOLD, RESET = "\033[92m", "\033[1m", "\033[0m"
        print("\n" + "="*95 + f"\n  SUMMARY (N={len(all_results)})\n" + "="*95)
        print(f"{'Method':<30} | {'PSNR':<10} | {'LPIPS':<10} | {'DISTS':<10} | {'ID Loss':<10} | {'FID':<10}")
        print("-" * 95)
        for m, name in method_labels.items():
            row = f"{name:<30}"
            for met in metric_keys:
                val = averages[m][met]
                s = f"{val:.4f}"
                row += f" | {GREEN}{BOLD}*{s}*{RESET}" if abs(val - best_scores[met]) < 1e-7 else f" | {s:<10}"
            f_val = fids.get(m)
            if f_val is not None:
                row += f" | {GREEN}{BOLD}*{f_val:.4f}*{RESET}" if best_fid and abs(f_val - best_fid) < 1e-7 else f" | {f_val:.4f}  "
            else: row += f" | {'N/A':<10}"
            print(row)
        print("-" * 95)
        with open(json_path, 'w') as f: json.dump({"summary": {"averages": averages, "fid": fids}}, f, indent=4)
    except Exception as e: print(f"Error: {e}"); raise e
if __name__ == "__main__": main()