import argparse
import os
import sys
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt

# ==========================================
# 【追加】自分自身のディレクトリをパスに追加
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==========================================

# --- Face Utils Import ---
# face_utils.py が同じディレクトリにあることを想定
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

# 【追加】FID計算用ライブラリ
try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: 'pytorch-fid' not found. FID calculation will be skipped.")


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
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
    # [0, 1] -> [-1, 1] & Resize
    img1_in = F.interpolate((img1 * 2) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
    img2_in = F.interpolate((img2 * 2) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
    with torch.no_grad():
        emb1 = net(img1_in)
        emb2 = net(img2_in)
    cosine_sim = F.cosine_similarity(emb1, emb2)
    return (1.0 - cosine_sim).item()

def compute_structural_uncertainty(uncertainty_map, latents_pass1, alpha=0.6):
    """
    不確実性マップに、Latent空間での構造情報（エッジ/変化量）重みを付与する。
    これにより、「背景のノイズ」ではなく「物体の輪郭付近の迷い」を強調する。
    
    Args:
        uncertainty_map (torch.Tensor): [B, 1, H, W] 従来の不確実性マップ
        latents_pass1 (torch.Tensor): [B, C, H, W] Pass 1で復元されたLatent
        alpha (float): 構造情報の重み (0.0~1.0). 0.6程度推奨。
    
    Returns:
        weighted_uncertainty (torch.Tensor): 重み付けされた新しいマップ
    """
    # 1. Latent空間での空間勾配（エッジ）を計算
    # 全チャネルの絶対値平均をとって構造を見る
    l_avg = torch.mean(torch.abs(latents_pass1), dim=1, keepdim=True) # [B, 1, H, W]
    
    # 空間微分 (Spatial Gradients)
    # dy: 上下画素の差分, dx: 左右画素の差分
    # パディングしてサイズを維持
    dy = torch.abs(l_avg[:, :, 1:, :] - l_avg[:, :, :-1, :])
    dy = F.pad(dy, (0, 0, 0, 1)) # 下側にパディング
    
    dx = torch.abs(l_avg[:, :, :, 1:] - l_avg[:, :, :, :-1])
    dx = F.pad(dx, (0, 1, 0, 0)) # 右側にパディング
    
    # 勾配の強さ (Saliency Map)
    gradient_magnitude = torch.sqrt(dx**2 + dy**2 + 1e-8)
    
    # 2. 正規化 (Min-Max Normalization to 0-1)
    def normalize_minmax(x):
        # バッチごとに正規化
        # x: [B, 1, H, W]
        min_val = x.view(x.shape[0], -1).min(dim=1, keepdim=True)[0].view(x.shape[0], 1, 1, 1)
        max_val = x.view(x.shape[0], -1).max(dim=1, keepdim=True)[0].view(x.shape[0], 1, 1, 1)
        return (x - min_val) / (max_val - min_val + 1e-8)
    
    unc_norm = normalize_minmax(uncertainty_map)
    grad_norm = normalize_minmax(gradient_magnitude)
    
    # 3. 統合 (Fusion)
    # 不確実性が高く、かつ構造情報(エッジ)がある場所を強調
    # (1-alpha) * Uncertainty + alpha * (Uncertainty * Gradient)
    weighted_map = (1 - alpha) * unc_norm + alpha * (unc_norm * grad_norm)
    
    return weighted_map

def save_heatmap_with_correlation(uncertainty_tensor, error_tensor, output_dir, base_fname, step_label, target_size=(256, 256)):
    """
    不確実性マップ保存 & 誤差相関計算
    """
    if uncertainty_tensor is None:
        return [0.0] * len(error_tensor)

    # Latent空間での平均 -> 画像サイズへアップサンプリング
    unc_mean = torch.mean(uncertainty_tensor, dim=1, keepdim=True)
    unc_upsampled = F.interpolate(unc_mean, size=target_size, mode='bilinear', align_corners=False).squeeze(1)

    unc_np = unc_upsampled.cpu().numpy()
    err_np = error_tensor.cpu().numpy()

    correlations = []
    for i in range(len(unc_np)):
        u_map = unc_np[i]
        e_map = err_np[i]

        # 相関計算
        u_flat = u_map.flatten()
        e_flat = e_map.flatten()
        if np.std(u_flat) > 1e-6 and np.std(e_flat) > 1e-6:
            corr = np.corrcoef(u_flat, e_flat)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

        # ヒートマップ保存
        u_min, u_max = u_map.min(), u_map.max()
        if (u_max - u_min) > 1e-6:
            u_norm = (u_map - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(u_map)
        
        cmap = plt.get_cmap('jet')
        colored_img = (cmap(u_norm)[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(colored_img)
        fname_no_ext = os.path.splitext(base_fname[i])[0]
        
        specific_dir = os.path.join(output_dir, str(i))
        os.makedirs(specific_dir, exist_ok=True)
        img.save(os.path.join(specific_dir, f"{fname_no_ext}_unc_{step_label}.png"))

    return correlations

def save_mask_tensor(mask_tensor, batch_files, batch_out_dir, suffix, target_size=(256, 256)):
    """
    マスクTensor (B, C, H, W) を画像として保存するヘルパー関数
    """
    if mask_tensor is None:
        return

    # Resize to target image size (Nearest Neighbor to keep 0/1)
    mask_resized = F.interpolate(mask_tensor, size=target_size, mode='nearest')
    mask_np = mask_resized.cpu().numpy() # (B, 1, H, W) or (B, C, H, W)
    
    for j, fname in enumerate(batch_files):
        if j >= len(mask_np): break
        
        # 1chを取り出す
        m = mask_np[j].squeeze() # (H, W)
        
        fname_no_ext = os.path.splitext(fname)[0]
        save_name = f"{fname_no_ext}_{suffix}.png"
        save_path = os.path.join(batch_out_dir, str(j), save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 0.0-1.0 -> 0-255
        mask_img_data = (m * 255).astype(np.uint8)
        Image.fromarray(mask_img_data, mode='L').save(save_path)

def create_retransmission_mask(uncertainty_map, rate):
    """
    不確実性マップから上位 rate% (不確実性が高い領域) を特定するマスクを作成
    """
    if uncertainty_map is None:
        return None
    
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
    """
    ランダムに rate% の領域を特定するベンチマーク用マスクを作成
    """
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
    """
    画像の保存とメトリック計算を行うヘルパー関数
    """
    gt_01 = torch.clamp((batch_input + 1.0) / 2.0, 0.0, 1.0)
    rec_01 = torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)
    
    results = {}
    
    for j, fname in enumerate(batch_files):
        # 画像保存
        fname_no_ext = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]
        save_name = f"{fname_no_ext}_{suffix}{ext}"
        
        x_rec_np = rec_01[j].cpu().permute(1, 2, 0).numpy()
        img_data = (x_rec_np * 255).astype(np.uint8)
        img = Image.fromarray(img_data)

        # 通常のバッチフォルダへ保存
        save_path = os.path.join(batch_out_dir, str(j), save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

        # FID用ディレクトリへの保存
        if fid_save_dir is not None:
            fid_save_path = os.path.join(fid_save_dir, fname)
            img.save(fid_save_path)

        # メトリクス計算
        metrics = {}
        metrics["psnr"] = calculate_psnr(gt_01[j:j+1], rec_01[j:j+1]).item()
        if loss_fn_lpips:
            metrics["lpips"] = loss_fn_lpips(gt_01[j:j+1]*2-1, rec_01[j:j+1]*2-1).item()
        if loss_fn_dists:
            metrics["dists"] = loss_fn_dists(gt_01[j:j+1], rec_01[j:j+1]).item()
        if loss_fn_id:
            metrics["id_loss"] = calculate_id_loss(gt_01[j:j+1], rec_01[j:j+1], loss_fn_id)
        
        results[fname] = metrics
        
    return results

def main():
    parser = argparse.ArgumentParser(description="DiffCom Retransmission Simulation with Structural Uncertainty")
    parser.add_argument("--input_dir", type=str, default="input_dir", help="Path to input images")
    parser.add_argument("--output_dir", type=str, default="results/", help="Path to output dir")
    parser.add_argument("--snr", type=float, default=-15.0, help="Channel SNR in dB")
    parser.add_argument("-r","--retransmission_rate", type=float, default=0.2, help="Retransmission rate (0.0-1.0)")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--face_model_path", type=str, default="models/face_parsing/79999_iter.pth", help="Path to face parsing BiSeNet model")
    # 【追加】構造情報の重み係数
    parser.add_argument("--struct_alpha", type=float, default=0.6, help="Weight for structural uncertainty (0.0-1.0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    
    sampler = DDIMSampler(model)

    # Initialize FaceParser
    face_parser = None
    if FACE_PARSING_AVAILABLE:
        if os.path.exists(args.face_model_path):
            print(f"Loading FaceParser from {args.face_model_path}...")
            face_parser = FaceParser(args.face_model_path, device='cuda')
        else:
            print(f"Warning: Face parsing model not found at {args.face_model_path}. Semantic retransmission will be skipped.")

    # Metrics init
    loss_fn_lpips = lpips.LPIPS(net='alex').cuda().eval() if LPIPS_AVAILABLE else None
    loss_fn_dists = DISTS().cuda().eval() if DISTS_AVAILABLE else None
    loss_fn_id = InceptionResnetV1(pretrained='vggface2').cuda().eval() if IDLOSS_AVAILABLE else None

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # -----------------------------------------------------------------------------------------------------
    # 【変更点】ログ出力とディレクトリ名にAlphaを含めるように変更
    # -----------------------------------------------------------------------------------------------------
    print(f"Starting simulation: SNR={args.snr}dB, Retransmission Rate={args.retransmission_rate*100}%, Alpha={args.struct_alpha}")

    # 実験全体の結果を保存するディレクトリ
    # ディレクトリ名に alpha 値を追加して重複を防ぐ
    experiment_dir = os.path.join(args.output_dir, f"snr{args.snr}dB_rate{args.retransmission_rate}_alpha{args.struct_alpha}")
    os.makedirs(experiment_dir, exist_ok=True)

    # FID計算用のディレクトリ準備
    fid_root = os.path.join(experiment_dir, "fid_images")
    fid_dirs = {
        "gt": os.path.join(fid_root, "gt"),
        "pass1": os.path.join(fid_root, "pass1"),
        "pass2_structural": os.path.join(fid_root, "pass2_structural"),
        "pass2_semantic": os.path.join(fid_root, "pass2_semantic"),
        "pass2_random": os.path.join(fid_root, "pass2_random"),
    }
    if FID_AVAILABLE:
        for d in fid_dirs.values():
            os.makedirs(d, exist_ok=True)
    
    all_results = {}
    # JSONファイル名にも alpha 値を追加
    json_filename = f"metrics_snr{args.snr}dB_rate{args.retransmission_rate}_alpha{args.struct_alpha}.json"
    json_path = os.path.join(experiment_dir, json_filename)
    # -----------------------------------------------------------------------------------------------------

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    def save_final_results(final_data):
        try:
            with open(json_path, 'w') as f:
                json.dump(final_data, f, indent=4, cls=NumpyEncoder)
            print(f"  Saved metrics to {json_path}")
        except Exception as e:
            print(f"  Error saving metrics: {e}")

    try:
        with torch.no_grad():
            for i in range(0, len(image_files), args.batch_size):
                batch_files = image_files[i : i + args.batch_size]
                actual_bs = len(batch_files)
                batch_idx = i // args.batch_size
                print(f"Processing Batch {batch_idx} ({actual_bs} images -> Padded to {args.batch_size})...")
                
                batch_out_dir = os.path.join(experiment_dir, f"batch{batch_idx}")
                os.makedirs(batch_out_dir, exist_ok=True)

                # 画像ロード
                batch_tensors = []
                for fname in batch_files:
                    img = Image.open(os.path.join(args.input_dir, fname)).convert("RGB").resize((256, 256), Image.BICUBIC)
                    batch_tensors.append(torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
                
                if actual_bs < args.batch_size:
                    pad_len = args.batch_size - actual_bs
                    last_tensor = batch_tensors[-1]
                    for _ in range(pad_len):
                        batch_tensors.append(last_tensor) 
                
                batch_input = torch.stack(batch_tensors).cuda()

                # === GT 保存 ===
                print("  Saving Ground Truth images...")
                valid_batch_input = batch_input[:actual_bs]
                gt_01 = torch.clamp((valid_batch_input + 1.0) / 2.0, 0.0, 1.0)
                gt_np = gt_01.cpu().permute(0, 2, 3, 1).numpy()
                
                for j, fname in enumerate(batch_files):
                    fname_no_ext = os.path.splitext(fname)[0]
                    ext = os.path.splitext(fname)[1]
                    save_name = f"{fname_no_ext}_gt{ext}"
                    save_path = os.path.join(batch_out_dir, str(j), save_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    img_data = (gt_np[j] * 255).astype(np.uint8)
                    Image.fromarray(img_data).save(save_path)

                    if FID_AVAILABLE:
                        Image.fromarray(img_data).save(os.path.join(fid_dirs["gt"], fname))

                # 1. Encode & Channel
                encoder_posterior = model.encode_first_stage(batch_input)
                z0 = encoder_posterior if isinstance(encoder_posterior, torch.Tensor) else encoder_posterior.mode()
                z_received_pass1 = add_awgn_channel(z0, args.snr)

                # === Pass 1: 初回送信 ===
                print("  [Pass 1] Decoding and estimating uncertainty...")
                samples_pass1, history_pass1 = sampler.sample_awgn(
                    S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                    noisy_latent=z_received_pass1, snr_db=args.snr, eta=0.0, verbose=True
                )
                
                uncertainty_map = history_pass1[0][1] if history_pass1 else None

                x_rec_pass1 = model.decode_first_stage(samples_pass1)
                valid_x_rec_pass1 = x_rec_pass1[:actual_bs]
                
                pass1_results = evaluate_and_save(
                    valid_x_rec_pass1, valid_batch_input, batch_files, batch_out_dir, "pass1",
                    loss_fn_lpips, loss_fn_dists, loss_fn_id,
                    fid_save_dir=fid_dirs["pass1"] if FID_AVAILABLE else None
                )

                error_map_pass1 = torch.mean((valid_batch_input - valid_x_rec_pass1) ** 2, dim=1)
                valid_uncertainty_map = uncertainty_map[:actual_bs] if uncertainty_map is not None else None
                corrs_pass1 = save_heatmap_with_correlation(
                    valid_uncertainty_map, error_map_pass1, batch_out_dir, batch_files, "pass1_detection"
                )
                print(f"  [Pass 1] Uncertainty-Error Correlation: {sum(corrs_pass1)/len(corrs_pass1):.4f}")

                # === Retransmission Logic (Method 1: Structural Uncertainty) ===
                pass2_struc_results = {}
                if args.retransmission_rate > 0.0 and uncertainty_map is not None:
                    print(f"  [Retransmission - Structural Uncertainty] Rate={args.retransmission_rate*100}% (alpha={args.struct_alpha})")
                    
                    structural_unc_map = compute_structural_uncertainty(
                        uncertainty_map, samples_pass1, alpha=args.struct_alpha
                    )
                    
                    mask_unc = create_retransmission_mask(structural_unc_map, args.retransmission_rate)
                    
                    # マスク保存
                    save_mask_tensor(mask_unc, batch_files, batch_out_dir, "mask_structural")

                    # マップ自体も保存（可視化用）
                    save_heatmap_with_correlation(
                        structural_unc_map[:actual_bs], error_map_pass1, batch_out_dir, batch_files, "map_structural"
                    )

                    samples_unc, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                        noisy_latent=z_received_pass1,
                        snr_db=args.snr,
                        mask=mask_unc,
                        x0=z0,
                        eta=0.0, verbose=True
                    )
                    
                    x_rec_unc = model.decode_first_stage(samples_unc)
                    valid_x_rec_unc = x_rec_unc[:actual_bs]
                    pass2_struc_results = evaluate_and_save(
                        valid_x_rec_unc, valid_batch_input, batch_files, batch_out_dir, "pass2_structural",
                        loss_fn_lpips, loss_fn_dists, loss_fn_id,
                        fid_save_dir=fid_dirs["pass2_structural"] if FID_AVAILABLE else None
                    )

                # === Retransmission Logic (Method 2: Semantic/Face-aware) ===
                pass2_sem_results = {}
                if args.retransmission_rate > 0.0 and uncertainty_map is not None and face_parser is not None:
                    print(f"  [Retransmission - Semantic] Rate={args.retransmission_rate*100}%")
                    
                    mask_sem_list = []
                    unc_np = uncertainty_map.cpu().numpy()
                    
                    for j in range(actual_bs):
                        img_tensor = valid_x_rec_pass1[j:j+1] 
                        try:
                            parsing_map = face_parser.get_parsing_map(img_tensor)
                            
                            fname_no_ext = os.path.splitext(batch_files[j])[0]
                            seg_save_name = f"{fname_no_ext}_segmap.png"
                            seg_save_path = os.path.join(batch_out_dir, str(j), seg_save_name)
                            os.makedirs(os.path.dirname(seg_save_path), exist_ok=True)
                            vis_seg_map = (parsing_map * 13).astype(np.uint8)
                            Image.fromarray(vis_seg_map, mode='L').save(seg_save_path)

                        except Exception as e:
                            print(f"    Face parsing failed for image {j}: {e}. Using empty mask.")
                            parsing_map = np.zeros((img_tensor.shape[2], img_tensor.shape[3]), dtype=np.uint8)

                        u_map_single = unc_np[j].squeeze()
                        latent_mask_np = create_semantic_uncertainty_mask(u_map_single, parsing_map, args.retransmission_rate)
                        
                        m_tensor = torch.from_numpy(latent_mask_np).unsqueeze(0).unsqueeze(0).float()
                        mask_sem_list.append(m_tensor)
                        
                        fname_no_ext = os.path.splitext(batch_files[j])[0]
                        save_name = f"{fname_no_ext}_mask_semantic.png"
                        save_path = os.path.join(batch_out_dir, str(j), save_name)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        vis_mask = F.interpolate(m_tensor, size=(256, 256), mode='nearest').squeeze().cpu().numpy()
                        mask_img_data = (vis_mask * 255).astype(np.uint8)
                        Image.fromarray(mask_img_data, mode='L').save(save_path)
                    
                    mask_semantic = torch.cat(mask_sem_list, dim=0).to(z0.device)
                    if actual_bs < args.batch_size:
                        pad_len = args.batch_size - actual_bs
                        pad = mask_semantic[-1:].repeat(pad_len, 1, 1, 1)
                        mask_semantic = torch.cat([mask_semantic, pad], dim=0)

                    samples_sem, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                        noisy_latent=z_received_pass1,
                        snr_db=args.snr,
                        mask=mask_semantic,
                        x0=z0,
                        eta=0.0, verbose=True
                    )
                    
                    x_rec_sem = model.decode_first_stage(samples_sem)
                    valid_x_rec_sem = x_rec_sem[:actual_bs]
                    pass2_sem_results = evaluate_and_save(
                        valid_x_rec_sem, valid_batch_input, batch_files, batch_out_dir, "pass2_semantic",
                        loss_fn_lpips, loss_fn_dists, loss_fn_id,
                        fid_save_dir=fid_dirs["pass2_semantic"] if FID_AVAILABLE else None
                    )

                # === Retransmission Logic (Benchmark: Random) ===
                pass2_rand_results = {}
                if args.retransmission_rate > 0.0:
                    print(f"  [Retransmission - Random] Rate={args.retransmission_rate*100}%")
                    mask_shape = (z0.shape[0], 1, z0.shape[2], z0.shape[3])
                    mask_rand = create_random_mask(mask_shape, args.retransmission_rate, z0.device)
                    
                    save_mask_tensor(mask_rand, batch_files, batch_out_dir, "mask_rand")

                    samples_rand, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                        noisy_latent=z_received_pass1,
                        snr_db=args.snr,
                        mask=mask_rand,
                        x0=z0,
                        eta=0.0, verbose=True
                    )
                    
                    x_rec_rand = model.decode_first_stage(samples_rand)
                    valid_x_rec_rand = x_rec_rand[:actual_bs]
                    pass2_rand_results = evaluate_and_save(
                        valid_x_rec_rand, valid_batch_input, batch_files, batch_out_dir, "pass2_rand",
                        loss_fn_lpips, loss_fn_dists, loss_fn_id,
                        fid_save_dir=fid_dirs["pass2_random"] if FID_AVAILABLE else None
                    )

                # === 結果統合 ===
                for j, fname in enumerate(batch_files):
                    all_results[fname] = {
                        "pass1": {
                            "metrics": pass1_results[fname],
                            "correlation": corrs_pass1[j]
                        },
                        "pass2_structural": {
                            "metrics": pass2_struc_results.get(fname) if pass2_struc_results else None
                        },
                        "pass2_semantic": {
                            "metrics": pass2_sem_results.get(fname) if pass2_sem_results else None
                        },
                        "pass2_random": {
                            "metrics": pass2_rand_results.get(fname) if pass2_rand_results else None
                        }
                    }
                
                # Cleanup
                del batch_input, z0, z_received_pass1, samples_pass1, history_pass1
                del x_rec_pass1, valid_x_rec_pass1
                if 'samples_unc' in locals(): del samples_unc
                if 'samples_sem' in locals(): del samples_sem
                if 'samples_rand' in locals(): del samples_rand
                gc.collect()
                torch.cuda.empty_cache()

        # === Summary Calculation ===
        method_labels = {
            "pass1": "Pass 1 (Initial)",
            "pass2_structural": "Pass 2 (Structural Unc)", # ラベル変更
            "pass2_semantic": "Pass 2 (Semantic/Face)",
            "pass2_random": "Pass 2 (Random)"
        }
        metric_keys = ["psnr", "lpips", "dists", "id_loss"]

        averages = {m_key: {} for m_key in method_labels}
        for m_key in method_labels:
            for met in metric_keys:
                vals = []
                for fname, res in all_results.items():
                    if res.get(m_key) and res[m_key].get("metrics") and met in res[m_key]["metrics"]:
                        vals.append(res[m_key]["metrics"][met])
                if len(vals) > 0:
                    averages[m_key][met] = sum(vals) / len(vals)
                else:
                    averages[m_key][met] = None

        # === FID Calculation ===
        fids = {}
        if FID_AVAILABLE:
            print("\nCalculating FID...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            target_methods = ["pass1"]
            if args.retransmission_rate > 0.0:
                target_methods.extend(["pass2_structural", "pass2_semantic", "pass2_random"])

            for m_key in target_methods:
                if m_key in fid_dirs and os.path.exists(fid_dirs[m_key]) and len(os.listdir(fid_dirs[m_key])) > 0:
                    try:
                        fid_value = fid_score.calculate_fid_given_paths(
                            [fid_dirs["gt"], fid_dirs[m_key]],
                            batch_size=50,
                            device=device,
                            dims=2048
                        )
                        fids[m_key] = fid_value
                        print(f"  FID ({m_key}): {fid_value:.4f}")
                    except Exception as e:
                        print(f"  Failed to calculate FID for {m_key}: {e}")
                        fids[m_key] = None
                else:
                    fids[m_key] = None

        # === Final Save ===
        final_output = {
            "per_image_results": all_results,
            "summary": {
                "averages": averages,
                "fid": fids
            }
        }
        save_final_results(final_output)

        # === Best Score Calculation ===
        best_scores = {}
        for met in metric_keys:
            valid_avgs = [averages[m][met] for m in method_labels if averages[m][met] is not None]
            if valid_avgs:
                if met == "psnr":
                    best_scores[met] = max(valid_avgs)
                else:
                    best_scores[met] = min(valid_avgs)
        
        valid_fids = [v for v in fids.values() if v is not None]
        best_fid = min(valid_fids) if valid_fids else None

        GREEN = "\033[92m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print("\n" + "="*95)
        print(f"  SIMULATION SUMMARY: Average Metrics (N={len(all_results)})")
        print("="*95)
        header = f"{'Method':<30} | {'PSNR':<10} | {'LPIPS':<10} | {'DISTS':<10} | {'ID Loss':<10} | {'FID':<10}"
        print(header)
        print("-" * 95)

        for m_key, m_name in method_labels.items():
            row_str = f"{m_name:<30}"
            for met in metric_keys:
                val = averages[m_key].get(met)
                if val is None:
                    row_str += f" | {'N/A':<10}"
                    continue
                val_str = f"{val:.4f}"
                is_best = False
                if met in best_scores and abs(val - best_scores[met]) < 1e-9:
                    is_best = True
                if is_best:
                    row_str += f" | {GREEN}{BOLD}*{val_str}* {RESET}"
                else:
                    row_str += f" | {val_str:<10}"
            
            fid_val = fids.get(m_key)
            if fid_val is not None:
                 fid_str = f"{fid_val:.4f}"
                 if best_fid is not None and abs(fid_val - best_fid) < 1e-6:
                     row_str += f" | {GREEN}{BOLD}*{fid_str}* {RESET}"
                 else:
                     row_str += f" | {fid_str:<10}    "
            else:
                 row_str += f" | {'N/A':<10}"

            print(row_str)
        print("-" * 95 + "\n")

    except KeyboardInterrupt:
        print("\n\n!!! Simulation Interrupted by User !!!")
        print("Saving results collected so far...")
        if 'all_results' in locals():
            save_final_results({"per_image_results": all_results, "summary": {"interrupted": True}})
        print("Exiting...")

    except Exception as e:
        print(f"\n\n!!! An error occurred: {e} !!!")
        if 'all_results' in locals():
             save_final_results({"per_image_results": all_results, "summary": {"error": str(e)}})
        raise e

    print("Simulation Finished.")

if __name__ == "__main__":
    main()