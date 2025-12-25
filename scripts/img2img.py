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

def create_retransmission_mask(uncertainty_map, rate):
    """
    不確実性マップから上位 rate% の領域を特定するマスクを作成
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
                      loss_fn_lpips, loss_fn_dists, loss_fn_id):
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
        save_path = os.path.join(batch_out_dir, str(j), save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray((x_rec_np * 255).astype(np.uint8)).save(save_path)

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
    parser = argparse.ArgumentParser(description="DiffCom Retransmission Simulation")
    parser.add_argument("--input_dir", type=str, default="input_dir", help="Path to input images")
    parser.add_argument("--output_dir", type=str, default="results/", help="Path to output dir")
    parser.add_argument("--snr", type=float, default=-10.0, help="Channel SNR in dB")
    parser.add_argument("--retransmission_rate", type=float, default=0.1, help="Retransmission rate (0.0-1.0)")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    
    sampler = DDIMSampler(model)

    # Metrics init
    loss_fn_lpips = lpips.LPIPS(net='alex').cuda().eval() if LPIPS_AVAILABLE else None
    loss_fn_dists = DISTS().cuda().eval() if DISTS_AVAILABLE else None
    loss_fn_id = InceptionResnetV1(pretrained='vggface2').cuda().eval() if IDLOSS_AVAILABLE else None

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Starting simulation: SNR={args.snr}dB, Retransmission Rate={args.retransmission_rate*100}%")

    # 実験全体の結果を保存するディレクトリ
    experiment_dir = os.path.join(args.output_dir, f"snr{args.snr}dB_rate{args.retransmission_rate}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 結果保存用変数
    all_results = {}
    json_path = os.path.join(experiment_dir, "metrics.json")

    # JSON保存用ヘルパー関数
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    def save_all_results():
        """現在のall_resultsをJSONに保存する"""
        try:
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=4, cls=NumpyEncoder)
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
                
                # バッチごとの画像保存用ディレクトリ
                batch_out_dir = os.path.join(experiment_dir, f"batch{batch_idx}")
                os.makedirs(batch_out_dir, exist_ok=True)

                # 画像ロード & パディング処理
                batch_tensors = []
                for fname in batch_files:
                    img = Image.open(os.path.join(args.input_dir, fname)).convert("RGB").resize((256, 256), Image.BICUBIC)
                    batch_tensors.append(torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
                
                # パディング: バッチサイズが足りない場合、最後の画像を複製して埋める
                if actual_bs < args.batch_size:
                    pad_len = args.batch_size - actual_bs
                    last_tensor = batch_tensors[-1]
                    for _ in range(pad_len):
                        batch_tensors.append(last_tensor) 
                
                # 常にサイズ[batch_size, C, H, W]のテンソルを作成
                batch_input = torch.stack(batch_tensors).cuda()

                # === GT 保存 (有効な部分のみ) ===
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
                    Image.fromarray((gt_np[j] * 255).astype(np.uint8)).save(save_path)

                # 1. Encode & Channel (Full Batch Processing)
                encoder_posterior = model.encode_first_stage(batch_input)
                z0 = encoder_posterior if isinstance(encoder_posterior, torch.Tensor) else encoder_posterior.mode()
                z_received_pass1 = add_awgn_channel(z0, args.snr)

                # === Pass 1: 初回送信 & 不確実性測定 ===
                print("  [Pass 1] Decoding and estimating uncertainty...")
                samples_pass1, history_pass1 = sampler.sample_awgn(
                    S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                    noisy_latent=z_received_pass1, snr_db=args.snr, eta=0.0, verbose=True
                )
                
                uncertainty_map = history_pass1[0][1] if history_pass1 else None

                # Pass 1 Decode & Evaluation (Slicing)
                x_rec_pass1 = model.decode_first_stage(samples_pass1)
                
                # 有効データのみスライス
                valid_x_rec_pass1 = x_rec_pass1[:actual_bs]
                pass1_results = evaluate_and_save(
                    valid_x_rec_pass1, valid_batch_input, batch_files, batch_out_dir, "pass1",
                    loss_fn_lpips, loss_fn_dists, loss_fn_id
                )

                # Pass 1 Correlation (Slicing)
                error_map_pass1 = torch.mean((valid_batch_input - valid_x_rec_pass1) ** 2, dim=1)
                valid_uncertainty_map = uncertainty_map[:actual_bs] if uncertainty_map is not None else None
                
                corrs_pass1 = save_heatmap_with_correlation(
                    valid_uncertainty_map, error_map_pass1, batch_out_dir, batch_files, "pass1_detection"
                )
                print(f"  [Pass 1] Uncertainty-Error Correlation: {sum(corrs_pass1)/len(corrs_pass1):.4f}")

                # === Retransmission Logic (Proposed: Uncertainty) ===
                pass2_unc_results = {}
                if args.retransmission_rate > 0.0 and uncertainty_map is not None:
                    print(f"  [Retransmission - Uncertainty] Rate={args.retransmission_rate*100}%")
                    mask_unc = create_retransmission_mask(uncertainty_map, args.retransmission_rate)
                    
                    # Save Masks (Slicing)
                    print("  Saving Uncertainty Mask images...")
                    valid_mask_unc = mask_unc[:actual_bs]
                    mask_unc_np = valid_mask_unc.cpu().permute(0, 2, 3, 1).numpy()
                    for j, fname in enumerate(batch_files):
                        fname_no_ext = os.path.splitext(fname)[0]
                        save_name = f"{fname_no_ext}_mask_unc.png"
                        save_path = os.path.join(batch_out_dir, str(j), save_name)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        mask_img_data = (mask_unc_np[j].squeeze() * 255).astype(np.uint8)
                        Image.fromarray(mask_img_data, mode='L').save(save_path)

                    # Sample Inpainting (Full Batch)
                    samples_unc, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                        noisy_latent=z_received_pass1,
                        snr_db=args.snr,
                        mask=mask_unc,
                        x0=z0,
                        eta=0.0, verbose=True
                    )
                    
                    # Decode & Evaluate (Slicing)
                    x_rec_unc = model.decode_first_stage(samples_unc)
                    valid_x_rec_unc = x_rec_unc[:actual_bs]
                    
                    pass2_unc_results = evaluate_and_save(
                        valid_x_rec_unc, valid_batch_input, batch_files, batch_out_dir, "pass2_unc",
                        loss_fn_lpips, loss_fn_dists, loss_fn_id
                    )

                # === Retransmission Logic (Benchmark: Random) ===
                pass2_rand_results = {}
                if args.retransmission_rate > 0.0:
                    print(f"  [Retransmission - Random] Rate={args.retransmission_rate*100}%")
                    mask_shape = (z0.shape[0], 1, z0.shape[2], z0.shape[3])
                    mask_rand = create_random_mask(mask_shape, args.retransmission_rate, z0.device)
                    
                    # Save Masks (Slicing)
                    print("  Saving Random Mask images...")
                    valid_mask_rand = mask_rand[:actual_bs]
                    mask_rand_np = valid_mask_rand.cpu().permute(0, 2, 3, 1).numpy()
                    for j, fname in enumerate(batch_files):
                        fname_no_ext = os.path.splitext(fname)[0]
                        save_name = f"{fname_no_ext}_mask_random.png"
                        save_path = os.path.join(batch_out_dir, str(j), save_name)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        mask_img_data = (mask_rand_np[j].squeeze() * 255).astype(np.uint8)
                        Image.fromarray(mask_img_data, mode='L').save(save_path)
                    
                    # Sample Inpainting (Full Batch)
                    samples_rand, _ = sampler.sample_inpainting_awgn(
                        S=args.ddim_steps, batch_size=args.batch_size, shape=z0.shape[1:],
                        noisy_latent=z_received_pass1,
                        snr_db=args.snr,
                        mask=mask_rand,
                        x0=z0,
                        eta=0.0, verbose=True
                    )
                    
                    # Decode & Evaluate (Slicing)
                    x_rec_rand = model.decode_first_stage(samples_rand)
                    valid_x_rec_rand = x_rec_rand[:actual_bs]
                    
                    pass2_rand_results = evaluate_and_save(
                        valid_x_rec_rand, valid_batch_input, batch_files, batch_out_dir, "pass2_rand",
                        loss_fn_lpips, loss_fn_dists, loss_fn_id
                    )

                # === 結果の統合とall_resultsへの追加 ===
                for j, fname in enumerate(batch_files):
                    all_results[fname] = {
                        "pass1": {
                            "metrics": pass1_results[fname],
                            "correlation": corrs_pass1[j]
                        },
                        "pass2_uncertainty": {
                            "metrics": pass2_unc_results.get(fname) if pass2_unc_results else None
                        },
                        "pass2_random": {
                            "metrics": pass2_rand_results.get(fname) if pass2_rand_results else None
                        }
                    }
                
                # バッチ完了ごとにJSONを保存 (途中経過を保証)
                save_all_results()
                
                # === メモリ解放 ===
                del batch_input, z0, z_received_pass1, samples_pass1, history_pass1, uncertainty_map
                del x_rec_pass1, error_map_pass1, pass1_results, corrs_pass1
                
                if 'samples_unc' in locals(): del samples_unc
                if 'x_rec_unc' in locals(): del x_rec_unc
                if 'mask_unc' in locals(): del mask_unc
                if 'pass2_unc_results' in locals(): del pass2_unc_results
                
                if 'samples_rand' in locals(): del samples_rand
                if 'x_rec_rand' in locals(): del x_rec_rand
                if 'mask_rand' in locals(): del mask_rand
                if 'pass2_rand_results' in locals(): del pass2_rand_results
                
                if 'gt_01' in locals(): del gt_01
                if 'valid_batch_input' in locals(): del valid_batch_input
                if 'valid_x_rec_pass1' in locals(): del valid_x_rec_pass1
                
                gc.collect() # Pythonオブジェクトの参照を切る
                torch.cuda.empty_cache() # GPUメモリを解放する

    except KeyboardInterrupt:
        print("\n\n!!! Simulation Interrupted by User !!!")
        print("Saving results collected so far...")
        save_all_results()
        print("Exiting...")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\n!!! An error occurred: {e} !!!")
        print("Saving results collected so far...")
        save_all_results()
        raise e

    print("\nSimulation Finished.")

if __name__ == "__main__":
    main()