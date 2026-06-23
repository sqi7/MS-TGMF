import argparse
import time
import os
import sys
from pathlib import Path

import torch


# 保证从项目根目录运行时可以正确导入 baseline
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from baseline.utils.config import Config
from baseline.datasets import build_dataloader
from baseline.models import build_net


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Params / FLOPs / FPS for K-Lane models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint, usually best.pth")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for FPS measurement")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--measure-flops", action="store_true", help="Try to measure FLOPs with thop")
    return parser.parse_args()


def move_to_device(data, device):
    """Recursively move tensors in a nested batch dict/list/tuple to GPU."""
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix saved by DataParallel."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def extract_state_dict(ckpt):
    """
    Support common checkpoint formats:
    - raw state_dict
    - {'state_dict': ...}
    - {'model': ...}
    - {'net': ...}
    - {'model_state_dict': ...}
    """
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint is not a dict.")

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
    ]

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    # 如果本身就是 state_dict
    tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
    if len(tensor_values) > 0:
        return ckpt

    raise KeyError(f"Cannot find state_dict in checkpoint. Available keys: {list(ckpt.keys())}")


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt)
    state_dict = strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[CKPT] Loaded: {ckpt_path}")
    print(f"[CKPT] Missing keys: {len(missing)}")
    print(f"[CKPT] Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[CKPT] First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[CKPT] First unexpected keys:", unexpected[:10])


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def build_model(cfg, ckpt_path, device):
    model = build_net(cfg)
    model = model.to(device)

    load_checkpoint(model, ckpt_path, device)

    model.eval()
    return model


def try_measure_flops(model, example_batch):
    """
    FLOPs measurement may fail because this model uses dict input and post-processing.
    If it fails, we safely skip it.
    """
    try:
        from thop import profile
    except Exception as e:
        print(f"[FLOPs] thop is not installed or cannot be imported: {e}")
        return None

    try:
        model.eval()
        with torch.no_grad():
            flops, params = profile(
                model,
                inputs=(example_batch,),
                verbose=False
            )
        return flops
    except Exception as e:
        print(f"[FLOPs] Failed to measure FLOPs with thop: {e}")
        print("[FLOPs] You can still report Params and FPS.")
        return None


def measure_fps(model, dataloader, device, warmup=30, iters=200):
    model.eval()

    data_iter = iter(dataloader)

    def next_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return move_to_device(batch, device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            batch = next_batch()
            _ = model(batch)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # measure
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iters):
            batch = next_batch()
            _ = model(batch)

            # batch size estimation
            # batch['proj'] or batch['proj_t'] normally has shape [B, C, H, W]
            if isinstance(batch, dict):
                if "proj" in batch and torch.is_tensor(batch["proj"]):
                    bs = batch["proj"].shape[0]
                elif "proj_t" in batch and torch.is_tensor(batch["proj_t"]):
                    bs = batch["proj_t"].shape[0]
                else:
                    bs = 1
            else:
                bs = 1

            total_samples += bs

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    fps = total_samples / elapsed

    return fps, elapsed, total_samples


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available, but cuda device was requested.")

    device = args.device

    cfg = Config.fromfile(args.config)

    # 固定 batch size 为用户指定值，通常 FPS 用 batch_size=1
    cfg.batch_size = args.batch_size

    # 防止可视化或保存图像影响 FPS
    if hasattr(cfg, "view"):
        cfg.view = False

    print("=" * 80)
    print("[Config]", args.config)
    print("[Checkpoint]", args.ckpt)
    print("[Device]", device)
    print("[Batch size]", args.batch_size)
    print("=" * 80)

    model = build_model(cfg, args.ckpt, device)

    total_params, trainable_params = count_params(model)
    print(f"[Params] Total Params: {total_params / 1e6:.3f} M")
    print(f"[Params] Trainable Params: {trainable_params / 1e6:.3f} M")

    # build test dataloader
    dataloader = build_dataloader(cfg.dataset.test, cfg, is_train=False)

    # example batch for FLOPs
    example_batch = next(iter(dataloader))
    example_batch = move_to_device(example_batch, device)

    flops = None
    if args.measure_flops:
        flops = try_measure_flops(model, example_batch)
        if flops is not None:
            print(f"[FLOPs] FLOPs: {flops / 1e9:.3f} G")

    fps, elapsed, samples = measure_fps(
        model=model,
        dataloader=dataloader,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    print("=" * 80)
    print("[Final Results]")
    print(f"Config: {os.path.basename(args.config)}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Params(M): {total_params / 1e6:.3f}")

    if flops is not None:
        print(f"FLOPs(G): {flops / 1e9:.3f}")
    else:
        print("FLOPs(G): N/A")

    print(f"FPS: {fps:.2f}")
    print(f"Measured samples: {samples}")
    print(f"Elapsed time: {elapsed:.3f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()