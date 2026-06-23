import os
import sys
from pathlib import Path
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "/root/epfs/K-Lane-main_时序")

from baseline.utils.config import Config
from baseline.engine.runner import Runner
from baseline.datasets.registry import build_dataloader


# =========================
# 1. 路径配置
# =========================

DATA_ROOT = "/root/epfs/data/KLane"
DESC_PATH = "/root/epfs/data/KLane/description_frames_test.txt"

OUT_ROOT = "/root/epfs/K-Lane-main_时序/vis_compare_paper"

PER_CLASS_LIMIT = 30

TARGET_TAGS = [
    "day", "night",
    "urban", "highway",
    "normal", "curve", "gentlecurve", "sharpcurve", "merging",
    "occ0", "occ1", "occ2", "occ3", "occ4", "occ5", "occ6", "occ456",
]


# =========================
# 2. 需要并列可视化的模型
#    你后面只需要改这里的 ckpt 路径
# =========================

MODEL_SPECS = [
    {
        "name": "Baseline",
        "config": "/root/epfs/K-Lane-main_时序/configs/Proj28_GFC-T3_Baseline_seed3407.py",
        "ckpt": "/root/epfs/K-Lane-main_时序/logs/你的baseline目录/日期/ckpt/best.pth",
        "use_temporal": False,
        "use_ms_context": False,
        "use_dcrb": False,
    },
    {
        "name": "TGMF",
        "config": "/root/epfs/K-Lane-main_时序/configs/Proj28_GFC-T3_TGMF_seed3407.py",
        "ckpt": "/root/epfs/K-Lane-main_时序/logs/你的tgmf目录/日期/ckpt/best.pth",
        "use_temporal": True,
        "use_ms_context": False,
        "use_dcrb": False,
    },
    {
        "name": "TGMF+MSCE",
        "config": "/root/epfs/K-Lane-main_时序/configs/Proj28_GFC-T3_TGMF_MSCE_seed3407.py",
        "ckpt": "/root/epfs/K-Lane-main_时序/logs/你的tgmf_msce目录/日期/ckpt/best.pth",
        "use_temporal": True,
        "use_ms_context": True,
        "use_dcrb": False,
    },
    {
        "name": "TGMF+MSCE+DCRB",
        "config": "/root/epfs/K-Lane-main_时序/configs/Proj28_GFC-T3_TGMF_MSCE_DCRB_seed3407.py",
        "ckpt": "/root/epfs/K-Lane-main_时序/logs/你的dcrb目录/日期/ckpt/best.pth",
        "use_temporal": True,
        "use_ms_context": True,
        "use_dcrb": True,
    },
]


# =========================
# 3. 描述文件与标签处理
# =========================

def load_desc_map(desc_path):
    desc_map = {}
    p = Path(desc_path)

    if not p.exists():
        print(f"[WARN] 找不到测试集描述文件: {desc_path}")
        return desc_map

    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        ts = line.split(",")[0].strip()
        desc_map[ts] = line

    return desc_map


def normalize_tag(s):
    s = s.strip().lower()
    s = s.replace(" ", "")
    s = s.replace("-", "")
    s = s.replace("_", "")
    return s


def extract_tags(desc_line):
    if not desc_line:
        return ["unknown"]

    raw_parts = [p.strip() for p in desc_line.split(",")]
    parts = [normalize_tag(p) for p in raw_parts[1:]]
    joined = " ".join(parts)
    raw_lower = desc_line.lower()

    tags = []

    if any("night" in p for p in parts):
        tags.append("night")

    if any(("day" in p) or ("daylight" in p) for p in parts):
        tags.append("day")

    if any("urban" in p for p in parts):
        tags.append("urban")

    if any("highway" in p for p in parts):
        tags.append("highway")

    if any("merging" in p for p in parts):
        tags.append("merging")

    if any("normal" in p for p in parts):
        tags.append("normal")

    if any("gentlecurve" in p for p in parts):
        tags.append("gentlecurve")
        tags.append("curve")
    elif any("sharpcurve" in p for p in parts):
        tags.append("sharpcurve")
        tags.append("curve")
    elif any("curve" in p for p in parts):
        tags.append("curve")

    occ_tag = None

    for k in range(7):
        if (
            f"occ{k}" in joined
            or f"occlusion{k}" in joined
            or f"occlusion {k}" in raw_lower
            or f"occ {k}" in raw_lower
        ):
            occ_tag = f"occ{k}"
            break

    for p in parts:
        if p in ["0", "1", "2", "3", "4", "5", "6"]:
            occ_tag = f"occ{p}"

    if occ_tag is not None:
        tags.append(occ_tag)

        if occ_tag in ["occ4", "occ5", "occ6"]:
            tags.append("occ456")
    else:
        tags.append("unknown_occ")

    final_tags = []
    seen = set()

    for t in tags:
        if t not in seen:
            seen.add(t)
            final_tags.append(t)

    return final_tags if final_tags else ["unknown"]


# =========================
# 4. 模型加载
# =========================

def check_file(path, name):
    if not Path(path).exists():
        raise FileNotFoundError(f"[ERROR] {name} 不存在: {path}")


def load_runner(model_spec):
    config_path = model_spec["config"]
    ckpt_path = model_spec["ckpt"]

    check_file(config_path, "config")
    check_file(ckpt_path, "ckpt")

    cfg = Config.fromfile(config_path)

    cfg.dataset_path = DATA_ROOT
    cfg.view = True
    cfg.gpus = 1
    cfg.is_eval_conditional = False

    # 强制覆盖模块开关，避免 config 里写错
    cfg.use_temporal = bool(model_spec.get("use_temporal", False))
    cfg.use_ms_context = bool(model_spec.get("use_ms_context", False))
    cfg.use_dcrb = bool(model_spec.get("use_dcrb", False))

    if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "test"):
        cfg.dataset.test.data_root = DATA_ROOT
        cfg.dataset.test.mode_item = "pc"

    runner = Runner(cfg)
    runner.load_ckpt(ckpt_path)
    runner.net.cuda()
    runner.net.eval()

    print(
        f"[LOAD] {model_spec['name']} | "
        f"TGMF={cfg.use_temporal}, MSCE={cfg.use_ms_context}, DCRB={cfg.use_dcrb}"
    )
    print(f"       config: {config_path}")
    print(f"       ckpt  : {ckpt_path}")

    return cfg, runner


def build_test_loader(reference_cfg):
    reference_cfg.view = True
    reference_cfg.is_eval_conditional = False
    reference_cfg.dataset_path = DATA_ROOT
    reference_cfg.batch_size = 1
    reference_cfg.workers = 4

    reference_cfg.dataset.test.data_root = DATA_ROOT
    reference_cfg.dataset.test.mode_item = "pc"

    test_loader = build_dataloader(
        reference_cfg.dataset.test,
        reference_cfg,
        is_train=False,
    )

    return test_loader


# =========================
# 5. Batch / 图像处理
# =========================

def move_to_cuda(batch):
    ret = {}

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            ret[k] = v.cuda(non_blocking=True)
        else:
            ret[k] = v

    return ret


def add_title(img, title, size=432):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    title_h = 50
    canvas = np.ones((size + title_h, size, 3), dtype=np.uint8) * 255
    canvas[title_h:, :, :] = img

    cv2.putText(
        canvas,
        title,
        (15, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def get_pred_and_gt(output, idx):
    lane_maps = output["lane_maps"]

    pred = lane_maps["rgb_conf_cls_idx"][idx]
    gt = lane_maps["rgb_cls_label"][idx]

    return np.ascontiguousarray(pred), np.ascontiguousarray(gt)


def get_timestamp_from_batch(batch, idx, batch_idx):
    meta = batch.get("meta", None)

    if isinstance(meta, dict) and "bev_tensor_label" in meta:
        name = os.path.basename(meta["bev_tensor_label"][idx])
        m = re.search(r"(\d+)", name)

        if m:
            return m.group(1)

    return f"batch{batch_idx:04d}_{idx}"


def make_compare_image(pred_dict, gt):
    panels = []

    for name, pred in pred_dict.items():
        panels.append(add_title(pred, name))

    panels.append(add_title(gt, "Ground Truth"))

    return np.concatenate(panels, axis=1)


def all_targets_full(counter):
    for tag in TARGET_TAGS:
        if counter[tag] < PER_CLASS_LIMIT:
            return False
    return True


def save_image_by_tags(compare_img, ts, tags, counter):
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    saved_any = False

    prefix = "_".join(tags)
    filename = f"{prefix}_{ts}_compare.png"

    for tag in tags:
        if tag not in TARGET_TAGS:
            continue

        if counter[tag] >= PER_CLASS_LIMIT:
            continue

        tag_dir = out_root / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        save_path = tag_dir / filename
        cv2.imwrite(str(save_path), compare_img)

        counter[tag] += 1
        saved_any = True

    if saved_any:
        all_dir = out_root / "all_renamed"
        all_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(all_dir / filename), compare_img)

    return saved_any, filename


# =========================
# 6. 主函数
# =========================

def main():
    desc_map = load_desc_map(DESC_PATH)

    # 加载所有模型
    loaded_models = []
    reference_cfg = None

    for spec in MODEL_SPECS:
        cfg, runner = load_runner(spec)
        loaded_models.append((spec["name"], cfg, runner))

        # 优先用时序模型的 cfg 构造 dataloader，确保 batch 内有 proj_t/proj_tm1/proj_tm2
        if reference_cfg is None or bool(spec.get("use_temporal", False)):
            reference_cfg = cfg

    test_loader = build_test_loader(reference_cfg)

    counter = defaultdict(int)
    total_saved = 0

    print("\n========== Start Visualization ==========")
    print(f"Output root: {OUT_ROOT}")
    print(f"Per class limit: {PER_CLASS_LIMIT}")
    print("Models:", [m[0] for m in loaded_models])

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch = move_to_cuda(batch)

            outputs = {}

            for name, _, runner in loaded_models:
                outputs[name] = runner.net(batch)

            # 所有模型输出 batch size 应该一致
            first_output = next(iter(outputs.values()))
            bs = len(first_output["lane_maps"]["rgb_cls_label"])

            for i in range(bs):
                ts = get_timestamp_from_batch(batch, i, batch_idx)
                desc_line = desc_map.get(ts, "")
                tags = extract_tags(desc_line)

                useful_tags = [
                    t for t in tags
                    if t in TARGET_TAGS and counter[t] < PER_CLASS_LIMIT
                ]

                if not useful_tags:
                    continue

                pred_dict = {}
                gt = None

                for name, output in outputs.items():
                    pred, cur_gt = get_pred_and_gt(output, i)
                    pred_dict[name] = pred

                    if gt is None:
                        gt = cur_gt

                compare = make_compare_image(pred_dict, gt)
                saved, filename = save_image_by_tags(compare, ts, tags, counter)

                if saved:
                    total_saved += 1
                    print(f"[INFO] saved {total_saved}: {filename}")
                    print("[COUNT]", dict(counter))

            if all_targets_full(counter):
                print("[DONE] 每个目标类别都已保存到上限")
                break

    print("\n========== Final Counts ==========")

    for tag in TARGET_TAGS:
        print(f"{tag}: {counter[tag]}")

    print(f"output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()