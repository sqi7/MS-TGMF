import pickle
import argparse
import numpy as np
import cv2
from pathlib import Path


def normalize_to_uint8(x):
    x = np.asarray(x).astype(np.float32)
    x = np.nan_to_num(x)
    if x.max() - x.min() < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x.min()) / (x.max() - x.min())
    return (x * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="occupancy", choices=["occupancy", "feature"])
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        obj = pickle.load(f)

    feat = np.asarray(obj[0])[0]    # (20736, 32, 7)
    coord = np.asarray(obj[1])[0]   # (20736, 3)

    print("feat shape:", feat.shape)
    print("coord shape:", coord.shape)
    print("coord min:", coord.min(axis=0), "coord max:", coord.max(axis=0))

    H = W = 144
    bev = np.zeros((H, W), dtype=np.float32)

    # coord 一般是 [batch/z?, y, x] 或 [z?, y, x]
    # 先尝试取后两维作为 y, x
    ys = coord[:, 1].astype(np.int32)
    xs = coord[:, 2].astype(np.int32)

    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    if args.mode == "occupancy":
        bev[ys[valid], xs[valid]] = 1.0
    else:
        # 只统计非零点特征，减少填充噪声
        point_valid = np.abs(feat).sum(axis=2) > 1e-6  # (20736, 32)
        score = point_valid.sum(axis=1).astype(np.float32)
        bev[ys[valid], xs[valid]] = score[valid]

    img = normalize_to_uint8(bev)

    # 加粗一点，论文图里更清楚
    img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)

    # 放大
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    # 论文图建议用灰度或蓝色系，别用太花的 jet
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 黑底白点更像 BEV pseudo-image
    # 如果方向反了，可以注释/取消下面这句
    img_color = cv2.flip(img_color, 0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img_color)
    print("saved:", out)


if __name__ == "__main__":
    main()
