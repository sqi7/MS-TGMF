'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
seed = 2021
load_from = None
finetune_from = None
log_dir = './logs/baseline'
view = False

net = dict(
    type='Detector',
    head_type='row',
    loss_type='row_ce'
)

pcencoder = dict(
    type='Projector',
    resnet='resnet34',
    pretrained=False,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)
featuremap_out_channel = 64

filter_mode = 'xyz'
list_filter_roi = [0.02, 46.08, -11.52, 11.52, -2.0, 1.5]  # get rid of 0, 0 points
list_roi_xy = [0.0, 46.08, -11.52, 11.52]
list_grid_xy = [0.04, 0.02]
list_img_size_xy = [1152, 1152]

backbone = dict(
    type='VitSegNet', # GFC-T
    image_size=144,
    patch_size=8,
    channels=64,
    dim=512,
    depth=3,
    heads=16,
    output_channels=1024,
    expansion_factor=4,
    dim_head=64,
    dropout=0.,
    emb_dropout=0., # TBD
    is_with_shared_mlp=False,
)

temporal = dict(
    type='TemporalGatedMemory',
    num_history=2,          # 使用 t-2, t-1, t 三帧
    fuse_at='backbone_out', # 在 GFC backbone 输出后、head 之前做融合
    align=False,            # 第一版先不做自车运动对齐
    in_channels=1024,       # 必须与 backbone.output_channels 一致
    hidden_channels=256,
    kernel_size=3,
    use_gate=True,            # 使用门控机制控制历史信息注入
    use_residual=True,
)

use_temporal = False   # 消融实验快速开关：False=baseline，True=temporal
temporal_cfg = temporal

heads = dict(
    type='RowSharNotReducRef',
    dim_feat=8, # input feat channels
    row_size=144,
    dim_shared=512,
    lambda_cls=1.,
    thr_ext = 0.3,
    off_grid = 2,
    dim_token = 1024,
    tr_depth = 1,
    tr_heads = 16,
    tr_dim_head = 64,
    tr_mlp_dim = 2048,
    tr_dropout = 0.,
    tr_emb_dropout = 0.,
    is_reuse_same_network = False,
)

conf_thr = 0.5
view = False

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 50, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 100)
]

optimizer = dict(
  type = 'Adam', #'AdamW',
  lr = 0.0001,
)

epochs = 35
batch_size = 24
total_iter = (2904 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 5
save_ep = 5

### Setting Here ###
dataset_path = '/root/epfs/data/KLane' # '/media/donghee/HDD_0/KLane'
### Setting Here ###
dataset_type = 'KLane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        mode_item='pc',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        mode_item='pc',
    )
)
workers=24


# Row-wise Smoothness / Continuity Loss
# 用于约束同一条车道线在相邻 row 上的位置变化不要剧烈跳变
use_smooth_loss = False   # 消融实验快速开关：False=不使用，True=使用
smooth_loss_weight = 0.005
smooth_loss_stage = 'second'   # first / second / both，默认只约束第二阶段输出


use_ms_context = False   # 消融实验快速开关：False=不使用，True=使用
ms_context = dict(
    in_channels=8,          # 多尺度模块接在 head 前，此处特征通道数是 8，需要与 heads.dim_feat 保持一致
    hidden_channels=16,
    dilation_rates=[1, 2, 4],
    use_residual=True,
)

# Merging-aware Loss Reweighting
# 该策略在 10 epoch 预实验中对 merging 指标没有收益，当前 DCRB 实验先关闭。
use_scene_loss_weight = False
scene_loss_weight = dict(
    merging=1.0,
    curve=1.0,
)

# Directional Context Refinement Block, DCRB
# 插入在 MSCE 之后、RowRef Head 之前，通过 1×k 与 k×1 条带卷积增强方向上下文，重点改善 merging / curve 等复杂结构场景。
use_dcrb = False   # 消融实验快速开关：False=不使用，True=使用
dcrb = dict(
    in_channels=8,
    hidden_channels=16,
    kernel_size=7,
    use_residual=True,
    alpha=0.7,              # DCRB residual scaling: F_out = F_in + alpha * DCRB(F_in)
)






# 第一步：等当前这轮跑完。
# 第二步：跑 Baseline 60轮，保证对照公平。
# 第三步：跑 TGMF+MSCE+DCRB 60轮，作为完整模型。
# 第四步：如果时间够，再跑 TGMF+MSCE 60轮，作为遮挡最优对照。
# 当前配置：MSCE + DCRB(alpha=0.7)，用于验证降低 DCRB 残差分支强度是否能改善 Occ4-6 / Total。