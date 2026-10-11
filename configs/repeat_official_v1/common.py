"""Shared official Train/Test protocol for repeated-seed experiments.

Important: dataset.test is the official test set. Runner.validate() uses this
set during training and selects best.pth by its confidence F1. It is not an
independent validation set.
"""

load_from = None
finetune_from = None
view = False

net = dict(type='Detector', head_type='row', loss_type='row_ce')

pcencoder = dict(
    type='Projector',
    resnet='resnet34',
    pretrained=False,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1],
)
featuremap_out_channel = 64

filter_mode = 'xyz'
list_filter_roi = [0.02, 46.08, -11.52, 11.52, -2.0, 1.5]
list_roi_xy = [0.0, 46.08, -11.52, 11.52]
list_grid_xy = [0.04, 0.02]
list_img_size_xy = [1152, 1152]

backbone = dict(
    type='VitSegNet',
    image_size=144,
    patch_size=8,
    channels=64,
    dim=512,
    depth=3,
    heads=16,
    output_channels=1024,
    expansion_factor=4,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    is_with_shared_mlp=False,
)

temporal = dict(
    type='TemporalGatedMemory',
    num_history=2,
    fuse_at='backbone_out',
    align=False,
    # VitSegNet emits dim / patch_size^2 = 8 channels when shared MLP is off.
    in_channels=8,
    hidden_channels=256,
    kernel_size=3,
    use_gate=True,
    use_residual=True,
)
temporal_cfg = temporal

heads = dict(
    type='RowSharNotReducRef',
    dim_feat=8,
    row_size=144,
    dim_shared=512,
    lambda_cls=1.0,
    thr_ext=0.3,
    off_grid=2,
    dim_token=1024,
    tr_depth=1,
    tr_heads=16,
    tr_dim_head=64,
    tr_mlp_dim=2048,
    tr_dropout=0.0,
    tr_emb_dropout=0.0,
    is_reuse_same_network=False,
)

conf_thr = 0.5
cls_lane_color = [
    (0, 0, 255), (0, 50, 255), (0, 255, 255),
    (0, 255, 0), (255, 0, 0), (255, 0, 100),
]

optimizer = dict(type='Adam', lr=0.0001)
epochs = 35
batch_size = 24
# Kept identical to the verified 35-epoch legacy runs.
total_iter = (2904 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)
eval_ep = 5
save_ep = 5

dataset_path = '/root/epfs/data/KLane'
dataset_type = 'KLane'
dataset = dict(
    train=dict(type=dataset_type, data_root=dataset_path, split='train', mode_item='pc'),
    test=dict(type=dataset_type, data_root=dataset_path, split='test', mode_item='pc'),
)
workers = 24

use_smooth_loss = False
smooth_loss_weight = 0.005
smooth_loss_stage = 'second'

ms_context = dict(
    in_channels=8,
    hidden_channels=16,
    dilation_rates=[1, 2, 4],
    use_residual=True,
)

use_scene_loss_weight = False
scene_loss_weight = dict(merging=1.0, curve=1.0)

dcrb = dict(
    in_channels=8,
    hidden_channels=16,
    kernel_size=7,
    use_residual=True,
    alpha=0.7,
)

# Protocol metadata consumed by the auditable launcher.
protocol_name = 'official_train_test_repeat_v1'
selection_split_name = 'official test (not an independent validation set)'
selection_log_name = 'test_selection.txt'
cudnn_benchmark = True
cudnn_deterministic = False
