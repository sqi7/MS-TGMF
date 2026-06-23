import torch.nn as nn
import torch

from baseline.models.registry import NET
from ..registry import build_pcencoder, build_backbone, build_heads


class TemporalGatedMemory(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=256, kernel_size=3,
                 use_gate=True, use_residual=True):
        super(TemporalGatedMemory, self).__init__()
        padding = kernel_size // 2
        self.use_gate = use_gate
        self.use_residual = use_residual

        self.hist_reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def forward(self, feat_t, feat_tm1, feat_tm2):
        hist = torch.cat([feat_tm2, feat_tm1], dim=1)
        hist = self.hist_reduce(hist)

        if self.use_gate:
            gate = self.gate(torch.cat([feat_t, hist], dim=1))
            fused = gate * hist
        else:
            fused = hist

        if self.use_residual:
            return feat_t + fused
        return fused


# ---------------- 多尺度上下文模块 ----------------
class MultiScaleContextBlock(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=256,
                 dilation_rates=(1, 2, 4), use_residual=True):
        super(MultiScaleContextBlock, self).__init__()
        self.use_residual = use_residual

        self.branches = nn.ModuleList()

        # 1x1 分支：保留局部通道信息
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        ))

        # 空洞卷积分支：用不同感受野提取多尺度上下文
        for rate in dilation_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ))

        num_branches = len(dilation_rates) + 1
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * num_branches, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        multi_scale_feats = [branch(feat) for branch in self.branches]
        fused = self.fuse(torch.cat(multi_scale_feats, dim=1))

        if self.use_residual:
            return feat + fused
        return fused


# ---------------- 方向上下文精炼模块 ----------------
class DirectionalContextRefinementBlock(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=16, kernel_size=7,
                 use_residual=True, alpha=1.0):
        super(DirectionalContextRefinementBlock, self).__init__()
        self.use_residual = use_residual
        self.alpha = alpha
        padding = kernel_size // 2

        self.refine = nn.Sequential(
            # 横向条带卷积：增强车道间关系与合流/分流横向拓扑感知
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, kernel_size),
                      padding=(0, padding), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 纵向条带卷积：增强车道线沿行方向的连续性与长距离结构传播
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(kernel_size, 1),
                      padding=(padding, 0), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 通道融合并恢复到输入通道数
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        refined = self.refine(feat)
        if self.use_residual:
            return feat + self.alpha * refined
        return refined


@NET.register_module
class Detector(nn.Module):
    def __init__(self,
                head_type='seg',
                loss_type='row_ce',
                cfg=None):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(144, 144, 3, 3)
        self.pcencoder = build_pcencoder(cfg)
        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg)
        self.head_type = head_type
        self.loss_type = loss_type

        self.use_temporal = bool(getattr(cfg, 'use_temporal', False))
        self.temporal_module = None
        if self.use_temporal:
            temporal_cfg = getattr(cfg, 'temporal_cfg', None)
            if temporal_cfg is None:
                temporal_cfg = getattr(cfg, 'temporal', None)
            if temporal_cfg is None:
                raise ValueError('当 use_temporal=True 时，配置中必须提供 temporal_cfg 或 temporal')

            if isinstance(temporal_cfg, dict):
                in_channels = temporal_cfg.get('in_channels', 1024)
                hidden_channels = temporal_cfg.get('hidden_channels', 256)
                kernel_size = temporal_cfg.get('kernel_size', 3)
                use_gate = temporal_cfg.get('use_gate', True)
                use_residual = temporal_cfg.get('use_residual', True)
            else:
                in_channels = getattr(temporal_cfg, 'in_channels', 1024)
                hidden_channels = getattr(temporal_cfg, 'hidden_channels', 256)
                kernel_size = getattr(temporal_cfg, 'kernel_size', 3)
                use_gate = getattr(temporal_cfg, 'use_gate', True)
                use_residual = getattr(temporal_cfg, 'use_residual', True)

            self.temporal_module = TemporalGatedMemory(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                use_gate=use_gate,
                use_residual=use_residual,
            )

        self.use_ms_context = bool(getattr(cfg, 'use_ms_context', False))
        self.ms_context_module = None
        if self.use_ms_context:
            ms_cfg = getattr(cfg, 'ms_context', None)
            if ms_cfg is None:
                raise ValueError('当 use_ms_context=True 时，配置中必须提供 ms_context')

            if isinstance(ms_cfg, dict):
                in_channels = ms_cfg.get('in_channels', 1024)
                hidden_channels = ms_cfg.get('hidden_channels', 256)
                dilation_rates = ms_cfg.get('dilation_rates', [1, 2, 4])
                use_residual = ms_cfg.get('use_residual', True)
            else:
                in_channels = getattr(ms_cfg, 'in_channels', 1024)
                hidden_channels = getattr(ms_cfg, 'hidden_channels', 256)
                dilation_rates = getattr(ms_cfg, 'dilation_rates', [1, 2, 4])
                use_residual = getattr(ms_cfg, 'use_residual', True)

            self.ms_context_module = MultiScaleContextBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                dilation_rates=dilation_rates,
                use_residual=use_residual,
            )

        self.use_dcrb = bool(getattr(cfg, 'use_dcrb', False))
        self.dcrb_module = None
        if self.use_dcrb:
            dcrb_cfg = getattr(cfg, 'dcrb', None)
            if dcrb_cfg is None:
                raise ValueError('当 use_dcrb=True 时，配置中必须提供 dcrb')

            if isinstance(dcrb_cfg, dict):
                in_channels = dcrb_cfg.get('in_channels', 8)
                hidden_channels = dcrb_cfg.get('hidden_channels', 16)
                kernel_size = dcrb_cfg.get('kernel_size', 7)
                use_residual = dcrb_cfg.get('use_residual', True)
                alpha = dcrb_cfg.get('alpha', 1.0)
            else:
                in_channels = getattr(dcrb_cfg, 'in_channels', 8)
                hidden_channels = getattr(dcrb_cfg, 'hidden_channels', 16)
                kernel_size = getattr(dcrb_cfg, 'kernel_size', 7)
                use_residual = getattr(dcrb_cfg, 'use_residual', True)
                alpha = getattr(dcrb_cfg, 'alpha', 1.0)

            self.dcrb_module = DirectionalContextRefinementBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                use_residual=use_residual,
                alpha=alpha,
            )

    def encode_single_proj(self, batch, proj_key='proj', is_get_features=False):
        temp_batch = batch.copy()
        temp_batch['proj'] = batch[proj_key]
        fea = self.pcencoder(temp_batch)

        if is_get_features:
            fea, list_features = self.backbone(fea, True)
            return fea, list_features
        else:
            fea = self.backbone(fea)
            return fea, None

    def forward(self, batch, is_get_features=False):
        output = {}

        if (not self.use_temporal) or ('proj_t' not in batch):
            fea = self.pcencoder(batch)

            if is_get_features:
                fea, list_features = self.backbone(fea, True)
                output.update({'features': list_features})
            else:
                fea = self.backbone(fea)
        else:
            fea_t, list_features = self.encode_single_proj(batch, proj_key='proj_t', is_get_features=is_get_features)
            fea_tm1, _ = self.encode_single_proj(batch, proj_key='proj_tm1', is_get_features=False)
            fea_tm2, _ = self.encode_single_proj(batch, proj_key='proj_tm2', is_get_features=False)
            fea = self.temporal_module(fea_t, fea_tm1, fea_tm2)

            if is_get_features:
                output.update({'features': list_features})

        if self.use_ms_context:
            fea = self.ms_context_module(fea)

        if self.use_dcrb:
            fea = self.dcrb_module(fea)

        if self.training:
            out = self.heads(fea)
            output.update(self.heads.loss(out, batch, self.loss_type))
        else:
            out = self.heads(fea)
            if self.head_type == 'seg':
                output.update({'conf': out[:,7,:,:], 'cls': out[:,:7,:,:]})
                output.update({
                    'lane_maps': self.heads.get_lane_map_numpy_with_label(
                                        output, batch, is_img=self.cfg.view)})
            elif self.head_type == 'row':
                output.update(self.heads.get_conf_and_cls_dict(out, batch))
                output.update({
                    'lane_maps': self.heads.get_lane_map_numpy_with_label(
                                        output, batch, is_img=self.cfg.view)})

        return output
