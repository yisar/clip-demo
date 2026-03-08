import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time

# --- 核心上采样启动函数 ---

def UPA(HR_img, lr_modality):
    """
    各向异性像素级联合双边上采样 (带进度日志版)
    HR_img: PIL Image (高分辨率指导图)
    lr_modality: [1, C, Hl, Wl] Tensor (低分辨率特征图)
    """
    # 强制开启梯度环境，确保在推理脚本中也能 backward
    with torch.enable_grad():
        start_time = time.time()
        USE_AMP = True
        AMP_DTYPE = torch.float16
        
        # 1. 准备数据并转为 CUDA
        hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0  
        H, W = hr.shape[-2:]
        Hl, Wl = lr_modality.shape[-2:]
        scale = int(H / Hl)
        
        # 2. 构造训练目标：用下采样的 HR 图像模拟 LR 输入
        lr_train_input = F.interpolate(hr, scale_factor=1/scale, mode="bicubic", align_corners=False)

        # 3. 初始化模型与优化器
        model = LearnablePixelwiseAnisoJBU_NoParent(Hl, Wl, scale=scale).cuda()
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=1e-1)
        max_steps = 5100 
        gamma = (1e-9 / 1e-1) ** (1.0 / max_steps)
        scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        print(f"\n[UPA] 开始即时优化 | 目标尺寸: {H}x{W} | 缩放倍率: {scale}x")

        # 4. 快速迭代优化 (通常 50 步足以让边缘收敛)
        for step in range(1, 21):
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                # 模型根据当前参数预测高分辨率图
                pred = model(lr_train_input, hr) 
                loss = F.l1_loss(pred, hr)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # 每 10 步打印一次进度
            if step % 10 == 0 or step == 1:
                mem_used = torch.cuda.memory_allocated() / 1024**2
                print(f"  > Step {step:2d}/50 | Loss: {loss.item():.6f} | GPU Mem: {mem_used:.1f}MB")

        # 5. 最终推理：使用学到的参数上采样真正的特征图
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            # 注意：lr_modality 需要转为 float32 保证模型输入兼容
            hr_feat = model(lr_modality.to(torch.float32), hr)
        
        end_time = time.time()
        print(f"[UPA] 优化完成 | 单片耗时: {end_time - start_time:.2f}s")
            
        return hr_feat

# --- 内部组件：各向异性计算逻辑 ---

@torch.no_grad()
def _build_offsets(R_max, device):
    offs = torch.arange(-R_max, R_max + 1, device=device)
    dY, dX = torch.meshgrid(offs, offs, indexing='ij')
    return dY.reshape(-1), dX.reshape(-1)

def _tanh_bound_pi(raw):
    return math.pi * torch.tanh(raw)

def gather_lr_scalar_general(map_lr, Ui, Vi):
    Hl, Wl = map_lr.shape[-2:]
    idx = (Ui * Wl + Vi).reshape(-1)
    return map_lr.view(-1).index_select(0, idx).view(Ui.shape)

def gs_jbu_aniso_noparent(feat_lr, guide_hr, scale, sx_map, sy_map, th_map, sr_map, 
                         R_max=4, alpha_dyn=2.0, C_chunk=40, Nn_chunk=10):
    """
    针对 4GB 显存优化的核心算子
    C_chunk=64: 通道分块处理，防止中间张量过大
    Nn_chunk=16: 空间邻域分块，降低显存峰值
    """
    _, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape
    dev = feat_lr.device
    dtype_feat = feat_lr.dtype

    y, x = torch.arange(Hh, device=dev).float(), torch.arange(Wh, device=dev).float()
    Y, X = torch.meshgrid(y, x, indexing='ij')
    u, v = (Y + 0.5) / scale - 0.5, (X + 0.5) / scale - 0.5
    uc, vc = torch.round(u).clamp(0, Hl-1).long(), torch.round(v).clamp(0, Wl-1).long()

    # 预计算半径图
    sigma_eff_hr = F.interpolate(torch.maximum(sx_map, sy_map), (Hh, Wh), mode='bilinear', align_corners=False)
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp_(min=1, max=R_max).long()

    dY_all, dX_all = _build_offsets(R_max, dev)
    num_s = torch.zeros(C, Hh, Wh, device=dev)
    den_s = torch.zeros(Hh, Wh, device=dev)
    m = torch.full((Hh, Wh), float("-inf"), device=dev)

    guide_lr = F.interpolate(guide_hr, size=(Hl, Wl), mode='bilinear', align_corners=False)
    feat_flat = feat_lr[0].permute(1, 2, 0).reshape(Hl*Wl, C).contiguous()

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        for n0 in range(0, dY_all.numel(), Nn_chunk):
            n1 = min(n0 + Nn_chunk, dY_all.numel())
            dY, dX = dY_all[n0:n1].view(-1,1,1), dX_all[n0:n1].view(-1,1,1)
            Bn = dY.shape[0]

            Ui, Vi = torch.clamp(uc + dY, 0, Hl-1), torch.clamp(vc + dX, 0, Wl-1)
            mask = ((dY**2 + dX**2) <= (R_map**2)).squeeze(0).squeeze(0)

            dx, dy = X - ((Vi.float()+0.5)*scale-0.5), Y - ((Ui.float()+0.5)*scale-0.5)
            sx = gather_lr_scalar_general(sx_map, Ui, Vi).clamp_min(1e-6)
            sy = gather_lr_scalar_general(sy_map, Ui, Vi).clamp_min(1e-6)
            th = gather_lr_scalar_general(th_map, Ui, Vi)
            sr = gather_lr_scalar_general(sr_map, Ui, Vi).clamp_min(1e-6)

            cos_t, sin_t = torch.cos(th), torch.sin(th)
            log_w = -((dx*cos_t + dy*sin_t)**2)/(2*sx**2 + 1e-8) - ((-dx*sin_t + dy*cos_t)**2)/(2*sy**2 + 1e-8)
            
            g_diff2 = sum((guide_hr[0, c] - gather_lr_scalar_general(guide_lr[0, c], Ui, Vi))**2 for c in range(3))
            log_w += -g_diff2 / (2.0 * sr**2 + 1e-8)
            log_w = torch.where(mask, log_w, torch.full_like(log_w, float("-inf")))

            m_chunk = torch.max(log_w, dim=0).values
            valid = torch.isfinite(m_chunk)
            if not valid.any(): continue
            
            m_new = torch.where(valid, torch.maximum(m, m_chunk), m)
            sc_old = torch.exp(m - m_new)
            den_s *= sc_old
            num_s *= sc_old.unsqueeze(0)

            s = torch.exp(log_w - m_new.unsqueeze(0)) 
            den_s += s.sum(0)
            
            idx_flat = (Ui * Wl + Vi).reshape(-1)
            for c0 in range(0, C, C_chunk):
                c1 = min(c0 + C_chunk, C)
                feat_chunk = feat_flat.index_select(0, idx_flat)[:, c0:c1].view(Bn, Hh, Wh, -1)
                num_s[c0:c1] += (feat_chunk * s.unsqueeze(-1)).sum(0).permute(2, 0, 1)
            
            m = m_new
            del Ui, Vi, log_w, s, dx, dy, mask
            if n0 % 32 == 0: torch.cuda.empty_cache()

    return (num_s / den_s.clamp_min(1e-8)).unsqueeze(0).to(dtype_feat)

# --- 模型定义 ---

class LearnablePixelwiseAnisoJBU_NoParent(nn.Module):
    def __init__(self, Hl, Wl, scale=16, init_sigma=16.0, init_sigma_r=0.12, R_max=8, alpha_dyn=2.0):
        super().__init__()
        self.scale, self.R_max, self.alpha_dyn = scale, R_max, alpha_dyn
        self.sx_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.sy_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.th_raw = nn.Parameter(torch.zeros((1, 1, Hl, Wl)))
        self.sr_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma_r))))

    def forward(self, feat_lr, guide_hr):
        return gs_jbu_aniso_noparent(
            feat_lr, guide_hr, self.scale, 
            torch.exp(self.sx_raw), torch.exp(self.sy_raw), 
            _tanh_bound_pi(self.th_raw), torch.exp(self.sr_raw), 
            R_max=self.R_max, alpha_dyn=self.alpha_dyn
        )