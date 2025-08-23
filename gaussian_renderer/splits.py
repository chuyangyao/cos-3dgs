"""
gaussian_renderer/splits.py
实现基于余弦调制的“精确且高效”的分裂计算（不依赖CUDA修改）

调制函数：cos(wave · (x - mu))，其中 wave 为三维向量。
思路：
- 沿 wave 方向定义 1D 变量 t，原高斯在该方向的包络为 exp(-t^2/(2σ_u^2))，σ_u 为该方向有效方差的平方根
- 被调制函数 f(t) = cos(ω t) * exp(-t^2/(2σ_u^2))，其中 ω = ||wave||
- 近似用多个局部高斯在驻点 t_m 附近匹配（位置 t_m、曲率匹配得到 σ_m、幅值由 |f(t_m)| 给出）
- 三维中采用“各向同性缩放因子”r_m，使沿 u 的方差匹配（无须旋转对齐，既快又稳定）
"""

import torch
import math
from typing import Optional, Dict

from utils.general_utils import build_scaling_rotation, build_rotation


def _newton_refine_t(omega: torch.Tensor, sigma_u: torch.Tensor, t0: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """
    使用少步牛顿法精炼驻点：解 tan(ω t) = - t / (ω σ_u^2)
    g(t) = tan(ω t) + t/(ω σ_u^2)；g'(t) = ω sec^2(ω t) + 1/(ω σ_u^2)
    限制步数确保数值稳定与速度。
    """
    t = t0.clone()
    # 避免除零
    eps = 1e-8
    for _ in range(steps):
        wt = omega * t
        # 处理接近 π/2 奇异点，夹取
        wt = torch.clamp(wt, -math.pi * 0.49, math.pi * 0.49)
        tan_wt = torch.tan(wt)
        sec2_wt = 1.0 / (torch.cos(wt) ** 2 + eps)
        g = tan_wt + t / (omega * (sigma_u ** 2) + eps)
        gp = omega * sec2_wt + 1.0 / (omega * (sigma_u ** 2) + eps)
        t = t - g / (gp + eps)
    return t


def compute_splits_precise(pc,
                           iteration: int = 0,
                           max_iteration: int = 40000,
                           max_k: Optional[int] = None,
                           t_sigma_range: float = None,
                           min_wave_norm: float = 0.05) -> Optional[Dict]:
    """
    精确但高效的分裂计算（无梯度），返回与现有管线兼容的 split_data。

    - max_k: 每个高斯的最大 k（最终 2k+1 个分裂）。若为 None，则读取 pc._max_splits 或默认 5。
    - t_sigma_range: 仅考虑 |t| <= t_sigma_range * σ_u 的驻点，控制速度与能量集中。
    - min_wave_norm: wave 范数过小则不分裂。
    """
    if not hasattr(pc, '_wave'):
        return None

    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rotation = pc.get_rotation
    features_dc = pc._features_dc
    features_rest = pc._features_rest
    wave = pc._wave

    N = xyz.shape[0]
    device = xyz.device

    wave_norm = torch.norm(wave, dim=1)
    # 读取更低的最小阈值
    try:
        from config_manager import config_manager
        cfg = config_manager.config
        min_wave_norm = getattr(cfg, 'split_min_wave_norm', min_wave_norm)
    except Exception:
        pass
    # 不再因全局阈值过小而直接返回 None，始终产出至少中心驻点（t=0）的近似，确保对 wave 可微

    omega = wave_norm  # ω = ||wave||
    u = wave / (wave_norm.unsqueeze(1) + 1e-8)  # 方向单位向量

    # 计算每个高斯沿 u 的有效标准差 σ_u：σ_u^2 = u^T Σ u，且 Σ = L L^T
    # 其中 L = build_scaling_rotation(s, q)，torch 实现方便：σ_u = || L^T u ||
    L = build_scaling_rotation(scaling, rotation)  # [N, 3, 3]
    Lt_u = torch.bmm(L.transpose(1, 2), u.unsqueeze(2)).squeeze(2)  # [N,3] × [N,3,1]
    sigma_u = Lt_u.norm(dim=1) + 1e-8  # [N]

    # 进度控制：前半程降低最大分裂数；读取速度/精度平衡配置
    if max_k is None:
        max_k = getattr(pc, '_max_splits', 5)
    progress = min(float(iteration) / float(max_iteration) if max_iteration > 0 else 1.0, 1.0)
    eff_max_k = max(1, int(round(max_k * (0.5 + 0.5 * progress))))
    # 读取配置
    try:
        from config_manager import config_manager
        cfg = config_manager.config
        steps = getattr(cfg, 'split_newton_steps', 1)
        fast_align = getattr(cfg, 'split_fast_align', True)
        if t_sigma_range is None:
            t_sigma_range = getattr(cfg, 'split_t_sigma_range', 1.5)
    except Exception:
        steps = 1
        fast_align = True
        if t_sigma_range is None:
            t_sigma_range = 1.5

    # 每个高斯的 M_i：|m| ≤ floor( ω σ_u t_sigma_range / π )，并进行子集筛选（仅对高wave计算）
    Mi = torch.floor((omega * sigma_u * t_sigma_range) / math.pi).to(torch.long)
    Mi = torch.clamp(Mi, min=0, max=eff_max_k)

    # 不再早退：即便 Mi 全为 0，也走中心驻点路径，保持可微与梯度回传

    # 子集选择：仅对 Mi>0 且 wave_norm 排名前 top_ratio 的点进行分裂
    try:
        from config_manager import config_manager
        cfg = config_manager.config
        top_ratio = getattr(cfg, 'split_top_ratio', 0.2)  # 默认仅对前20%高wave的点分裂
    except Exception:
        top_ratio = 0.2
    active_mask = Mi > 0
    if active_mask.any():
        active_indices = torch.where(active_mask)[0]
        # 依据 wave_norm 排序
        active_wave = wave_norm[active_indices]
        k = max(1, int(active_indices.numel() * top_ratio))
        topk = torch.topk(active_wave, k=k, largest=True).indices
        keep_indices = active_indices[topk]
        drop_mask = active_mask.clone()
        drop_mask[:] = False
        drop_mask[keep_indices] = True
        # 非保留的激活点取消分裂
        Mi[active_mask & (~drop_mask)] = 0
    # 每次调用的全局候选上限（按 wave 排序筛选），避免一次性过大
    try:
        from config_manager import config_manager
        cfg = config_manager.config
        max_active = getattr(cfg, 'split_max_active_per_call', 0)
        if max_active and max_active > 0:
            # 选出 wave 较大的前 max_active 个 index 保留其 Mi，其余置 0
            nonzero_idx = torch.where(Mi > 0)[0]
            if nonzero_idx.numel() > max_active:
                chosen = torch.topk(wave_norm[nonzero_idx], k=max_active, largest=True).indices
                keep = nonzero_idx[chosen]
                mask_keep = torch.zeros_like(Mi, dtype=torch.bool)
                mask_keep[keep] = True
                Mi[~mask_keep] = 0
    except Exception:
        pass

    # 预计算每个高斯的分裂数: 2*Mi + 1
    n_splits_per = 2 * Mi + 1
    total_splits = int(n_splits_per.sum().item())
    # 早退1：若所有 Mi==0（仅中心项），且N很大，直接跳过，避免无意义复制
    try:
        from config_manager import config_manager
        cfg = config_manager.config
        skip_center_minN = getattr(cfg, 'split_skip_center_minN', 500000)
        max_extra_ratio = getattr(cfg, 'split_max_extra_ratio', 0.15)
    except Exception:
        skip_center_minN = 500000
        max_extra_ratio = 0.15
    if torch.count_nonzero(Mi).item() == 0 and N >= skip_center_minN:
        return None
    # OOM防护：若本次分裂会导致过多新增点，直接跳过（返回None）
    max_allow = int((1.0 + max_extra_ratio) * N)
    if total_splits > max_allow:
        return None

    # 以可微方式构造输出：收集列表，最后cat
    list_xyz = []
    list_opacity = []
    list_scaling = []
    list_rotation = []
    list_fdc = []
    list_frest = []
    list_orig = []

    for i in range(N):
        k_i = int(Mi[i].item())
        if k_i == 0:
            # 仅使用中心驻点 t=0，计算依赖 ω 的 σ_m，从而对 wave 产生梯度
            om = omega[i].clamp_min(1e-8)
            sig_u = sigma_u[i].clamp_min(1e-8)
            t_sel = torch.zeros(1, device=device, dtype=xyz.dtype)
            om_b = torch.ones_like(t_sel) * om
            sig_b = torch.ones_like(t_sel) * sig_u
            wt = om_b * t_sel
            envelope = torch.exp(-0.5 * (t_sel ** 2) / (sig_b ** 2))
            cos_wt = torch.cos(wt)
            sin_wt = torch.sin(wt)
            f_t = cos_wt * envelope
            sigma2 = sig_b ** 2
            term1 = - (om_b ** 2) * cos_wt
            term2 = ((t_sel ** 2) / (sigma2 ** 2) - 1.0 / sigma2) * cos_wt
            term3 = (2.0 * om_b * t_sel / sigma2) * sin_wt
            f2_t = (term1 + term2 + term3) * envelope
            eps = 1e-8
            sigma_m = torch.sqrt(torch.clamp(torch.abs(f_t) / (torch.abs(f2_t) + eps), min=1e-6))

            u_i = u[i]
            xyz_base = xyz[i]
            scaling_base = scaling[i]
            rotation_base = rotation[i]
            feat_dc_base = features_dc[i]
            feat_rest_base = features_rest[i]

            # 快速或精确对齐
            try:
                from config_manager import config_manager
                fast_align = getattr(config_manager.config, 'split_fast_align', True)
            except Exception:
                fast_align = True
            if fast_align:
                R_i = build_rotation(rotation_base.unsqueeze(0))[0]
                axes = [R_i[:, 0], R_i[:, 1], R_i[:, 2]]
                dots = torch.stack([torch.abs(torch.dot(u_i, a)) for a in axes])
                main_axis = int(torch.argmax(dots).item())
                L_i = build_scaling_rotation(scaling_base.unsqueeze(0), rotation_base.unsqueeze(0))[0]
                sigma_other = []
                for j in range(3):
                    if j == main_axis:
                        continue
                    vj = axes[j]
                    sigma_other.append(torch.norm(L_i.transpose(0, 1) @ vj, p=2))
                s = [None, None, None]
                s[main_axis] = sigma_m
                idx_other = 0
                for j in range(3):
                    if j == main_axis:
                        continue
                    s[j] = torch.ones_like(sigma_m) * sigma_other[idx_other]
                    idx_other += 1
                scaling_i = torch.stack(s, dim=1)
                rotation_i = rotation_base.unsqueeze(0)
            else:
                R_i = build_rotation(rotation_base.unsqueeze(0))[0]
                cols = [R_i[:, 0], R_i[:, 1], R_i[:, 2]]
                dots = [torch.abs(torch.dot(u_i, c)) for c in cols]
                aux_idx = int(torch.tensor(dots).argmin().item())
                aux_axis = cols[aux_idx]
                v1 = u_i / (u_i.norm() + 1e-8)
                v2 = aux_axis - torch.dot(aux_axis, v1) * v1
                v2 = v2 / (v2.norm() + 1e-8)
                v3 = torch.linalg.cross(v1, v2)
                R_prime = torch.stack([v1, v2, v3], dim=1)
                L_i = build_scaling_rotation(scaling_base.unsqueeze(0), rotation_base.unsqueeze(0))[0]
                sigma_v2 = torch.norm(L_i.transpose(0, 1) @ v2, p=2)
                sigma_v3 = torch.norm(L_i.transpose(0, 1) @ v3, p=2)
                def _matrix_to_quaternion_single(Rm: torch.Tensor) -> torch.Tensor:
                    m00 = Rm[0, 0]; m11 = Rm[1, 1]; m22 = Rm[2, 2]
                    trace = m00 + m11 + m22
                    if trace > 0:
                        S = torch.sqrt(trace + 1.0) * 2.0
                        qw = 0.25 * S
                        qx = (Rm[2, 1] - Rm[1, 2]) / S
                        qy = (Rm[0, 2] - Rm[2, 0]) / S
                        qz = (Rm[1, 0] - Rm[0, 1]) / S
                    elif (m00 > m11) and (m00 > m22):
                        S = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
                        qw = (Rm[2, 1] - Rm[1, 2]) / S
                        qx = 0.25 * S
                        qy = (Rm[0, 1] + Rm[1, 0]) / S
                        qz = (Rm[0, 2] + Rm[2, 0]) / S
                    elif m11 > m22:
                        S = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
                        qw = (Rm[0, 2] - Rm[2, 0]) / S
                        qx = (Rm[0, 1] + Rm[1, 0]) / S
                        qy = 0.25 * S
                        qz = (Rm[1, 2] + Rm[2, 1]) / S
                    else:
                        S = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
                        qw = (Rm[1, 0] - Rm[0, 1]) / S
                        qx = (Rm[0, 2] + Rm[2, 0]) / S
                        qy = (Rm[1, 2] + Rm[2, 1]) / S
                        qz = 0.25 * S
                    q = torch.tensor([qw, qx, qy, qz], device=Rm.device, dtype=Rm.dtype)
                    q = q / (q.norm() + 1e-8)
                    return q
                quat_i = _matrix_to_quaternion_single(R_prime)
                s1 = sigma_m
                s2 = torch.ones_like(s1) * sigma_v2
                s3 = torch.ones_like(s1) * sigma_v3
                scaling_i = torch.stack([s1, s2, s3], dim=1)
                rotation_i = quat_i.unsqueeze(0)

            xyz_i = xyz_base.unsqueeze(0)  # t=0
            sign_m = torch.ones_like(sigma_m)
            fdc_i = feat_dc_base.unsqueeze(0) * sign_m.view(-1, 1, 1)
            frest_i = feat_rest_base.unsqueeze(0) * sign_m.view(-1, 1, 1)
            w_m = torch.abs(f_t).view(-1, 1)

            list_xyz.append(xyz_i)
            list_opacity.append(w_m)
            list_scaling.append(scaling_i)
            list_rotation.append(rotation_i)
            list_fdc.append(fdc_i)
            list_frest.append(frest_i)
            list_orig.append(torch.full((1,), i, dtype=torch.long, device=device))
            continue

        # 该高斯的所有 t_m 初值（以 cos 的极值 mπ/ω 为起点），仅取 |t| ≤ t_range
        om = omega[i].clamp_min(1e-8)  # tensor scalar，保持可微
        sig_u = sigma_u[i].clamp_min(1e-8)
        t_range = t_sigma_range * sig_u

        # m from -k_i..k_i
        m_vals = torch.arange(-k_i, k_i + 1, device=device, dtype=xyz.dtype)
        t0 = m_vals * (math.pi / om)
        om_b = torch.ones_like(t0) * om
        sig_b = torch.ones_like(t0) * sig_u
        t_refined = _newton_refine_t(om_b, sig_b, t0, steps=steps)

        # 过滤 |t| > t_range 的点
        valid_mask = torch.abs(t_refined) <= t_range
        if not valid_mask.any():
            # 回退为不分裂
            list_xyz.append(xyz[i].unsqueeze(0))
            list_opacity.append(opacity[i].unsqueeze(0))
            list_scaling.append(scaling[i].unsqueeze(0))
            list_rotation.append(rotation[i].unsqueeze(0))
            list_fdc.append(features_dc[i].unsqueeze(0))
            list_frest.append(features_rest[i].unsqueeze(0))
            list_orig.append(torch.full((1,), i, dtype=torch.long, device=device))
            continue

        t_sel = t_refined[valid_mask]

        # 计算 f(t), f''(t) 以及 σ_m
        # f(t) = cos(ωt) * exp(-t^2/(2σ^2))
        # f''(t) = [ -ω^2 cos(ω t) + (t^2/σ^4 - 1/σ^2) cos(ω t) + (2 ω t/σ^2) sin(ω t) ] * exp(-t^2/(2σ^2))
        wt = om_b[:t_sel.numel()] * t_sel
        envelope = torch.exp(-0.5 * (t_sel ** 2) / (sig_b[:t_sel.numel()] ** 2))
        cos_wt = torch.cos(wt)
        sin_wt = torch.sin(wt)
        f_t = cos_wt * envelope

        sigma2 = sig_b[:t_sel.numel()] ** 2
        term1 = - (om_b[:t_sel.numel()] ** 2) * cos_wt
        term2 = ((t_sel ** 2) / (sigma2 ** 2) - 1.0 / sigma2) * cos_wt
        term3 = (2.0 * om_b[:t_sel.numel()] * t_sel / sigma2) * sin_wt
        f2_t = (term1 + term2 + term3) * envelope

        # σ_m^2 = |f(t_m)| / (|f''(t_m)| + eps)
        eps = 1e-8
        sigma_m = torch.sqrt(torch.clamp(torch.abs(f_t) / (torch.abs(f2_t) + eps), min=1e-6))
        
        # 位置与缩放（各向异性，沿 wave 对齐主轴，保留正交方向方差）
        u_i = u[i]
        xyz_base = xyz[i]
        scaling_base = scaling[i]
        rotation_base = rotation[i]
        feat_dc_base = features_dc[i]
        feat_rest_base = features_rest[i]

        # 构造与 wave 对齐的旋转矩阵/快速对齐
        if fast_align:
            # 快速路径：沿最接近wave方向的原主轴调整尺度，复用原rotation；支持参数化替代
            R_i = build_rotation(rotation_base.unsqueeze(0))[0]
            axes = [R_i[:, 0], R_i[:, 1], R_i[:, 2]]
            dots = torch.stack([torch.abs(torch.dot(u_i, a)) for a in axes])
            main_axis = int(torch.argmax(dots).item())
            # 计算正交方向标准差
            L_i = build_scaling_rotation(scaling_base.unsqueeze(0), rotation_base.unsqueeze(0))[0]
            sigma_other = []
            for j in range(3):
                if j == main_axis:
                    continue
                vj = axes[j]
                sigma_other.append(torch.norm(L_i.transpose(0, 1) @ vj, p=2))
            # 模式选择：参数化替代（mixed/parametric）
            use_parametric = False
            try:
                from config_manager import config_manager
                mode = getattr(config_manager.config, 'split_mode', 'mixed')
                use_parametric = (mode in ('parametric', 'mixed')) and hasattr(pc, '_split_sigma_alpha')
            except Exception:
                pass
            # 沿主轴的尺度：参数化 or 精确 sigma_m
            if use_parametric:
                # 严格参数化：若索引异常则对齐尺寸后再取值，禁止回退精确
                if not hasattr(pc, '_split_sigma_alpha') or pc._split_sigma_alpha.shape[0] != pc.get_xyz.shape[0]:
                    # 自动对齐尺寸（填1），避免索引错误
                    with torch.no_grad():
                        device = L_i.device
                        fixed = torch.ones((pc.get_xyz.shape[0], 3), device=device)
                        if hasattr(pc, '_split_sigma_alpha') and pc._split_sigma_alpha.numel() > 0:
                            n_old = min(pc._split_sigma_alpha.shape[0], pc.get_xyz.shape[0])
                            fixed[:n_old] = pc._split_sigma_alpha[:n_old].to(device)
                        pc._split_sigma_alpha = fixed
                alpha_i = pc._split_sigma_alpha[i, main_axis].clamp_min(0.2).clamp_max(5.0)
                s_main = alpha_i * torch.norm(L_i.transpose(0, 1) @ axes[main_axis], p=2)
            else:
                s_main = sigma_m
            s = [None, None, None]
            s[main_axis] = s_main
            idx_other = 0
            for j in range(3):
                if j == main_axis:
                    continue
                s[j] = torch.ones_like(sigma_m) * sigma_other[idx_other]
                idx_other += 1
            scaling_i = torch.stack(s, dim=1)
            rotation_i = rotation_base.unsqueeze(0).expand(scaling_i.shape[0], -1)
        else:
            # 精确路径：构造与wave对齐的旋转
            R_i = build_rotation(rotation_base.unsqueeze(0))[0]
            cols = [R_i[:, 0], R_i[:, 1], R_i[:, 2]]
            dots = [torch.abs(torch.dot(u_i, c)) for c in cols]
            aux_idx = int(torch.tensor(dots).argmin().item())
            aux_axis = cols[aux_idx]
            v1 = u_i / (u_i.norm() + 1e-8)
            v2 = aux_axis - torch.dot(aux_axis, v1) * v1
            v2 = v2 / (v2.norm() + 1e-8)
            v3 = torch.linalg.cross(v1, v2)
            R_prime = torch.stack([v1, v2, v3], dim=1)
            # 计算正交方向的标准差
            L_i = build_scaling_rotation(scaling_base.unsqueeze(0), rotation_base.unsqueeze(0))[0]
            sigma_v2 = torch.norm(L_i.transpose(0, 1) @ v2, p=2)
            sigma_v3 = torch.norm(L_i.transpose(0, 1) @ v3, p=2)
            # 矩阵->四元数
            def _matrix_to_quaternion_single(Rm: torch.Tensor) -> torch.Tensor:
                m00 = Rm[0, 0]; m11 = Rm[1, 1]; m22 = Rm[2, 2]
                trace = m00 + m11 + m22
                if trace > 0:
                    S = torch.sqrt(trace + 1.0) * 2.0
                    qw = 0.25 * S
                    qx = (Rm[2, 1] - Rm[1, 2]) / S
                    qy = (Rm[0, 2] - Rm[2, 0]) / S
                    qz = (Rm[1, 0] - Rm[0, 1]) / S
                elif (m00 > m11) and (m00 > m22):
                    S = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
                    qw = (Rm[2, 1] - Rm[1, 2]) / S
                    qx = 0.25 * S
                    qy = (Rm[0, 1] + Rm[1, 0]) / S
                    qz = (Rm[0, 2] + Rm[2, 0]) / S
                elif m11 > m22:
                    S = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
                    qw = (Rm[0, 2] - Rm[2, 0]) / S
                    qx = (Rm[0, 1] + Rm[1, 0]) / S
                    qy = 0.25 * S
                    qz = (Rm[1, 2] + Rm[2, 1]) / S
                else:
                    S = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
                    qw = (Rm[1, 0] - Rm[0, 1]) / S
                    qx = (Rm[0, 2] + Rm[2, 0]) / S
                    qy = (Rm[1, 2] + Rm[2, 1]) / S
                    qz = 0.25 * S
                q = torch.tensor([qw, qx, qy, qz], device=Rm.device, dtype=Rm.dtype)
                q = q / (q.norm() + 1e-8)
                return q
            quat_i = _matrix_to_quaternion_single(R_prime)
            s1 = sigma_m
            s2 = torch.ones_like(s1) * sigma_v2
            s3 = torch.ones_like(s1) * sigma_v3
            scaling_i = torch.stack([s1, s2, s3], dim=1)
            rotation_i = quat_i.unsqueeze(0).expand(scaling_i.shape[0], -1)

        # 位置与属性设置（构造计算图）
        xyz_i = xyz_base.unsqueeze(0) + t_sel.unsqueeze(1) * u_i.unsqueeze(0)
        sign_m = torch.sign(cos_wt[:t_sel.numel()])
        sign_m = torch.where(sign_m == 0, torch.ones_like(sign_m), sign_m)
        fdc_i = feat_dc_base.unsqueeze(0) * sign_m.view(-1, 1, 1)
        frest_i = feat_rest_base.unsqueeze(0) * sign_m.view(-1, 1, 1)
        w_m = torch.abs(f_t[:t_sel.numel()]).view(-1, 1)

        list_xyz.append(xyz_i)
        list_opacity.append(w_m)
        list_scaling.append(scaling_i)
        list_rotation.append(rotation_i)
        list_fdc.append(fdc_i)
        list_frest.append(frest_i)
        list_orig.append(torch.full((t_sel.numel(),), i, dtype=torch.long, device=device))
    # 拼接
    split_xyz = torch.cat(list_xyz, dim=0)
    split_opacity = torch.cat(list_opacity, dim=0)
    split_scaling = torch.cat(list_scaling, dim=0)
    split_rotation = torch.cat(list_rotation, dim=0)
    split_features_dc = torch.cat(list_fdc, dim=0)
    split_features_rest = torch.cat(list_frest, dim=0)
    original_indices = torch.cat(list_orig, dim=0)

    # 能量归一化：针对每个原始 i，使其所有分裂的 alpha 之和等于原始 alpha
    weight_sum = torch.zeros((N,), device=device)
    weight_sum.scatter_add_(0, original_indices, split_opacity.squeeze(1))
    weight_sum = weight_sum.clamp(min=1e-8)

    # 扩展到每个分裂条目
    norm_factor = opacity.squeeze(1)[original_indices] / weight_sum[original_indices]
    split_opacity = split_opacity * norm_factor.unsqueeze(1)

    return {
        'split_xyz': split_xyz,
        'split_opacity': split_opacity,
        'split_scaling': split_scaling,
        'split_rotation': split_rotation,
        'split_features_dc': split_features_dc,
        'split_features_rest': split_features_rest,
        'n_splits': total_splits,
        'n_original': N,
        'original_indices': original_indices,
    }


# 仅用于导出：分块版分裂，避免一次性占用显存/内存
def compute_splits_precise_chunked(pc,
                                   iteration: int = 0,
                                   max_iteration: int = 40000,
                                   max_k: Optional[int] = None,
                                   t_sigma_range: Optional[float] = None,
                                   min_wave_norm: Optional[float] = None,
                                   batch_size: int = 150000,
                                   relax_limits: bool = True) -> Optional[Dict]:
    """
    仅在导出预分裂时使用：将点分成多个批次，分别调用 compute_splits_precise，再拼接结果。
    - relax_limits=True 时，临时放宽 OOM 防线，避免返回 None。
    """
    import types
    N = pc.get_xyz.shape[0]
    if N == 0:
        return None

    # 读取并可选放宽限制
    try:
        from config_manager import config_manager
        cfg = config_manager.config
        original_skip_center = getattr(cfg, 'split_skip_center_minN', 500000)
        original_extra_ratio = getattr(cfg, 'split_max_extra_ratio', 0.15)
        if relax_limits:
            cfg.split_skip_center_minN = 1 << 30
            cfg.split_max_extra_ratio = 10.0
    except Exception:
        cfg = None
        original_skip_center = None
        original_extra_ratio = None

    # 构造一个浅包装，暴露与 pc 相同的接口，但数据为切片
    class _ShallowPC:
        def __init__(self, base, sl):
            self._xyz = base.get_xyz[sl]
            self._opacity = base.get_opacity[sl]
            self._scaling = base.get_scaling[sl]
            self._rotation = base.get_rotation[sl]
            self._features_dc = base._features_dc[sl]
            self._features_rest = base._features_rest[sl]
            self._wave = base._wave[sl] if hasattr(base, '_wave') else None
            self._max_splits = getattr(base, '_max_splits', 5)
        @property
        def get_xyz(self):
            return self._xyz
        @property
        def get_opacity(self):
            return self._opacity
        @property
        def get_scaling(self):
            return self._scaling
        @property
        def get_rotation(self):
            return self._rotation

    # 结果累积
    pieces = {
        'split_xyz': [],
        'split_opacity': [],
        'split_scaling': [],
        'split_rotation': [],
        'split_features_dc': [],
        'split_features_rest': [],
        'original_indices': [],
    }

    # 分块遍历
    start = 0
    while start < N:
        end = min(start + batch_size, N)
        sl = slice(start, end)
        view_pc = _ShallowPC(pc, sl)
        part = compute_splits_precise(
            view_pc,
            iteration=iteration,
            max_iteration=max_iteration,
            max_k=max_k,
            t_sigma_range=t_sigma_range,
            min_wave_norm=min_wave_norm if (min_wave_norm is not None) else 0.0,
        )
        if part is not None:
            pieces['split_xyz'].append(part['split_xyz'])
            pieces['split_opacity'].append(part['split_opacity'])
            pieces['split_scaling'].append(part['split_scaling'])
            pieces['split_rotation'].append(part['split_rotation'])
            pieces['split_features_dc'].append(part['split_features_dc'])
            pieces['split_features_rest'].append(part['split_features_rest'])
            # 重要：偏移原索引
            pieces['original_indices'].append(part['original_indices'] + start)
        start = end

    # 恢复限制参数
    if cfg is not None and relax_limits:
        cfg.split_skip_center_minN = original_skip_center
        cfg.split_max_extra_ratio = original_extra_ratio

    if len(pieces['split_xyz']) == 0:
        return None

    # 拼接
    split_xyz = torch.cat(pieces['split_xyz'], dim=0)
    split_opacity = torch.cat(pieces['split_opacity'], dim=0)
    split_scaling = torch.cat(pieces['split_scaling'], dim=0)
    split_rotation = torch.cat(pieces['split_rotation'], dim=0)
    split_features_dc = torch.cat(pieces['split_features_dc'], dim=0)
    split_features_rest = torch.cat(pieces['split_features_rest'], dim=0)
    original_indices = torch.cat(pieces['original_indices'], dim=0)

    return {
        'split_xyz': split_xyz,
        'split_opacity': split_opacity,
        'split_scaling': split_scaling,
        'split_rotation': split_rotation,
        'split_features_dc': split_features_dc,
        'split_features_rest': split_features_rest,
        'n_splits': split_xyz.shape[0],
        'n_original': N,
        'original_indices': original_indices,
    }
