# -*- coding: utf-8 -*-
"""
DreamerV3-style World Model (Torch)
-----------------------------------
- 与 DreamerV3 一致的 world model 细节：
  * RMSNorm + GELU
  * BlockLinear + 门控 RSSM 核心
  * 离散组潜变量 (stoch_groups x stoch_classes) + unimix
  * 双 KL:  dyn = KL(stop(post)||prior), rep = KL(post||stop(prior))
  * 向量观测 symlog / 奖励 two-hot / 折扣 Bernoulli
  * 图像分支: uint8 编码 (4x stride2 conv) / 解码 (transpose conv) + 像素 MSE
- 外部接口保持不变：
  * class DreamerV3WorldModel
  * predict(self, s_t, a_t) -> (r_pred, s1_pred, d_pred)
  * train(self, s,a,r,s1,d,length)  # 也兼容 train(mode: bool)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List  # <<< 补上 List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- Utils -------------------------
def symlog(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0 + eps)

def symexp(y: torch.Tensor) -> torch.Tensor:
    s = torch.sign(y)
    return s * (torch.exp(torch.abs(y)) - 1.0)

def two_hot(x: torch.Tensor, n_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    # x: [...], return [..., n_bins]
    x = x.clamp(vmin, vmax)
    t = (x - vmin) / max(vmax - vmin, 1e-8) * (n_bins - 1)
    l = torch.floor(t); u = torch.ceil(t)
    w_u = (t - l); w_l = 1.0 - w_u
    l = l.long().clamp(0, n_bins - 1); u = u.long().clamp(0, n_bins - 1)
    out = torch.zeros((*x.shape, n_bins), device=x.device, dtype=torch.float32)
    out.scatter_(-1, l.unsqueeze(-1), w_l.unsqueeze(-1))
    out.scatter_(-1, u.unsqueeze(-1), w_u.unsqueeze(-1))
    return out

def two_hot_expectation(p: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    n_bins = p.size(-1)
    bins = torch.linspace(vmin, vmax, n_bins, device=p.device, dtype=p.dtype)
    return (p * bins).sum(dim=-1)

def unimix_logits(logits: torch.Tensor, mix: float = 0.01) -> torch.Tensor:
    # logits: [..., C]
    K = logits.size(-1)
    probs = (1.0 - mix) * torch.softmax(logits, dim=-1) + mix * (1.0 / K)
    return torch.log(probs.clamp_min(1e-8))

# ------------------------- Norm / Blocks -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        x2 = x.pow(2).mean(dim=-1, keepdim=True)
        xhat = x * torch.rsqrt(x2 + self.eps)
        return self.g * xhat

class BlockLinear(nn.Module):
    """分组线性层（与 DreamerV3 的 block 线性一致的参数/统计特性）"""
    def __init__(self, in_dim: int, out_dim: int, n_blocks: int = 8, bias: bool = True):
        super().__init__()
        assert in_dim % n_blocks == 0 and out_dim % n_blocks == 0, "dims must be divisible by n_blocks"
        self.n_blocks = n_blocks
        self.in_per  = in_dim // n_blocks
        self.out_per = out_dim // n_blocks
        self.weight = nn.Parameter(torch.zeros(n_blocks, self.in_per, self.out_per))
        self.bias   = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        B = x.shape[:-1]
        x = x.view(*B, self.n_blocks, self.in_per)
        y = torch.einsum('...bi,bio->...bo', x, self.weight)  # each block independent
        y = y.reshape(*B, self.n_blocks * self.out_per)
        if self.bias is not None:
            y = y + self.bias
        return y

class GLU(nn.Module):
    def __init__(self, dim: int, blocks: int):
        super().__init__()
        self.lin = BlockLinear(dim, dim * 2, n_blocks=blocks)
        self.norm = RMSNorm(dim * 2)
    def forward(self, x):
        a, b = self.norm(self.lin(x)).chunk(2, dim=-1)
        return a * torch.sigmoid(b)

class DreamerMLP(nn.Module):
    """
    MLP with RMSNorm + (BlockLinear or fallback Linear).
    - 对于中间层：若 in/out 维均可被 blocks 整除且 blocks>1，则使用 BlockLinear；否则回退 nn.Linear
    - 对于最后一层（投影到 out_dim）：同样判定，常见情况如 255/1 无法整除 -> 用 nn.Linear
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_layers: int = 2, act=nn.SiLU, blocks: int = 8):
        super().__init__()
        assert n_layers >= 1
        self.blocks = int(blocks)
        self.act = act
        layers: List[nn.Module] = []
        d = int(in_dim)

        # 中间层
        for _ in range(max(0, n_layers - 1)):
            layers.append(RMSNorm(d))
            if self._use_block(d, hidden):
                layers.append(BlockLinear(d, hidden, self.blocks))
            else:
                layers.append(nn.Linear(d, hidden))
            layers.append(self.act())
            d = int(hidden)

        # 最后一层
        layers.append(RMSNorm(d))
        if self._use_block(d, out_dim):
            layers.append(BlockLinear(d, int(out_dim), self.blocks))
        else:
            layers.append(nn.Linear(d, int(out_dim)))

        self.net = nn.Sequential(*layers)

    def _use_block(self, in_dim: int, out_dim: int) -> bool:
        return (self.blocks is not None) and (self.blocks > 1) and \
               (in_dim % self.blocks == 0) and (out_dim % self.blocks == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------- CNN Enc/Dec for images -------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, img_hw: Tuple[int, int]):
        super().__init__()
        H, W = img_hw
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GELU(),
        )
        h4, w4 = H // 16, W // 16
        self.proj = nn.Linear(256 * h4 * w4, out_dim)
        self.norm = RMSNorm(out_dim)
    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        # x: [BN, C, H, W], uint8
        x = x_uint8.float() / 255.0 - 0.5
        x = self.conv(x).flatten(1)
        x = self.proj(x)
        return self.norm(x)

class CNNDecoder(nn.Module):
    def __init__(self, in_dim: int, out_ch: int, img_hw: Tuple[int, int]):
        super().__init__()
        H, W = img_hw
        self.h4, self.w4 = H // 16, W // 16
        self.fc = nn.Linear(in_dim, 256 * self.h4 * self.w4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32,  out_ch, 4, 2, 1),
        )
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat)
        x = x.view(x.size(0), 256, self.h4, self.w4)
        x = self.deconv(x)
        return torch.sigmoid(x)

# ------------------------- Config -------------------------
@dataclass
class DreamerV3WMArgs:
    # observation / action
    obs_dim: int
    action_dim: int
    # core sizes
    deter: int = 4096
    hidden: int = 2048
    stoch_groups: int = 32
    stoch_classes: int = 32
    blocks: int = 8
    # losses
    kl_dyn_scale: float = 1.0
    kl_rep_scale: float = 1.0
    kl_free_nats: float = 1.0
    unimix_rate: float = 0.01
    obs_loss_scale: float = 1.0
    rew_loss_scale: float = 3.5
    done_loss_scale: float = 1.0
    # heads / bins
    rew_bins: int = 255
    rew_min: float = -1000.0
    rew_max: float = 0.0
    # optimization
    lr: float = 3e-4
    grad_clip: float = 100.0
    # action embedding
    action_embed: int = 64
    # graph pooling (保留你原先的可选 GCN 池化)
    use_graph_pool: bool = False

    # ---- image branch ----
    use_images: bool = False
    img_channels: int = 3
    img_height: int = 84
    img_width: int = 84
    enc_img_dim: int = 512
    obs_img_loss_scale: float = 1.0

# ------------------------- World Model -------------------------
class DreamerV3WorldModel(nn.Module):
    def __init__(self, args: DreamerV3WMArgs, adj: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.args   = args
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # graph pooling buffer
        self.register_buffer('A_hat', None)
        if adj is not None and args.use_graph_pool:
            A = torch.as_tensor(adj, dtype=torch.float32)
            I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
            A = A + I
            deg = A.sum(dim=-1, keepdim=True).clamp_min(1.0)
            self.A_hat = (A / deg)  # 规范化邻接

        # ----- Encoders -----
        enc_vec_in = args.obs_dim * (2 if self.A_hat is not None else 1)
        self.enc_vec = DreamerMLP(enc_vec_in, args.hidden, args.hidden, n_layers=2, blocks=args.blocks)

        # 期望的 posterior 编码维度（向量编码 + 可选图像编码）
        self.expected_enc_dim = int(args.hidden + (args.enc_img_dim if args.use_images else 0))

        if args.use_images:
            hw = (args.img_height, args.img_width)
            self.enc_img = CNNEncoder(args.img_channels, args.enc_img_dim, hw)

        # ----- Action encoder -----
        self.n_actions    = args.action_dim
        self.action_embed = nn.Embedding(self.n_actions, args.action_embed) if args.action_embed and args.action_embed > 0 else None
        act_in = args.action_embed if self.action_embed is not None else self.n_actions

        # ----- RSSM core -----
        self.G = args.stoch_groups; self.C = args.stoch_classes
        self.zdim = self.G * self.C
        core_in = self.zdim + act_in
        self.pre_norm = RMSNorm(core_in)
        self.pre_gate = GLU(core_in, blocks=args.blocks)
        self.gru      = nn.GRUCell(core_in, args.deter)
        self.post_mlp = DreamerMLP(args.deter, args.hidden, args.deter, n_layers=2, blocks=args.blocks)
        self.h_init   = nn.Parameter(torch.zeros(args.deter))

        # ----- Prior / Posterior heads -----
        post_in = args.deter + self.expected_enc_dim   # <<< 用期望维度，保持固定
        self.prior_head = DreamerMLP(args.deter, args.hidden, self.zdim, n_layers=2, blocks=args.blocks)
        self.post_head  = DreamerMLP(post_in,   args.hidden, self.zdim, n_layers=2, blocks=args.blocks)

        # ----- Decoders (vector + reward + discount + image?) -----
        feat_in = args.deter + self.zdim
        self.dec_obs = DreamerMLP(feat_in, args.hidden, args.obs_dim, n_layers=2, blocks=args.blocks)
        self.dec_rew = DreamerMLP(feat_in, args.hidden, args.rew_bins, n_layers=2, blocks=args.blocks)
        self.dec_dis = DreamerMLP(feat_in, args.hidden, 1, n_layers=2, blocks=args.blocks)
        if args.use_images:
            self.dec_img = CNNDecoder(feat_in, args.img_channels, (args.img_height, args.img_width))

        # move & optim
        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr)

    # ------------ helpers ------------
    def _encode_vec(self, o: torch.Tensor) -> torch.Tensor:
        # o: [*, N, D] or [*, D]
        if self.A_hat is not None and o.dim() >= 2:
            N, D = o.size(-2), o.size(-1)
            o2 = o.reshape(-1, N, D)
            pooled = torch.einsum('ij,bjd->bid', self.A_hat, o2)
            x = torch.cat([o2, pooled], dim=-1).reshape(*o.shape[:-2], N, -1)
        else:
            x = o
        x = symlog(x)
        return self.enc_vec(x)

    def _act_encode(self, a: torch.Tensor) -> torch.Tensor:
        # 支持 [B,N] / [B,N,1] / [B,N,A] (离散或 onehot)
        if a.dim() >= 3 and a.size(-1) == 1:
            a = a.squeeze(-1)
        if a.dtype in (torch.long, torch.int64):
            if self.action_embed is not None:
                return self.action_embed(a)
            return F.one_hot(a.clamp(min=0, max=self.n_actions-1), num_classes=self.n_actions).float()
        else:
            if self.action_embed is not None:
                idx = a.argmax(dim=-1)
                return self.action_embed(idx)
            return a

    def _flat_BN(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.dim() < 2: raise ValueError("expected at least [B,N,...]")
        B, N = x.size(0), x.size(1)
        return x.reshape(B * N, *x.shape[2:]), B, N

    def _split_groups(self, x: torch.Tensor) -> torch.Tensor:
        # [..., G*C] -> [..., G, C]
        return x.view(*x.shape[:-1], self.G, self.C)

    def _cat_groups(self, x: torch.Tensor) -> torch.Tensor:
        # [..., G, C] -> [..., G*C]
        return x.reshape(*x.shape[:-2], self.G * self.C)

    def _sample_st(self, logp: torch.Tensor) -> torch.Tensor:
        # straight-through Gumbel-Softmax one-hot
        u = torch.rand_like(logp).clamp_(1e-8, 1-1e-8)
        g = -torch.log(-torch.log(u))
        y = F.softmax(logp + g, dim=-1)
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y

    def _core(self, h: torch.Tensor, z_cat: torch.Tensor, a_enc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_cat, a_enc], dim=-1)
        x = self.pre_gate(self.pre_norm(x))   # GLU 门控
        h = self.gru(x, h)                    # GRU 更新
        h = self.post_mlp(h)                  # 后 MLP
        return h

    def _prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._split_groups(self.prior_head(h))
        logp   = unimix_logits(logits, self.args.unimix_rate)
        p      = torch.softmax(logp, dim=-1)
        return logp, p

    # --- 关键修复：posterior 输入维度自适应（补零到期望维度） ---
    def _pad_enc(self, enc: torch.Tensor) -> torch.Tensor:
        """将 enc 补到 expected_enc_dim（图像分支缺失时用 0 填充）。"""
        D = enc.size(-1)
        if D == self.expected_enc_dim:
            return enc
        if D < self.expected_enc_dim:
            pad = torch.zeros(*enc.shape[:-1], self.expected_enc_dim - D, device=enc.device, dtype=enc.dtype)
            return torch.cat([enc, pad], dim=-1)
        # D > expected：异常场景，先截断到期望长度（更稳妥可改为线性投影）
        return enc[..., :self.expected_enc_dim]

    def _posterior(self, h: torch.Tensor, enc_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_t = self._pad_enc(enc_t)                         # <<< 补零对齐
        x = torch.cat([h, enc_t], dim=-1)                    # [BN, deter + expected_enc_dim]
        logits = self._split_groups(self.post_head(x))
        logp   = unimix_logits(logits, self.args.unimix_rate)
        p      = torch.softmax(logp, dim=-1)
        return logp, p

    def _decode_all(self, h: torch.Tensor, z_cat: torch.Tensor, want_image: bool = False):
        feat = torch.cat([h, z_cat], dim=-1)
        obs_symlog = self.dec_obs(feat)
        rew_logits = self.dec_rew(feat)
        dis_logits = self.dec_dis(feat)     # continuation logit
        img_hat = None
        if want_image and self.args.use_images:
            img_hat = self.dec_img(feat)
        return obs_symlog, rew_logits, dis_logits, img_hat

    # ------------ public API ------------
    def eval(self):
        nn.Module.train(self, False)
        return self
    def train_mode(self, flag: bool = True):
        nn.Module.train(self, flag)
        return self

    @torch.no_grad()
    def predict(self, s_t: torch.Tensor, a_t: torch.Tensor):
        """单步：posterior@t (用观测) -> prior@t+1 -> decode"""
        self.eval()
        args = self.args
        s_t = s_t.to(self.device).float()   # [B,N,obs_dim]
        a_t = a_t.to(self.device)

        B, N = s_t.shape[0], s_t.shape[1]
        BN   = B * N

        # encoders
        enc_vec = self._encode_vec(s_t)                     # [B,N,Hvec]
        e = enc_vec.reshape(BN, -1)
        if args.use_images:
            # 预测时一般没有图像；此处补零到期望维度，确保与训练一致
            e = self._pad_enc(e)

        # action
        a_enc = self._act_encode(a_t)                       # [B,N,Ae]
        a = a_enc.reshape(BN, -1)

        # init states
        h = self.h_init.expand(BN, -1)
        z_cat = torch.zeros(BN, self.G * self.C, device=self.device)

        # posterior@t（向量 + 零填图像）
        h1 = self._core(h, z_cat, a)
        post_logp, post_p = self._posterior(h1, e)
        z = self._cat_groups(self._sample_st(post_logp))

        # prior@t+1
        h2 = self._core(h1, z, a)
        prior_logp, prior_p = self._prior(h2)
        z2 = self._cat_groups(prior_p)  # 期望更平滑

        # decode
        obs_symlog, rew_logits, dis_logits, _ = self._decode_all(h2, z2, want_image=False)

        # map to outputs
        p_rew = torch.softmax(rew_logits, dim=-1)
        rew_symlog = two_hot_expectation(p_rew, args.rew_min, args.rew_max)
        r  = symexp(rew_symlog)[..., None]                   # [BN,1]
        s1 = symexp(obs_symlog)                              # [BN,obs_dim]
        cont_prob = torch.sigmoid(dis_logits)                # continuation prob
        d  = 1.0 - cont_prob                                 # [BN,1]

        return r.view(B, N, 1), s1.view(B, N, args.obs_dim), d.view(B, N, 1)

    # --- dual-mode train(): supports nn.Module.train(mode) and WM update ---
    def train(self, *args, **kwargs):
        # case 1: behave like nn.Module.train(mode: bool)
        if (len(args) == 1 and isinstance(args[0], (bool,))) or ('mode' in kwargs and isinstance(kwargs['mode'], bool)):
            flag = args[0] if len(args) == 1 else kwargs.get('mode', True)
            return nn.Module.train(self, flag)

        # case 2: world model training step: (s, a, r, s1, d, length, image=None, image1=None)
        s, a, r, s1, d, length = args[:6]
        image  = kwargs.get('image', None)   # [B,T,N,C,H,W] or None
        image1 = kwargs.get('image1', None)  # 可忽略
        cfg, device = self.args, self.device
        nn.Module.train(self, True)

        # to device / slice
        s = s[:, :length].to(device).float()        # [B,T,N,Obs]
        a = a[:, :length].to(device)                # [B,T,N,(A)]
        r = r[:, :length].to(device).float()        # [B,T,N,1]
        d = d[:, :length].to(device).float()        # [B,T,N,1]
        if image is not None:
            image = image[:, :length].to(device)    # [B,T,N,C,H,W]

        # targets
        s_tgt = symlog(s)                            # vec obs -> symlog
        r_tgt = symlog(r.squeeze(-1))               # [B,T,N]
        cont_tgt = 1.0 - d.squeeze(-1)              # [B,T,N]

        # encoders（逐步 or 预编码）
        enc_vec = self._encode_vec(s)               # [B,T,N,Hvec]
        if cfg.use_images and image is not None:
            # 展平成 [B*N*T, C, H, W] 编码再还原
            B, T, N, C, H, W = image.shape
            img_bn = image.permute(0,2,1,3,4,5).reshape(B*N*T, C, H, W)
            enc_img_bn = self.enc_img(img_bn)                       # [B*N*T, Himg]
            enc_img = enc_img_bn.view(B, N, T, -1).permute(0,2,1,3) # [B,T,N,Himg]
        else:
            enc_img = None

        # action enc
        a_enc = self._act_encode(a)                  # [B,T,N,Ae]

        # to [B*N, T, ...]
        B, T, N = s.shape[0], s.shape[1], s.shape[2]
        e_vec   = enc_vec.permute(0,2,1,3).reshape(B*N, T, -1)
        a_e     = a_enc.permute(0,2,1,3).reshape(B*N, T, -1)
        s_tgt   = s_tgt.permute(0,2,1,3).reshape(B*N, T, -1)
        r_tgt   = r_tgt.permute(0,2,1).reshape(B*N, T)
        cont_tgt= cont_tgt.permute(0,2,1).reshape(B*N, T)
        if enc_img is not None:
            e_img = enc_img.permute(0,2,1,3).reshape(B*N, T, -1)
            e_tok = lambda t: self._pad_enc(torch.cat([e_vec[:, t], e_img[:, t]], dim=-1))
        else:
            e_img = None
            e_tok = lambda t: self._pad_enc(e_vec[:, t])            # <<< 无图像时补零

        BN = B * N
        h = self.h_init.expand(BN, -1)
        z_cat = torch.zeros(BN, self.G * self.C, device=device)

        # Accumulate losses
        L_obs = L_rew = L_dis = L_dyn = L_rep = 0.0
        L_img = 0.0

        for t in range(T):
            # RSSM: core -> prior/post
            h = self._core(h, z_cat, a_e[:, t])
            prior_logp, prior_p = self._prior(h)
            post_logp,  post_p  = self._posterior(h, e_tok(t))
            z_onehot = self._sample_st(post_logp)            # [BN,G,C]
            z_cat = self._cat_groups(z_onehot)               # [BN,G*C]

            # decode
            obs_symlog, rew_logits, dis_logits, img_hat = self._decode_all(h, z_cat, want_image=(cfg.use_images and image is not None))

            # --- losses ---
            # 1) vector reconstruction (symlog)
            L_obs_t = F.mse_loss(obs_symlog, s_tgt[:, t], reduction='none').mean(dim=-1).mean()

            # 2) reward two-hot CE
            rew_th  = two_hot(r_tgt[:, t], cfg.rew_bins, cfg.rew_min, cfg.rew_max)
            L_rew_t = -(rew_th * F.log_softmax(rew_logits, dim=-1)).sum(dim=-1).mean()

            # 3) discount BCE (continuation)
            L_dis_t = F.binary_cross_entropy_with_logits(dis_logits.squeeze(-1), cont_tgt[:, t])

            # 4) image reconstruction (if any)
            if cfg.use_images and image is not None:
                img_t = image.permute(0,2,1,3,4,5).reshape(BN, T, *image.shape[3:])[:, t]  # [BN,C,H,W]
                img_target = (img_t.float()/255.0).clamp(0,1)
                L_img_t = F.mse_loss(img_hat, img_target, reduction='none').mean(dim=(1,2,3)).mean()
            else:
                L_img_t = 0.0

            # 5) KL: dyn / rep with free-nats
            def kl_cat(p_logp, p_p, q_logp, q_p):
                return (p_p * (p_logp - q_logp)).sum(dim=-1).sum(dim=-1)

            kl_dyn_t = kl_cat(post_logp.detach(), post_p.detach(), prior_logp, prior_p)
            kl_rep_t = kl_cat(post_logp, post_p, prior_logp.detach(), prior_p.detach())

            fn = cfg.kl_free_nats
            kl_dyn_t = torch.clamp(kl_dyn_t - fn, min=0.0).mean()
            kl_rep_t = torch.clamp(kl_rep_t - fn, min=0.0).mean()

            # accumulate
            L_obs += L_obs_t; L_rew += L_rew_t; L_dis += L_dis_t
            L_dyn += kl_dyn_t; L_rep += kl_rep_t
            if cfg.use_images and image is not None:
                L_img += L_img_t

        T_inv = 1.0 / T
        L_obs *= T_inv; L_rew *= T_inv; L_dis *= T_inv; L_dyn *= T_inv; L_rep *= T_inv
        if cfg.use_images and image is not None:
            L_img *= T_inv
        else:
            L_img = torch.as_tensor(0.0, device=device)

        loss = (cfg.obs_loss_scale * L_obs +
                cfg.rew_loss_scale * L_rew +
                cfg.done_loss_scale * L_dis +
                cfg.kl_dyn_scale * L_dyn +
                cfg.kl_rep_scale * L_rep +
                cfg.obs_img_loss_scale * L_img)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
        self.opt.step()

        return {
            'wm/loss': float(loss.detach().cpu()),
            'wm/obs_loss': float(L_obs.detach().cpu()),
            'wm/img_loss': float(L_img.detach().cpu()),
            'wm/rew_loss': float(L_rew.detach().cpu()),
            'wm/dis_loss': float(L_dis.detach().cpu()),
            'wm/kl_dyn': float(L_dyn.detach().cpu()),
            'wm/kl_rep': float(L_rep.detach().cpu()),
        }

    @staticmethod
    def from_p_args(p_args: dict, adj: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        args = DreamerV3WMArgs(
            # core I/O
            obs_dim=int(p_args.get('obs_dim')),
            action_dim=int(p_args.get('act_n')),
            # sizes
            deter=int(p_args.get('deter', 4096)),
            hidden=int(p_args.get('hidden', 2048)),
            stoch_groups=int(p_args.get('stoch_groups', p_args.get('stoch_g', 32))),
            stoch_classes=int(p_args.get('stoch_classes', p_args.get('stoch_c', 32))),
            blocks=int(p_args.get('blocks', 8)),
            # losses / KL
            kl_dyn_scale=float(p_args.get('kl_dyn_scale', 1.0)),
            kl_rep_scale=float(p_args.get('kl_rep_scale', 1.0)),
            kl_free_nats=float(p_args.get('kl_free_nats', 1.0)),
            unimix_rate=float(p_args.get('unimix', 0.01)),
            obs_loss_scale=float(p_args.get('obs_loss_scale', 1.0)),
            rew_loss_scale=float(p_args.get('rew_loss_scale', 3.5)),
            done_loss_scale=float(p_args.get('done_loss_scale', 1.0)),
            # reward bins
            rew_bins=int(p_args.get('rew_bins', 255)),
            rew_min=float(p_args.get('rew_min', -1000.0)),
            rew_max=float(p_args.get('rew_max', 0.0)),
            # optim
            lr=float(p_args.get('lr', 3e-4)),
            grad_clip=float(p_args.get('grad_clip', 100.0)),
            # action emb
            action_embed=int(p_args.get('action_embed', 64)),
            # graph pool
            use_graph_pool=bool(p_args.get('use_graph_pool', False)),

            # ---- image branch ----
            use_images=bool(p_args.get('use_images', False)),
            img_channels=int(p_args.get('img_channels', 3)),
            img_height=int(p_args.get('img_height', 84)),
            img_width=int(p_args.get('img_width', 84)),
            enc_img_dim=int(p_args.get('enc_img_dim', 512)),
            obs_img_loss_scale=float(p_args.get('obs_img_loss_scale', 1.0)),
        )
        return DreamerV3WorldModel(args, adj=adj, device=device)
