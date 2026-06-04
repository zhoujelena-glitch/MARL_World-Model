# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:09:45 2022

@author: 86153
"""

import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from algorithms.models import CategoricalActor, EnsembledModel, SquashedGaussianActor, ParameterizedModel_MBPPO
import random
import multiprocessing as mp
from torch import distributed as dist
import argparse
from algorithms.algo.agent.DPPO import DPPOAgent
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer
from algorithms.models import GraphConvolutionalModel
from algorithms.models_dreamer import DreamerV3WorldModel


class ModelBasedAgent(nn.ModuleList):
    def __init__(self, logger, device, agent_args, env_args, **kwargs):
        # 保持与你现有工程一致的 super 调用方式
        super().__init__(logger, device, agent_args, env_args, **kwargs)
        self.logger = logger
        self.device = device
        self.lr_p = agent_args.lr_p
        self.p_args = agent_args.p_args

        # —— 选择世界模型
        model_type = str(getattr(self.p_args, "model_type", "")).lower()
        if model_type == "dreamerv3":
            # 将必要信息透传给 DreamerV3
            p_dict = dict(getattr(self.p_args, "__dict__", {}))
            p_dict.update({"obs_dim": self.observation_dim, "act_n": self.action_dim})
            self.ps = DreamerV3WorldModel.from_p_args(p_dict, adj=getattr(self, "adj", None), device=self.device)
            self.wm_done_thr = float(getattr(self.p_args, "wm_done_thr", 0.5))
            print("[DMPO] World Model = DreamerV3WorldModel (RSSM, image-ready)")
        else:
            self.ps = GraphConvolutionalModel(
                self.logger,
                getattr(self, "adj", None),
                self.observation_dim,
                self.action_dim,
                self.n_agent,
                self.p_args,
            ).to(self.device)
            print("[DMPO] World Model = GraphConvolutionalModel")

        # 旧世界模型仍使用外部优化器；DreamerV3 内部自带优化器（见 updateModel 分支）
        self.optimizer_p = Adam(self.ps.parameters(), lr=self.lr_p)

    def updateModel(self, trajs, length=1):
        """
        训练世界模型
        输入（每个 traj 的字典）：
            s:  [T, N, state_dim]
            a:  [T, N] (discrete) or [T, N, action_dim] (continuous/onehot)
            r:  [T, N, 1]
            s1: [T, N, state_dim]
            d:  [T, N, 1]
            img (可选):  [T, N, C, H, W] (uint8)
            img1(可选):  [T, N, C, H, W] (uint8)
        """
        time_t = time.time()
        loss_total = 0.0

        ss, actions, rs, s1s, ds = [], [], [], [], []
        imgs, img1s = [], []

        for traj in trajs:
            s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
            img = traj.get("img", None)
            img1 = traj.get("img1", None)

            s, a, r, s1, d = [torch.as_tensor(item, device=self.device) for item in [s, a, r, s1, d]]
            if img is not None:
                img = torch.as_tensor(img, device=self.device, dtype=torch.uint8)
            if img1 is not None:
                img1 = torch.as_tensor(img1, device=self.device, dtype=torch.uint8)

            ss.append(s)
            actions.append(a)
            rs.append(r)
            s1s.append(s1)
            ds.append(d)
            if img is not None:
                imgs.append(img)
            if img1 is not None:
                img1s.append(img1)

        ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
        imgs = torch.stack(imgs, dim=0) if imgs else None            # [B,T,N,C,H,W] or None
        img1s = torch.stack(img1s, dim=0) if img1s else None

        # DreamerV3: 内部优化并返回 metrics；旧模型：返回 (loss, state_err)
        ret = self.ps.train(ss, actions, rs, s1s, ds, length, image=imgs, image1=img1s)

        if isinstance(ret, tuple):  # 兼容旧世界模型
            loss, rel_state_error = ret
            self.optimizer_p.zero_grad()
            loss.sum().backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.ps.parameters(), 5.0)
            self.optimizer_p.step()
            rel_err_val = float(rel_state_error.detach().cpu())
            self.logger.log(p_loss_total=float(loss.sum().detach().cpu()), p_update=None)
        else:  # DreamerV3 风格返回 dict，已在内部优化
            metrics = ret or {}
            rel_err_val = float(metrics.get("wm/obs_loss", 0.0))
            # 关键指标直接打点
            self.logger.log(
                p_loss_total=metrics.get("wm/loss", 0.0),
                wm_obs_loss=metrics.get("wm/obs_loss", 0.0),
                wm_img_loss=metrics.get("wm/img_loss", 0.0),
                wm_kl_dyn=metrics.get("wm/kl_dyn", 0.0),
                wm_kl_rep=metrics.get("wm/kl_rep", 0.0),
            )

        self.logger.log(model_update_time=time.time() - time_t)
        return rel_err_val

    def validateModel(self, trajs, length=1):
        """简单验证：单步 s,a -> 预测 s1，与真实 s1 做 RMSE。"""
        with torch.no_grad():
            ss, actions, rs, s1s, ds = [], [], [], [], []
            for traj in trajs:
                s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                s, a, r, s1, d = [torch.as_tensor(x, device=self.device) for x in [s, a, r, s1, d]]
                ss.append(s)
                actions.append(a)
                s1s.append(s1)

            ss = torch.stack(ss, dim=0)  # [B,T,N,D]
            actions = torch.stack(actions, dim=0)  # [B,T,N] or [B,T,N,A]
            s1s = torch.stack(s1s, dim=0)  # [B,T,N,D]

            T = int(length)
            s = ss[:, :T].reshape(-1, ss.size(2), ss.size(3))  # [B*T, N, D]
            a = actions[:, :T].reshape(-1, actions.size(2), *actions.shape[3:])  # [B*T, N, ...]
            s1_true = s1s[:, :T].reshape(-1, s1s.size(2), s1s.size(3))

            r_pred, s1_pred, d_pred = self.ps.predict(s, a)
            rel_err = (s1_pred - s1_true).pow(2).mean().sqrt()  # RMSE
            return float(rel_err.detach().cpu())

    def model_step(self, s, a):
        """
        使用世界模型做一步想象：
            s: [B, N, state_dim]
            a: [B, N] (discrete) or [B, N, action_dim] (continuous/onehot)
        返回: rs, s1s, ds, s  （均在 CPU 上）
        """
        with torch.no_grad():
            while s.dim() <= 2:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            while a.dim() <= 2:
                a = a.unsqueeze(-1)
            s = s.to(self.device)
            a = a.to(self.device)

            rs, s1s, dprob = self.ps.predict(s, a)  # dprob = done 概率
            thr = float(getattr(self, "wm_done_thr", 0.5))
            ds = (dprob > thr).float()
            return rs.detach().cpu(), s1s.detach().cpu(), ds.detach().cpu(), s.detach().cpu()

    def load_model(self, pretrained_model):
        dic = torch.load(pretrained_model, map_location=self.device)
        # 这里原实现是 dic['']，容易 KeyError；保留原样以避免改动其余逻辑
        self.load_state_dict(dic.get("", {}))


class HiddenAgent(ModelBasedAgent):
    def __init__(self, logger, device, agent_args, **kwargs):
        super().__init__(logger, device, agent_args, **kwargs)
        self.hidden_state_dim = agent_args.hidden_state_dim
        self.embedding_sizes = agent_args.embedding_sizes
        self.embedding_layers = self._init_embedding_layers()
        self.optimizer_p.add_param_group({"params": self.embedding_layers.parameters()})

    def act(self, s, requires_log=False):
        s = s.detach()
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s).detach()
        return super().act(s, requires_log)

    def get_logp(self, s, a):
        s = s.detach()
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s).detach()
        return super().get_logp(s, a)

    def updateModel(self, s, a, r, s1, d):
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s)
        if s1.size()[-1] != self.hidden_state_dim:
            s1 = self._state_embedding(s1)
        return super().updateModel(s, a, r, s1, d)

    def model_step(self, s, a):
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s)
        return super().model_step(s, a)

    def _init_embedding_layers(self):
        embedding_layers = nn.ModuleList()
        for _ in range(self.n_agent):
            embedding_layers.append(MLP(self.embedding_sizes, activation=nn.ReLU))
        return embedding_layers.to(self.device)

    def _state_embedding(self, s):
        embeddings = []
        for i in range(self.n_agent):
            embeddings.append(self.embedding_layers[i](s.select(dim=-2, index=i).to(self.device)))
        embeddings = torch.stack(embeddings, dim=-2)
        return embeddings


class DMPOAgent(ModelBasedAgent, DPPOAgent):
    def __init__(self, logger, device, agent_args, env_args, **kwargs):
        super().__init__(logger, device, agent_args, env_args, **kwargs)

    def checkConverged(self, ls_info):
        rs = [info[0] for info in ls_info]
        r_converged = len(rs) > 8 and np.mean(rs[-3:]) < np.mean(rs[:-5])
        entropies = [info[1] for info in ls_info]
        entropy_converged = len(entropies) > 8 and np.abs(np.mean(entropies[-3:]) / np.mean(entropies[:-5]) - 1) < 1e-2
        kls = [info[2] for info in ls_info]
        kl_exceeded = False
        if self.target_kl is not None:
            kls = [kl > 1.5 * self.target_kl for kl in kls]
            kl_exceeded = any(kls)
        return kl_exceeded or (r_converged and entropy_converged)


class MB_DPPOAgent_Hidden(HiddenAgent, DMPOAgent):
    def __init__(self, logger, device, agent_args, **kwargs):
        super().__init__(logger, device, agent_args, **kwargs)

    def checkConverged(self, ls_info):
        rs = [info[0] for info in ls_info]
        r_converged = len(rs) > 8 and np.mean(rs[-3:]) < np.mean(rs[:-5])
        entropies = [info[1] for info in ls_info]
        entropy_converged = len(entropies) > 8 and np.abs(np.mean(entropies[-3:]) / np.mean(entropies[:-5]) - 1) < 1e-2
        kls = [info[2] for info in ls_info]
        kl_exceeded = False
        if self.target_kl is not None:
            kls = [kl > 1.5 * self.target_kl for kl in kls]
            kl_exceeded = any(kls)
        return kl_exceeded or (r_converged and entropy_converged)
