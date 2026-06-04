# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 01:56:13 2022

@author: 86153
"""

import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
# from ray.state import actors
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
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse


class MultiCollect:
    def __init__(self, adjacency, device='cuda'):
        """
        Method: 'gather', 'reduce_mean', 'reduce_sum'.
        Adjacency: torch Tensor.
        Everything outward would be in the same device specifed in the initialization parameter.
        """
        self.device = device
        n = adjacency.size()[0]
        adjacency = adjacency > 0  # Adjacency Matrix, with size n_agent*n_agent.
        adjacency = adjacency | torch.eye(n, device=device).bool()  # include self-loop
        adjacency = adjacency.to(device)
        self.degree = adjacency.sum(dim=1)  # Number of information available to the agent.
        self.indices = []
        index_full = torch.arange(n, device=device)
        for i in range(n):
            self.indices.append(torch.masked_select(index_full, adjacency[i]))

    def gather(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim_i] for i in range(n_agent)]
        """
        return self._collect('gather', tensor)

    def reduce_mean(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [batch_size, n_agent, dim]
        """
        return self._collect('reduce_mean', tensor)

    def reduce_sum(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [batch_size, n_agent, dim]
        """
        return self._collect('reduce_sum', tensor)

    def _collect(self, method, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape:
            gather: [[batch_size, dim_i] for i in range(n_agent)]
            reduce: [batch_size, n_agent, dim]
        """
        tensor = tensor.to(self.device)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        result = []
        for i in range(n):
            if method == 'gather':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).view(b, -1))
            elif method == 'reduce_mean':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).mean(dim=1))
            else:
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).sum(dim=1))
        if method != 'gather':
            result = torch.stack(result, dim=1)
        return result


class Trajectory:
    def __init__(self, **kwargs):
        """
        数据默认形状：[T, N, ...]。
        必需键：s, a, r, s1, d, logp
        可选键：img, img1（[T, N, C, H, W]，uint8）
        """
        base = ["s", "a", "r", "s1", "d", "logp"]
        optional = [k for k in ("img", "img1") if k in kwargs]
        self._names = base + optional

        # 基础键存在性校验
        for name in base:
            if name not in kwargs:
                raise KeyError(f"Trajectory missing key: {name}")

        # 保存为 dict
        self.dict = {name: kwargs[name] for name in self._names}

        # 轨迹长度：按 s 的时间维度
        self.length = self.dict["s"].size(0)

        # 简要一致性检查（可选键如果存在，时间步要匹配）
        for optk in optional:
            if self.dict[optk].size(0) != self.length:
                raise ValueError(f"{optk}.shape[0] ({self.dict[optk].size(0)}) != s.shape[0] ({self.length})")

    def getFraction(self, length, start=None):
        """
        裁剪子序列，所有已有键一起裁剪。
        """
        if self.length < length:
            length = self.length
        start_max = self.length - length
        if start is None:
            start = torch.randint(low=0, high=start_max + 1, size=(1,)).item()
        start = min(max(start, 0), start_max)

        new_dict = {name: self.dict[name][start:start + length] for name in self._names}
        return Trajectory(**new_dict)

    def __getitem__(self, key):
        assert key in self._names, f"key {key} not in {self._names}"
        return self.dict[key]

    def get(self, key, default=None):
        return self.dict.get(key, default)

    @classmethod
    def names(cls):
        # 保持向后兼容：类方法仍返回基础键
        return ["s", "a", "r", "s1", "d", "logp"]

    def keys(self):
        # 实例级键集合（含可选键）
        return list(self._names)


class TrajectoryBuffer:
    def __init__(self, device="cuda"):
        self.device = device
        # 基础键
        self.s, self.a, self.r, self.s1, self.d, self.logp = [], [], [], [], [], []
        # 可选图像键
        self.img, self.img1 = [], []

    def store(self, s, a, r, s1, d, logp, img=None, img1=None):
        """
        逐步写入一条（或一批）时间步的数据：
            s:   [B, N, D] 或 [N, D] 或 [D]
            a:   [B, N] / [B, N, A] / [N] / [N, A]
            r:   标量/向量 -> [B, N] 视图
            s1:  与 s 同形
            d:   标量/向量 -> [B, N]（bool）
            logp:[B, N, 1] 兼容
            img:  可选，[B, N, C, H, W] / [N, C, H, W] / [C, H, W]（uint8 或会转为 uint8）
            img1: 可选，形同 img
        """
        device = self.device
        [s, r, s1, logp] = [torch.as_tensor(item, device=device, dtype=torch.float32) for item in [s, r, s1, logp]]
        d = torch.as_tensor(d, device=device)
        if d.dtype != torch.bool:
            d = d.to(torch.bool)
        a = torch.as_tensor(a, device=device)

        # 形状对齐：s 至少 [B, N, *]
        while s.dim() <= 2:
            s = s.unsqueeze(0)
        b, n, _ = s.size()

        # d: -> [B, N]
        if d.dim() == 0:
            d = d.unsqueeze(0).unsqueeze(0).expand(1, n)
        elif d.dim() == 1:
            d = d.unsqueeze(0)
        d = d[:, :n]

        # r: -> [B, N]
        if r.dim() == 0:
            r = r.unsqueeze(0).unsqueeze(0).expand(1, n)
        elif r.dim() == 1:
            r = r.unsqueeze(0)
        r = r[:, :n]

        # a/view 到 [B, N, *]（若是离散 [B,N] 会变成 [B,N,1]）
        [s, a, r, s1, d, logp] = [item.view(b, n, -1) for item in [s, a, r, s1, d, logp]]

        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.d.append(d)
        self.logp.append(logp)

        # ---- 处理图像 ----
        def _prep_image(x):
            if x is None:
                return None
            x = torch.as_tensor(x, device=device)
            # 允许 [N,C,H,W] 或 [C,H,W] 或 [B,N,C,H,W]
            if x.dim() == 3:
                # [C,H,W] -> [B=1, N, C,H,W]（广播到 N 个智能体）
                x = x.unsqueeze(0).unsqueeze(0).expand(1, n, *x.shape)
            elif x.dim() == 4:
                # [N,C,H,W] -> [B=1,N,C,H,W]
                x = x.unsqueeze(0)
            # 现在期望 [B,N,C,H,W]
            while x.dim() < 5:
                x = x.unsqueeze(0)
            if x.size(0) != b or x.size(1) != n:
                # 尝试广播 batch 维
                if x.size(0) == 1 and x.size(1) == n:
                    x = x.expand(b, n, *x.shape[2:])
                else:
                    raise ValueError(f"img shape {tuple(x.shape)} not compatible with s shape [B={b},N={n},...]")
            # 存为 uint8 节省显存
            if x.dtype != torch.uint8:
                x = x.clamp(0, 255).to(torch.uint8)
            return x

        img = _prep_image(img)
        img1 = _prep_image(img1)
        if img is not None:
            self.img.append(img)
        if img1 is not None:
            self.img1.append(img1)

    def retrieve(self, length=None):
        """
        返回一组 Trajectory（list），每个字段形状：
          基础键: [T, N, *]
          图像键: [T, N, C, H, W]（uint8）
        """
        base_names = ["s", "a", "r", "s1", "d", "logp"]
        trajs = []
        traj_all = {}
        if self.s == []:
            return []

        # 堆叠基础键：dim=1 作为时间维 T
        for name in base_names:
            traj_all[name] = torch.stack(self.__getattribute__(name), dim=1)  # [B, T, N, *]

        # 可选图像键：只有在完整记录时才纳入
        have_img = len(self.img) == len(self.s) and len(self.img) > 0
        have_img1 = len(self.img1) == len(self.s) and len(self.img1) > 0
        if have_img:
            traj_all["img"] = torch.stack(self.img, dim=1)     # [B, T, N, C, H, W]
        if have_img1:
            traj_all["img1"] = torch.stack(self.img1, dim=1)   # [B, T, N, C, H, W]

        # 拆分 batch -> 多条 Trajectory
        n = traj_all["s"].size(0)
        for i in range(n):
            traj_dict = {}
            for name, tensor in traj_all.items():
                traj_dict[name] = tensor[i]  # [T, ...]
            trajs.append(Trajectory(**traj_dict))
        return trajs


class ModelBuffer:
    def __init__(self, max_traj_num):
        self.max_traj_num = max_traj_num
        self.trajectories = []
        self.ptr = -1
        self.count = 0

    def storeTraj(self, traj):
        if self.count < self.max_traj_num:
            self.trajectories.append(traj)
            self.ptr = (self.ptr + 1) % self.max_traj_num
            self.count = min(self.count + 1, self.max_traj_num)
        else:
            self.trajectories[self.ptr] = traj
            self.ptr = (self.ptr + 1) % self.max_traj_num

    def storeTrajs(self, trajs):
        for traj in trajs:
            self.storeTraj(traj)

    def sampleTrajs(self, n_traj):
        traj_idxs = np.random.choice(range(self.count), size=(n_traj,), replace=True)
        return [self.trajectories[i] for i in traj_idxs]
